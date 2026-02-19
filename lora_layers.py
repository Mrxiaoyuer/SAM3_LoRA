"""
LoRA (Low-Rank Adaptation) implementation for SAM3 model fine-tuning.
Supports selective application to different transformer components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Set, Tuple
import math


class MultiheadAttentionLoRA(nn.Module):
    """
    Custom MultiheadAttention with packed QKV projection and SDPA.

    This replaces nn.MultiheadAttention so LoRA can be injected into:
    - qkv: packed Q/K/V projection
    - out_proj: output projection

    For self-attention (query is key is value), Q/K/V are projected in one GEMM.
    Attention is computed with torch.nn.functional.scaled_dot_product_attention
    when attention weights are not requested.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = False,
        # Copy weights from existing MHA
        in_proj_weight: Optional[torch.Tensor] = None,
        in_proj_bias: Optional[torch.Tensor] = None,
        out_proj_weight: Optional[torch.Tensor] = None,
        out_proj_bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first
        self.dropout = dropout

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Packed QKV projection: output layout is [Q | K | V]
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Initialize from existing MHA weights if provided
        if in_proj_weight is not None:
            self.qkv.weight.data = in_proj_weight.clone()

        if in_proj_bias is not None:
            self.qkv.bias.data = in_proj_bias.clone()

        if out_proj_weight is not None:
            self.out_proj.weight.data = out_proj_weight.clone()

        if out_proj_bias is not None:
            self.out_proj.bias.data = out_proj_bias.clone()

        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _expand_attn_mask(
        self,
        attn_mask: torch.Tensor,
        batch_size: int,
        tgt_len: int,
        src_len: int,
    ) -> torch.Tensor:
        """Normalize attn_mask to shape broadcastable with (B, H, L, S)."""
        if attn_mask.dim() == 2:
            # (L, S) -> (1, 1, L, S)
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
        elif attn_mask.dim() == 3:
            # (B, L, S) -> (B, 1, L, S)
            if attn_mask.shape[0] == batch_size:
                attn_mask = attn_mask.unsqueeze(1)
            # (B*H, L, S) -> (B, H, L, S)
            elif attn_mask.shape[0] == batch_size * self.num_heads:
                attn_mask = attn_mask.view(batch_size, self.num_heads, tgt_len, src_len)
            else:
                # Unknown 3D format, let broadcasting try to handle it.
                attn_mask = attn_mask.unsqueeze(1)
        elif attn_mask.dim() != 4:
            raise ValueError(f"Unsupported attn_mask rank: {attn_mask.dim()}")

        expected_shape = (batch_size, self.num_heads, tgt_len, src_len)
        if attn_mask.shape != expected_shape:
            attn_mask = attn_mask.expand(expected_shape)
        return attn_mask

    def _build_sdpa_mask(
        self,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        batch_size: int,
        tgt_len: int,
        src_len: int,
        dtype: torch.dtype,
        device: torch.device,
        is_causal: bool,
    ) -> Tuple[Optional[torch.Tensor], bool]:
        """
        Build mask for scaled_dot_product_attention.

        - nn.MultiheadAttention semantics: bool mask True = blocked
        - SDPA semantics: bool mask True = allowed
        """
        sdpa_attn_mask: Optional[torch.Tensor] = None

        if attn_mask is not None:
            attn_mask = self._expand_attn_mask(attn_mask, batch_size, tgt_len, src_len)
            if attn_mask.dtype == torch.bool:
                # Convert blocked-mask to keep-mask for SDPA.
                sdpa_attn_mask = ~attn_mask
            else:
                sdpa_attn_mask = attn_mask.to(dtype=dtype)

        if key_padding_mask is not None:
            # key_padding_mask: (B, S), True means blocked.
            keep_mask = ~key_padding_mask.unsqueeze(1).unsqueeze(2)
            if sdpa_attn_mask is None:
                sdpa_attn_mask = keep_mask
            elif sdpa_attn_mask.dtype == torch.bool:
                sdpa_attn_mask = sdpa_attn_mask & keep_mask
            else:
                padding_bias = torch.zeros(
                    (batch_size, 1, tgt_len, src_len),
                    dtype=dtype,
                    device=device,
                ).masked_fill(~keep_mask, float("-inf"))
                sdpa_attn_mask = sdpa_attn_mask + padding_bias

        use_causal_flag = is_causal and sdpa_attn_mask is None
        if is_causal and sdpa_attn_mask is not None:
            causal_keep = torch.ones((tgt_len, src_len), dtype=torch.bool, device=device).tril()
            causal_keep = causal_keep.unsqueeze(0).unsqueeze(0)
            if sdpa_attn_mask.dtype == torch.bool:
                sdpa_attn_mask = sdpa_attn_mask & causal_keep
            else:
                causal_bias = torch.zeros(
                    (1, 1, tgt_len, src_len),
                    dtype=dtype,
                    device=device,
                ).masked_fill(~causal_keep, float("-inf"))
                sdpa_attn_mask = sdpa_attn_mask + causal_bias
            use_causal_flag = False

        return sdpa_attn_mask, use_causal_flag

    def _project_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        same_qk: bool,
        same_kv: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Project Q/K/V from packed qkv layer while preserving LoRA forward path.

        Fast path (self-attention): one GEMM.
        Cross-attention path: 2-3 GEMMs depending on key/value sharing.
        """
        d = self.embed_dim
        if same_qk and same_kv:
            qkv = self.qkv(query)
            return qkv[..., :d], qkv[..., d:2 * d], qkv[..., 2 * d:]

        q = self.qkv(query)[..., :d]
        if same_kv:
            kv = self.qkv(key)
            return q, kv[..., d:2 * d], kv[..., 2 * d:]

        k = self.qkv(key)[..., d:2 * d]
        v = self.qkv(value)[..., 2 * d:]
        return q, k, v

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass using packed QKV projection and SDPA.
        """
        same_qk = query is key
        same_kv = key is value

        # Handle batch_first
        if self.batch_first:
            # Input: (batch, seq, embed_dim)
            batch_size, tgt_len, _ = query.shape
            src_len = key.shape[1]
        else:
            # Input: (seq, batch, embed_dim)
            tgt_len, batch_size, _ = query.shape
            src_len = key.shape[0]
            # Convert to batch_first for easier processing
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        # Project Q, K, V from packed qkv layer.
        q, k, v = self._project_qkv(query, key, value, same_qk=same_qk, same_kv=same_kv)

        # Reshape for multi-head attention
        # (batch, seq, embed_dim) -> (batch, num_heads, seq, head_dim)
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        sdpa_attn_mask, use_causal_flag = self._build_sdpa_mask(
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            batch_size=batch_size,
            tgt_len=tgt_len,
            src_len=src_len,
            dtype=q.dtype,
            device=q.device,
            is_causal=is_causal,
        )

        attn_weights = None
        if need_weights:
            # Fallback path when attention maps are requested.
            scale = 1.0 / math.sqrt(self.head_dim)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
            if use_causal_flag:
                causal_keep = torch.ones((tgt_len, src_len), dtype=torch.bool, device=q.device).tril()
                attn_weights = attn_weights.masked_fill(
                    ~causal_keep.unsqueeze(0).unsqueeze(0),
                    float("-inf"),
                )
            if sdpa_attn_mask is not None:
                if sdpa_attn_mask.dtype == torch.bool:
                    # SDPA bool mask True means keep.
                    attn_weights = attn_weights.masked_fill(~sdpa_attn_mask, float("-inf"))
                else:
                    attn_weights = attn_weights + sdpa_attn_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout_layer(attn_weights)
            attn_output = torch.matmul(attn_weights, v)
        else:
            # Fast path: lets PyTorch dispatch Flash/Mem-Efficient SDPA kernels.
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=sdpa_attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=use_causal_flag,
            )

        # Reshape back: (batch, num_heads, seq, head_dim) -> (batch, seq, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.embed_dim)

        # Output projection - LoRA is applied here
        attn_output = self.out_proj(attn_output)

        # Convert back if not batch_first
        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        if need_weights:
            if average_attn_weights:
                attn_weights = attn_weights.mean(dim=1)
            return attn_output, attn_weights
        else:
            return attn_output, None


class LoRALayer(nn.Module):
    """
    LoRA layer that replaces a linear layer with low-rank adaptation.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: Rank of the low-rank matrices (r in the paper)
        alpha: Scaling factor (typically set to rank)
        dropout: Dropout probability for LoRA weights
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply LoRA transformation: x @ (A @ B) * scaling
        """
        # x shape: (..., in_features)
        lora_out = self.dropout(x) @ self.lora_A @ self.lora_B
        return lora_out * self.scaling


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.
    Combines the original frozen linear layer with a LoRA layer.

    Exposes weight/bias properties to maintain compatibility with modules
    that access these attributes directly (e.g., nn.MultiheadAttention).
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Freeze the original layer
        self.original_layer = original_layer
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # Store original layer attributes for compatibility
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features

        # Create LoRA layer
        self.lora = LoRALayer(
            in_features=original_layer.in_features,
            out_features=original_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

    @property
    def weight(self) -> torch.Tensor:
        """Proxy to original layer's weight for compatibility with nn.MultiheadAttention."""
        return self.original_layer.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        """Proxy to original layer's bias for compatibility with nn.MultiheadAttention."""
        return self.original_layer.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: original output + LoRA output
        """
        return self.original_layer(x) + self.lora(x)


class LoRAConfig:
    """
    Configuration for LoRA application to SAM3 model.

    Args:
        rank: Rank of LoRA matrices
        alpha: Scaling factor
        dropout: Dropout probability
        target_modules: Which modules to apply LoRA to
        apply_to_vision_encoder: Whether to apply LoRA to vision encoder
        apply_to_text_encoder: Whether to apply LoRA to text encoder
        apply_to_geometry_encoder: Whether to apply LoRA to geometry encoder
        apply_to_detr_encoder: Whether to apply LoRA to DETR encoder
        apply_to_detr_decoder: Whether to apply LoRA to DETR decoder
        apply_to_mask_decoder: Whether to apply LoRA to mask decoder
    """

    def __init__(
        self,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
        # Component-level control
        apply_to_vision_encoder: bool = True,
        apply_to_text_encoder: bool = True,
        apply_to_geometry_encoder: bool = False,
        apply_to_detr_encoder: bool = True,
        apply_to_detr_decoder: bool = True,
        apply_to_mask_decoder: bool = False,
    ):
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout

        # Default target modules: attention and MLP projections across architectures
        # - q_proj, k_proj, v_proj: legacy config names; mapped to packed "qkv" in MHA
        # - qkv: Fused Q/K/V projection (ViT-style and packed MHA replacement)
        # - proj: Output projection in vision backbone (different from out_proj)
        # - out_proj: Output projection in MultiheadAttention
        # - c_fc, c_proj: MLP layers in CLIP-style language backbone
        # - linear1, linear2: FFN layers in transformer encoder/decoder
        if target_modules is None:
            target_modules = [
                # Standard attention projections
                "q_proj", "k_proj", "v_proj", "out_proj",
                # Vision backbone (ViT-style)
                "qkv",  # Fused Q/K/V projection
                "proj",  # Output projection (note: will also match out_proj, c_proj)
                "fc1", "fc2",  # MLP layers in vision backbone
                # Language backbone (CLIP-style) MLP
                "c_fc", "c_proj",
                # Transformer FFN layers
                "linear1", "linear2",
            ]
        self.target_modules = set(target_modules)

        # Component flags
        self.apply_to_vision_encoder = apply_to_vision_encoder
        self.apply_to_text_encoder = apply_to_text_encoder
        self.apply_to_geometry_encoder = apply_to_geometry_encoder
        self.apply_to_detr_encoder = apply_to_detr_encoder
        self.apply_to_detr_decoder = apply_to_detr_decoder
        self.apply_to_mask_decoder = apply_to_mask_decoder

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": list(self.target_modules),
            "apply_to_vision_encoder": self.apply_to_vision_encoder,
            "apply_to_text_encoder": self.apply_to_text_encoder,
            "apply_to_geometry_encoder": self.apply_to_geometry_encoder,
            "apply_to_detr_encoder": self.apply_to_detr_encoder,
            "apply_to_detr_decoder": self.apply_to_detr_decoder,
            "apply_to_mask_decoder": self.apply_to_mask_decoder,
        }


def apply_lora_to_model(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """
    Apply LoRA to specified modules in the SAM3 model.

    This function:
    1. Replaces nn.MultiheadAttention with MultiheadAttentionLoRA (packed qkv + out_proj)
    2. Applies LoRA to all matching Linear layers

    Args:
        model: SAM3 model to apply LoRA to
        config: LoRA configuration

    Returns:
        Model with LoRA applied
    """

    # CRITICAL: Freeze all base model parameters first
    for param in model.parameters():
        param.requires_grad = False

    def should_apply_lora_to_component(module_name: str) -> bool:
        """Check component-level flags to determine if we should apply LoRA."""
        if ("vision_encoder" in module_name or "vision_backbone" in module_name) and not config.apply_to_vision_encoder:
            return False
        if ("text_encoder" in module_name or "language_backbone" in module_name) and not config.apply_to_text_encoder:
            return False
        if "geometry_encoder" in module_name and not config.apply_to_geometry_encoder:
            return False
        if ("detr_encoder" in module_name or "transformer.encoder" in module_name) and not config.apply_to_detr_encoder:
            return False
        if ("detr_decoder" in module_name or "transformer.decoder" in module_name) and not config.apply_to_detr_decoder:
            return False
        if "mask_decoder" in module_name and not config.apply_to_mask_decoder:
            return False
        return True

    def should_apply_lora(module_name: str) -> bool:
        """Determine if LoRA should be applied to this module."""
        if not should_apply_lora_to_component(module_name):
            return False

        # Check if module name matches target modules
        module_basename = module_name.split('.')[-1]

        # Backward-compatibility mapping:
        # packed MHA projection is named "qkv", but configs may still list q/k/v.
        if module_basename == "qkv":
            if "qkv" in config.target_modules:
                return True
            if {"q_proj", "k_proj", "v_proj"} & config.target_modules:
                return True

        # Direct basename match (e.g., "qkv", "proj", "linear1", etc.)
        if module_basename in config.target_modules:
            return True

        # Also check for substring match for flexibility
        for target in config.target_modules:
            if target in module_basename:
                return True

        return False

    # Track replacements
    mha_replaced = []
    lora_modules_applied = []

    # STEP 1: Replace nn.MultiheadAttention with MultiheadAttentionLoRA
    # This enables LoRA to be applied to Q, K, V, and out_proj inside MHA
    mha_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            if should_apply_lora_to_component(name):
                mha_to_replace.append((name, module))

    for name, mha in mha_to_replace:
        # Get parent module and attribute name
        *parent_path, attr_name = name.split('.')
        parent = model
        for p in parent_path:
            parent = getattr(parent, p)

        # Create replacement with separate Q, K, V projections
        new_mha = MultiheadAttentionLoRA(
            embed_dim=mha.embed_dim,
            num_heads=mha.num_heads,
            dropout=mha.dropout,
            bias=mha.in_proj_bias is not None,
            batch_first=mha.batch_first,
            in_proj_weight=mha.in_proj_weight,
            in_proj_bias=mha.in_proj_bias,
            out_proj_weight=mha.out_proj.weight,
            out_proj_bias=mha.out_proj.bias if mha.out_proj.bias is not None else None,
        )

        # Freeze the new MHA parameters
        for param in new_mha.parameters():
            param.requires_grad = False

        setattr(parent, attr_name, new_mha)
        mha_replaced.append(name)

    print(f"Replaced {len(mha_replaced)} nn.MultiheadAttention modules with MultiheadAttentionLoRA")

    # STEP 2: Apply LoRA to all matching Linear layers
    # For replaced MHA modules this includes qkv and out_proj.
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and should_apply_lora(name):
            # Get parent module and attribute name
            *parent_path, attr_name = name.split('.')
            parent = model
            for p in parent_path:
                parent = getattr(parent, p)

            # Replace with LoRA linear
            lora_linear = LoRALinear(
                module,
                rank=config.rank,
                alpha=config.alpha,
                dropout=config.dropout,
            )
            setattr(parent, attr_name, lora_linear)
            lora_modules_applied.append(name)

    print(f"Applied LoRA to {len(lora_modules_applied)} modules:")
    for module_name in lora_modules_applied[:15]:  # Show first 15
        print(f"  - {module_name}")
    if len(lora_modules_applied) > 15:
        print(f"  ... and {len(lora_modules_applied) - 15} more")

    return model


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    Get all LoRA parameters from the model.

    Args:
        model: Model with LoRA layers

    Returns:
        List of LoRA parameters
    """
    lora_params = []
    for module in model.modules():
        if isinstance(module, LoRALayer):
            lora_params.extend([module.lora_A, module.lora_B])
    return lora_params


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count total and trainable parameters in the model.

    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_percentage": 100 * trainable_params / total_params if total_params > 0 else 0,
    }


def save_lora_weights(model: nn.Module, save_path: str):
    """
    Save only LoRA weights (not the full model).

    Args:
        model: Model with LoRA layers
        save_path: Path to save LoRA weights
    """
    lora_state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            lora_state_dict[f"{name}.lora_A"] = module.lora_A
            lora_state_dict[f"{name}.lora_B"] = module.lora_B

    torch.save(lora_state_dict, save_path)
    print(f"Saved LoRA weights to {save_path}")


def load_lora_weights(model: nn.Module, load_path: str):
    """
    Load LoRA weights into a model.

    Args:
        model: Model with LoRA layers
        load_path: Path to LoRA weights
    """
    lora_state_dict = torch.load(load_path)
    model.load_state_dict(lora_state_dict, strict=False)
    print(f"Loaded LoRA weights from {load_path}")
