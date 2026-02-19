"""
LoRA (Low-Rank Adaptation) Layer Implementation for SAM3

This module implements LoRA layers that can be injected into transformer models
for efficient fine-tuning.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
            batch_size, tgt_len, _ = query.shape
            src_len = key.shape[1]
        else:
            tgt_len, batch_size, _ = query.shape
            src_len = key.shape[0]
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        # Project Q, K, V from packed qkv layer.
        q, k, v = self._project_qkv(query, key, value, same_qk=same_qk, same_kv=same_kv)

        # Reshape for multi-head attention
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

        # Reshape back
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
    LoRA layer that adds low-rank adaptation to a linear transformation.

    This implements the LoRA technique from "LoRA: Low-Rank Adaptation of Large Language Models"
    where a linear layer's weight W is augmented with a low-rank update: W' = W + BA
    where B is (out_features x r) and A is (r x in_features), with r << min(in_features, out_features)

    Args:
        in_features: Size of input features
        out_features: Size of output features
        rank: Rank of the low-rank matrices (r)
        alpha: Scaling factor for LoRA (controls the magnitude of updates)
        dropout: Dropout probability for LoRA path
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize LoRA parameters using Kaiming uniform initialization for A and zero for B."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LoRA layer.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        # Apply dropout to input
        x_dropout = self.dropout(x)

        # Low-rank adaptation: (x @ A^T) @ B^T
        result = F.linear(x_dropout, self.lora_B @ self.lora_A)

        # Scale by alpha/r
        return result * self.scaling

    def merge_weights(self) -> torch.Tensor:
        """
        Merge LoRA weights into a single weight matrix.

        Returns:
            Merged weight matrix of shape (out_features, in_features)
        """
        return (self.lora_B @ self.lora_A) * self.scaling


class LinearWithLoRA(nn.Module):
    """
    Linear layer with LoRA adaptation.

    This wraps an existing nn.Linear layer and adds a LoRA layer in parallel.
    The original linear layer's weights are frozen.

    Args:
        linear: Original linear layer to adapt
        rank: Rank of LoRA matrices
        alpha: LoRA scaling factor
        dropout: Dropout probability for LoRA path
    """

    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Store original linear layer and freeze it
        self.linear = linear
        for param in self.linear.parameters():
            param.requires_grad = False

        # Store dimensions for compatibility
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        # Add LoRA adaptation
        self.lora = LoRALayer(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

    @property
    def weight(self):
        """Expose weight for compatibility with nn.Linear."""
        return self.linear.weight

    @property
    def bias(self):
        """Expose bias for compatibility with nn.Linear."""
        return self.linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: original linear transformation + LoRA adaptation.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        # Original linear transformation (frozen)
        result = self.linear(x)

        # Add LoRA adaptation
        result = result + self.lora(x)

        return result

    def merge_weights(self) -> nn.Linear:
        """
        Merge LoRA weights into the original linear layer.

        Returns:
            New linear layer with merged weights
        """
        merged_weight = self.linear.weight.data + self.lora.merge_weights()

        new_linear = nn.Linear(
            self.linear.in_features,
            self.linear.out_features,
            bias=self.linear.bias is not None,
        )
        new_linear.weight.data = merged_weight
        if self.linear.bias is not None:
            new_linear.bias.data = self.linear.bias.data.clone()

        return new_linear
