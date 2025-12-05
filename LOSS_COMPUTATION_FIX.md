# Loss Computation Fix - Using Sam3LossWrapper

## Problem

Training loss was ~10,000 and barely decreasing. After examining the original SAM3 training code, I found three critical issues:

### 1. **Incorrect Weight Values**

**Original (Wrong)**:
```python
weight_dict = {
    "loss_ce": 2.0,
    "loss_bbox": 5.0,
    "loss_giou": 2.0,
    "loss_mask": 5.0,
    "loss_dice": 5.0
}
```

**Correct (from SAM3 training config)**:
```python
weight_dict = {
    "loss_ce": 20.0,           # 10x higher
    "presence_loss": 20.0,
    "loss_bbox": 5.0,
    "loss_giou": 2.0,
    "loss_mask": 200.0,        # 40x higher!
    "loss_dice": 10.0          # 2x higher
}
```

### 2. **Incorrect Loss Computation Method**

**Original (Wrong)**:
```python
# Manually calling get_loss() and weighting
l_cls = self.criterion_cls.get_loss(outputs, targets, indices, num_boxes)
l_box = self.criterion_box.get_loss(outputs, targets, indices, num_boxes)
l_mask = self.criterion_mask.get_loss(outputs, targets, indices, num_boxes)

losses.update(l_cls)
losses.update(l_box)
losses.update(l_mask)

total_loss = 0
for k, v in losses.items():
    if k in weight_dict:
        total_loss += v * weight_dict[k]
```

**Correct (using Sam3LossWrapper)**:
```python
# Use Sam3LossWrapper which handles everything
loss_dict = self.loss_wrapper(outputs_list, find_targets)
total_loss = loss_dict[CORE_LOSS_KEY]
```

### 3. **Missing Sam3LossWrapper Features**

The original approach missed:
- **Auxiliary outputs**: DETR-style models produce aux_outputs from intermediate decoder layers
- **Proper num_boxes normalization**: The wrapper handles global/local/none normalization modes
- **Loss function __call__()**: Loss functions have `__call__()` that handles weighting internally

## The Fix

### Step 1: Import Sam3LossWrapper

```python
from sam3.train.loss.sam3_loss import Sam3LossWrapper
```

### Step 2: Create Loss Functions with Correct Weights

```python
loss_fns = [
    Boxes(weight_dict={
        "loss_bbox": 5.0,
        "loss_giou": 2.0
    }),
    IABCEMdetr(
        pos_weight=10.0,
        weight_dict={
            "loss_ce": 20.0,
            "presence_loss": 20.0
        },
        pos_focal=False,
        alpha=0.25,
        gamma=2,
        use_presence=True,
        pad_n_queries=200,
    ),
    Masks(
        weight_dict={
            "loss_mask": 200.0,  # Correct weight!
            "loss_dice": 10.0
        },
        focal_alpha=0.25,
        focal_gamma=2.0,
        compute_aux=False
    )
]
```

### Step 3: Instantiate Sam3LossWrapper

```python
self.loss_wrapper = Sam3LossWrapper(
    loss_fns_find=loss_fns,
    matcher=self.matcher,
    normalization="local",  # Use local (no distributed training)
    normalize_by_valid_object_num=False,
)
```

### Step 4: Update Training Loop

**Before**:
```python
outputs = outputs_list[-1]  # Only last output
targets = self.model.back_convert(input_batch.find_targets[0])
indices = self.matcher(outputs, targets)
# Manual loss computation...
```

**After**:
```python
outputs_list = self.model(input_batch)  # Full SAM3Output
find_targets = [self.model.back_convert(t) for t in input_batch.find_targets]
loss_dict = self.loss_wrapper(outputs_list, find_targets)
total_loss = loss_dict[CORE_LOSS_KEY]
```

## Expected Results

**Before fix**:
```
Epoch 2: loss=10,090
Epoch 3: loss=10,087
Epoch 4: loss=10,085  ← Barely decreasing
```

**After fix**:
```
Epoch 1: loss=2.0-5.0    ← Much lower initial loss
Epoch 2: loss=1.0-3.0    ← Steady decrease
Epoch 5: loss=0.5-1.5    ← Converging
Epoch 10: loss=0.3-0.8   ← Good progress
```

## Why This Matters

1. **Mask loss is the dominant component**: With 1008x1008 masks, mask losses are naturally large. The high weight (200.0) is designed to balance all loss components properly.

2. **Loss functions handle weighting internally**: The `LossWithWeights.__call__()` method computes a "core_loss" that's already weighted. Manually weighting again would be wrong.

3. **Auxiliary outputs improve training**: DETR models benefit from loss on intermediate layers, which Sam3LossWrapper handles automatically.

## References

- Original SAM3 config: `sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml`
- Loss wrapper: `sam3/train/loss/sam3_loss.py`
- Loss functions: `sam3/train/loss/loss_fns.py`

---

**Created:** 2025-12-05
**Issue:** Loss ~10,000 and barely decreasing
**Fix:** Use Sam3LossWrapper with correct weights from original SAM3 training
