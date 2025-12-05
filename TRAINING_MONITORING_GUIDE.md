# Training Monitoring Guide

## Current Status: Loss ~10,500

Your current loss of **~10,500** may seem high, but it could be **NORMAL** for an untrained SAM model with high-resolution masks!

## Why Initial Loss Can Be High

### 1. High-Resolution Masks
- Your masks are 1008 x 1008 = **1,016,064 pixels**
- Mask losses are computed pixel-wise
- Even small per-pixel losses add up to thousands

### 2. Untrained Model
- Random initial predictions don't match ground truth
- Classification loss is high (predicting wrong classes)
- Box loss is high (boxes in wrong locations)
- Mask loss is huge (wrong pixels segmented)

### 3. DETR-Style Architecture
- SAM uses 200 object queries
- Most queries don't match ground truth initially
- Unmatched queries contribute to loss

## Expected Loss Progression

**If training is working correctly:**

```
Epoch 1:  loss = 5,000-15,000  ← You are here
Epoch 2:  loss = 2,000-8,000   ← Should decrease
Epoch 5:  loss = 1,000-3,000   ← Steady decline
Epoch 10: loss = 100-1,000     ← Getting reasonable
Epoch 20: loss = 50-200        ← Good progress
Epoch 50+: loss = 10-50        ← Converging
```

**If training is broken:**

```
Epoch 1:  loss = 10,500
Epoch 2:  loss = 10,800  ← Increasing!
Epoch 5:  loss = 11,200  ← Still going up!
Epoch 10: loss = NaN     ← Exploded!
```

## How to Monitor

### ✅ Good Signs:

1. **Loss Decreases Over Time**
   ```
   Epoch 1: loss=10,500
   Epoch 2: loss=8,200   ← Down by 22%!
   Epoch 3: loss=6,100   ← Keeps decreasing!
   ```

2. **Validation Loss Improves**
   ```
   Epoch 1: val_loss=11,000
   Epoch 2: val_loss=8,500
   Epoch 3: val_loss=6,800
   ```

3. **No NaN or Inf**
   - All loss values are finite numbers
   - No `loss=nan` or `loss=inf`

4. **Progress Bars Move Smoothly**
   - Training doesn't hang
   - Iterations complete at steady pace

### ❌ Bad Signs:

1. **Loss Stays Constant**
   ```
   Epoch 1: loss=10,500
   Epoch 2: loss=10,490
   Epoch 3: loss=10,510  ← Not learning!
   ```

2. **Loss Increases**
   ```
   Epoch 1: loss=10,500
   Epoch 2: loss=12,000  ← Going up!
   Epoch 3: loss=15,000  ← Unstable!
   ```

3. **NaN or Inf Appears**
   ```
   Epoch 3: loss=nan  ← Numerical instability!
   ```

4. **Gradient Issues**
   - Warnings about gradient clipping
   - "Gradient overflow" messages

## What To Do Now

### 1. Let Training Run for 5-10 Epochs

**Monitor the trend:**
```bash
# Watch the training output
# Look for loss values decreasing

Epoch 1: loss=10,500
Epoch 2: loss=?       ← Wait and see
Epoch 3: loss=?       ← Is it going down?
```

### 2. Check After 5 Epochs

**If loss has decreased to <5,000:**
- ✅ Training is working!
- Continue to 50-100 epochs
- Loss should stabilize around 10-50

**If loss is still >8,000:**
- ⚠️ Possible issue
- Check if it's slowly decreasing (even 5% per epoch is progress)
- May need more epochs or lower learning rate

**If loss is increasing or NaN:**
- ❌ Training broken
- Stop and investigate
- Possible causes:
  - Learning rate too high (try 1e-6)
  - Gradient explosion (check for very large losses)
  - Data issues (corrupted annotations)

### 3. Monitor Validation Loss

**During validation:**
```
Validation: 100%|████████| 26/26 [00:21<00:00]
```

- Should also decrease over epochs
- May be higher than training loss (normal)
- Should not diverge wildly from training loss

### 4. Early Stopping Criteria

**Stop training if:**
- Loss reaches NaN
- Loss keeps increasing for 3+ epochs
- Loss stays exactly the same for 10+ epochs
- GPU runs out of memory repeatedly

**Keep training if:**
- Loss is decreasing (even slowly)
- Validation loss is improving
- No error messages appear

## Current Fix Applied

Reduced learning rate from `5e-5` to `1e-5` for better stability:
```yaml
learning_rate: 1e-5  # More conservative
```

This should help if the loss was unstable due to too large updates.

## Expected Timeline

With 700 training samples, batch size 3, and 200 epochs:
- **Iterations per epoch:** 234
- **Time per iteration:** ~3 seconds
- **Time per epoch:** ~12 minutes
- **Total training time:** ~40 hours

**Recommendation:**
- Let it train for at least 10 epochs (~2 hours)
- Check progress periodically
- Stop if loss trends in wrong direction

## What Success Looks Like

After successful training:

**Loss Values:**
```
Best epoch: loss=15-30, val_loss=20-40
```

**Inference:**
```bash
python3 inference_lora.py \
  --config configs/full_lora_config.yaml \
  --image test.jpg \
  --prompt "object" \
  --threshold 0.3

# Expected output:
Detected objects: 1-3
Max confidence: 0.5-0.9
```

## Summary

| Metric | Current | Target (After Training) |
|--------|---------|------------------------|
| Initial Loss | 10,500 | 5,000-15,000 ✅ |
| After 10 Epochs | ? | <1,000 ✅ |
| After 50 Epochs | ? | 10-50 ✅ |
| Final Loss | ? | 15-30 ✅ |
| Detections | 0 | 1-3 per image ✅ |

**Current status: POSSIBLY NORMAL** ✅

Monitor for decreasing trend over next 5-10 epochs!

---

**Created:** 2025-12-05
**Current Loss:** ~10,500
**Action:** Monitor training, verify loss decreases
