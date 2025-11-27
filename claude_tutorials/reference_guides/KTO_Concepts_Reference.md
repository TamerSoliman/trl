# KTO Concepts Reference

## Prospect Theory Foundation

### Original Kahneman-Tversky Theory

Prospect theory models how humans perceive gains and losses:

**Key insights:**
1. **Reference dependence:** Outcomes evaluated relative to reference point
2. **Loss aversion:** Losses loom larger than equivalent gains
3. **Diminishing sensitivity:** Marginal impact decreases with magnitude

**Value function:**
```
v(x) = { x^α           if x ≥ 0 (gains)
       { -λ(-x)^β      if x < 0 (losses)

where λ > 1 (loss aversion coefficient)
```

### Application to LLM Alignment

KTO adapts prospect theory to preference learning:

**Reference point:** Reference model π_ref
**Gains:** Desirable outputs (better than reference)
**Losses:** Undesirable outputs (worse than reference)

**Asymmetry in loss:**
```
L_desirable   ∝ 1 - σ(r_desirable - KL)    # Penalized by KL
L_undesirable ∝ 1 - σ(KL - r_undesirable)  # Rewarded by KL
```

This creates **different thresholds** for accepting gains vs. rejecting losses.

---

## KTO Loss Derivation

### Starting Point

We want to maximize the probability that desirable outputs are preferred:

```
max E_{(x,y,label)~D}[
    label * log P(y better than ref) +
    (1-label) * log P(y worse than ref)
]
```

### Log-Ratio as Utility

Define utility as log-ratio relative to reference:

```
r(x, y) = log π_θ(y|x) / π_ref(y|x)
        = log π_θ(y|x) - log π_ref(y|x)
```

**Interpretation:** How much better/worse is y under policy vs. reference?

### KL Calibration

To account for global policy shift, subtract mean KL divergence:

```
KL = E_{x,y'~D}[log π_θ(y'|x) / π_ref(y'|x)]
```

This prevents the loss from rewarding **all** outputs just because the policy has shifted.

### Final KTO Loss

**For desirable outputs (label=True):**
```
L_chosen = 1 - σ(β * (r - KL))
         = 1 - 1/(1 + exp(-β(r - KL)))
```

**For undesirable outputs (label=False):**
```
L_rejected = 1 - σ(β * (KL - r))
           = 1 - 1/(1 + exp(-β(KL - r)))
```

**Temperature β:** Controls sensitivity (higher β = sharper threshold)

**Code:** `kto_trainer.py:1166-1188`

---

## Mathematical Properties

### Asymmetry

Notice: L_chosen ≠ L_rejected even for same |r|

**Example with β=1, KL=0:**

| r | L_chosen | L_rejected |
|---|----------|------------|
| +2 | 0.12 | 0.88 |
| +1 | 0.27 | 0.73 |
| 0 | 0.50 | 0.50 |
| -1 | 0.73 | 0.27 |
| -2 | 0.88 | 0.12 |

**Interpretation:**
- Desirable outputs with r=+2 get low loss (0.12) - good!
- Undesirable outputs with r=+2 get high loss (0.88) - penalized!

### Gradients

```
∂L_chosen/∂θ = -β * σ(β(r-KL)) * (1-σ(β(r-KL))) * ∂r/∂θ
             = -β * sigmoid_derivative * ∂log_ratio/∂θ
```

**Key property:** Gradient vanishes when:
1. r >> KL (already very desirable)
2. r << KL (hopeless)

This implements **diminishing sensitivity** from prospect theory.

---

## KL Divergence Estimation

### Why KL is Needed

Without KL term, the loss becomes:

```
L = E[1 - σ(β * log_ratio)]
```

**Problem:** Policy can minimize loss by increasing ALL log_ratios (mode collapse or reference drift).

**Solution:** Subtract mean KL to center the distribution.

### Estimation via Mismatched Pairs

**Goal:** Estimate E_{x,y'~D}[log π(y'|x) / π_ref(y'|x)]

**Method:** For each batch, rotate completions:
```
Original pairs:       [(x₁,y₁), (x₂,y₂), (x₃,y₃), ...]
KL pairs (rotated):   [(x₁,y₂), (x₂,y₃), (x₃,y₄), ...]
```

**Why rotation works:**
- Approximates drawing y' independently from y
- Requires only data in current batch (no extra sampling)
- Computationally efficient

**Limitation:** Only approximates true KL if batch is diverse.

**Code:** `kto_trainer.py:87-95`

### Variance and Batch Size

**Small batches:** High variance in KL estimate → unstable training
**Large batches:** Low variance → stable KL, stable training

**Recommendation:** Batch size ≥ 16 for reliable KL estimation.

---

## Comparison: KTO vs DPO Loss

### DPO Loss (Pairwise)

```
L_DPO = -log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))
      = -log σ(β * (r_w - r_l))
```

**Properties:**
- Requires paired data (y_w, y_l) for same prompt x
- Symmetric: swapping y_w and y_l flips sign but same functional form
- No KL term (comparison is internal to each pair)

### KTO Loss (Binary)

```
L_chosen = 1 - σ(β * (r - KL))
L_rejected = 1 - σ(β * (KL - r))
```

**Properties:**
- Requires only binary labels (no pairing)
- Asymmetric: different functional forms for chosen/rejected
- Requires KL estimation (comparison to reference distribution)

### When KTO ≈ DPO

If we pair KTO examples manually:
```
x: "Translate: hello"
y_w: "bonjour" (desirable, r_w = +1)
y_l: "buenos" (undesirable, r_l = -1)

L_KTO = L_chosen(r_w) + L_rejected(r_l)
      = [1 - σ(β(r_w - KL))] + [1 - σ(β(KL - r_l))]

If KL ≈ 0:
      ≈ [1 - σ(βr_w)] + [1 - σ(-βr_l)]
      ≈ 2 - σ(βr_w) - σ(βr_l)
```

**Not equivalent** to DPO! KTO treats each output independently relative to KL, while DPO directly models pairwise preference.

---

## Desirable/Undesirable Weighting

### The Problem

Real datasets often have imbalanced classes:

```
Dataset: 80% desirable, 20% undesirable
```

**Without weighting:** Model overfits to majority class (predicts everything desirable).

### The Solution

Weight losses inversely proportional to class frequency:

```python
total_loss = (desirable_weight * L_desirable.mean() +
              undesirable_weight * L_undesirable.mean())
```

### Optimal Weighting (from KTO Paper)

**Recommendation:** Set weights such that:

```
desirable_weight * num_desirable ≈ undesirable_weight * num_undesirable
```

**Example:**
```
num_desirable = 800
num_undesirable = 200

Option 1: desirable_weight = 1.0, undesirable_weight = 4.0
Option 2: desirable_weight = 0.25, undesirable_weight = 1.0
```

Both achieve balance: 800 * 1.0 ≈ 200 * 4.0 = 800

**Flexibility range (from paper):** ±33% is acceptable
```
desirable_weight ∈ [baseline, baseline * 1.33]
```

**Code:** `kto_trainer.py:740-758`

---

## APO-Zero Variant

### Standard APO Loss

APO (Anchored Preference Optimization) was developed independently, targeting similar goals.

**Standard APO (paired):**
```
L_APO = -log σ(β * (r_w - r_l - margin))
```

### APO-Zero (Unpaired)

KTO includes APO's unpaired variant:

```python
# Desirable
L_chosen = 1 - σ(β * r)

# Undesirable
L_rejected = σ(β * r)
```

**Key difference from KTO:** No KL term!

**When to use:**
- Data is **definitely** better/worse than model's current behavior (not relative to fixed reference)
- Want to avoid KL computation overhead
- Training on synthetic data where model's default is clearly wrong

**Tradeoffs:**
- ✅ Simpler (no KL dataset needed)
- ✅ Faster (fewer forward passes)
- ❌ Less theoretically grounded
- ❌ More prone to reference drift

**Code:** `kto_trainer.py:1169-1188`

---

## Precomputed Reference Logprobs

### Memory Bottleneck

Standard KTO requires both models in VRAM:

```
Total VRAM = size(policy_model) + size(ref_model) + activations
           ≈ 2 * model_size (for same architecture)
```

**7B model example:**
- Policy: 14 GB
- Reference: 14 GB
- Total: ~28 GB → Requires A100 40GB

### Precomputation Strategy

**Key insight:** Reference model is frozen - its outputs are deterministic!

**Approach:**
1. Before training, compute ref_logprobs for entire dataset
2. Cache in dataset as new column
3. During training, use cached values (no ref_model forward pass)

**Memory savings:**
```
Total VRAM = size(policy_model) + activations
           ≈ 1 * model_size
```

**7B model:** ~14 GB → Fits on RTX 4090

### Implementation

```python
config = KTOConfig(precompute_ref_log_probs=True)

# First time through get_train_dataloader:
for batch in dataset:
    ref_logprobs = ref_model(batch)  # Compute once
    cache.append(ref_logprobs)

dataset = dataset.add_column("reference_logps", cache)

# During training:
policy_logprobs = policy_model(batch)
ref_logprobs = batch["reference_logps"]  # From cache!
loss = kto_loss(policy_logprobs, ref_logprobs)
```

**Code:** `kto_trainer.py:845-887`

### Caveats

**Static reference:** Once precomputed, reference logprobs never update

**When this matters:**
- Fine-tuning in stages (want to update reference between stages)
- Continual learning scenarios

**When this is fine:**
- Single training run
- Memory-constrained environments
- Reference model is truly "frozen" (e.g., base model)

---

## Hyperparameter Guide

### Beta (β)

**Range:** 0.01 - 1.0
**Default:** 0.1

**Effect:**
- **Low β (0.01):** Gentle preference shaping, large movements allowed
- **High β (1.0):** Sharp threshold, small movements only

**Recommendation:**
- Start with 0.1
- Increase if model deviates too much from reference
- Decrease if model is too conservative

### Learning Rate

**Range:** 1e-7 to 1e-5
**Default:** 5e-7 (lower than SFT!)

**Why lower?** Preference learning is more sensitive than SFT. Large updates can destabilize the policy-reference relationship.

### Desirable/Undesirable Weights

**Default:** 1.0 / 1.0

**Tuning:**
```python
# Rule of thumb
weight_ratio = num_majority_class / num_minority_class

# Then set:
majority_weight = 1.0
minority_weight = weight_ratio

# Or normalize both
desirable_weight = num_undesirable / num_total
undesirable_weight = num_desirable / num_total
```

### Batch Size

**Minimum:** 16 (for stable KL estimation)
**Recommended:** 32-64

**Effective batch size via gradient accumulation:**
```python
per_device_batch_size = 4
gradient_accumulation_steps = 8
# Effective = 4 * 8 * num_gpus
```

---

## Metrics Interpretation

### During Training

```python
{
    "loss": 0.42,
    "rewards/chosen": 2.5,
    "rewards/rejected": -1.8,
    "rewards/margins": 4.3,
    "kl": 0.05,
}
```

**Healthy training:**
- loss decreasing
- rewards/margins increasing
- kl stable and low (< 0.5)

**Warning signs:**
- kl rapidly increasing → reduce LR or increase β
- margins not increasing → data quality issues or β too high
- loss oscillating → reduce LR or increase batch size

### KL Divergence

**Typical range:** 0.01 - 0.2

**Interpretation:**
- **KL ≈ 0:** Policy very close to reference (early training)
- **KL = 0.1:** Moderate deviation (healthy)
- **KL > 0.5:** Large deviation (may overfit or drift)

**Actions:**
- If KL too high: increase β, reduce LR
- If KL stuck at 0: wait (normal early on), check data diversity

---

## Summary Table

| Feature | KTO | Notes |
|---------|-----|-------|
| **Loss type** | Binary classification | Desirable vs undesirable |
| **Key formula** | L = 1 - σ(β(r - KL)) | Asymmetric for chosen/rejected |
| **Data requirement** | Binary labels | Easier to collect than pairs |
| **Reference model** | Required (or precompute) | Can cache logprobs to save memory |
| **KL estimation** | Via mismatched pairs | Requires batch size ≥ 16 |
| **Theoretical basis** | Prospect theory | Loss aversion, reference dependence |
| **Memory** | High (2 models) or Medium (precompute) | 28 GB or 14 GB for 7B |
| **Training speed** | Slower than DPO | Extra KL forward passes |
| **Class imbalance** | Handled via weights | Automatic warnings for bad weights |

**Best for:** Binary feedback data, thumbs up/down scenarios, when pairwise comparisons unavailable
