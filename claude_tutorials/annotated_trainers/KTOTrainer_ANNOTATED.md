# KTOTrainer: Kahneman-Tversky Optimization

**Source:** `trl/trainer/kto_trainer.py` (1500+ lines)
**Purpose:** Align models using binary preference data (desirable/undesirable) without pairwise comparisons

---

## Overview

KTOTrainer implements **Kahneman-Tversky Optimization**, a preference learning algorithm inspired by Kahneman and Tversky's prospect theory. Unlike DPO/reward modeling which require pairwise comparisons, KTO works with **binary labels** (thumbs up/down).

### Key Insight

**Traditional:** "Response A is better than response B for prompt X" (pairwise)
**KTO:** "Response Y is desirable/undesirable for prompt X" (binary)

This makes data collection much easier - you don't need to generate or compare multiple responses per prompt.

---

## Core Concept: KTO Loss

### Mathematical Foundation

KTO models human preferences using **prospect theory**, which accounts for asymmetric value functions for gains and losses.

**Loss for desirable (chosen) outputs:**
```
L_chosen = 1 - σ(β * (log π_θ(y|x) / π_ref(y|x) - KL))
         = 1 - σ(β * (log_ratio - KL))

where:
  σ = sigmoid function
  β = temperature parameter (typically 0.1)
  log_ratio = log(policy) - log(reference)
  KL = estimated KL divergence
```

**Loss for undesirable (rejected) outputs:**
```
L_rejected = 1 - σ(β * (KL - log π_θ(y'|x) / π_ref(y'|x)))
           = 1 - σ(β * (KL - log_ratio))
```

**Code location:** `kto_trainer.py:1126-1201`

**Key asymmetry:** Chosen and rejected losses are **NOT symmetric** - the KL term appears with opposite signs. This reflects prospect theory's observation that losses loom larger than gains.

---

## Dataset Format

### Binary Preference Format

KTO requires datasets with **binary labels** instead of pairwise comparisons:

```python
{
    "prompt": "What is the capital of France?",
    "completion": "The capital of France is Paris.",
    "label": True  # True = desirable, False = undesirable
}
```

**Alternative conversational format:**
```python
{
    "prompt": [
        {"role": "user", "content": "What is 2+2?"}
    ],
    "completion": [
        {"role": "assistant", "content": "4"}
    ],
    "label": True
}
```

### Converting Pairwise to Binary

If you have pairwise preference data, TRL can automatically convert it:

```python
from trl import KTOTrainer

# This dataset has "chosen" and "rejected" columns
dataset = load_dataset("Anthropic/hh-rlhf")

# KTOTrainer automatically "unpairs" it into binary format
trainer = KTOTrainer(
    model=model,
    train_dataset=dataset,  # Automatically converted
    ...
)
```

**How it works:** Each pair (prompt, chosen, rejected) becomes two examples:
1. (prompt, chosen, label=True)
2. (prompt, rejected, label=False)

**Code:** `data_utils.py:maybe_unpair_preference_dataset()`

---

## KL Divergence Estimation

### The KL Dataset

KTO requires estimating KL[π_θ || π_ref] to calibrate the loss. This is done by creating **mismatched pairs**:

```python
def _get_kl_dataset(batch):
    """
    Creates mismatched pairs by rotating completions by +1.

    Original:
      [(prompt_1, completion_1), (prompt_2, completion_2), ...]

    KL dataset:
      [(prompt_1, completion_2), (prompt_2, completion_3), ...]
    """
    batch["answer_input_ids"] = [batch["answer_input_ids"][-1]] + batch["answer_input_ids"][:-1]
    return batch
```

**Code location:** `kto_trainer.py:87-95`

**Why mismatched?** We need E_x,y'~D[KL(π_θ(·|x) || π_ref(·|x))] where y' is drawn independently from y. Rotating completions approximates this.

### KL Computation

```python
KL_logps_policy = model(prompt_i, completion_j)  # i ≠ j
KL_logps_ref = ref_model(prompt_i, completion_j)

kl = (KL_logps_policy - KL_logps_ref).mean()
```

**Code location:** `kto_trainer.py:1156-1160`

---

## Loss Computation Pipeline

### Step 1: Forward Pass

```python
def forward(self, model, batch):
    # Compute KL log probs (mismatched pairs)
    KL_logps = self._compute_kl_logps(model, batch)

    # Compute completion log probs (matched pairs)
    outputs = model(batch["completion_input_ids"],
                   attention_mask=batch["completion_attention_mask"])
    completion_logps = self.get_batch_logps(outputs.logits, batch["completion_labels"])

    # Split into chosen/rejected based on labels
    chosen_idx = [i for i, label in enumerate(batch["label"]) if label is True]
    rejected_idx = [i for i, label in enumerate(batch["label"]) if label is False]

    chosen_logps = completion_logps[chosen_idx]
    rejected_logps = completion_logps[rejected_idx]

    return chosen_logps, rejected_logps, KL_logps
```

**Code location:** `kto_trainer.py:1075-1124`

**Note:** Batches can have **variable numbers** of chosen vs rejected examples. Some batches might be all chosen or all rejected.

### Step 2: KTO Loss

```python
def kto_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    policy_KL_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    reference_KL_logps,
):
    # Estimate KL divergence
    kl = (policy_KL_logps - reference_KL_logps).mean().clamp(min=0)

    # Chosen losses (if any chosen in batch)
    if len(policy_chosen_logps) > 0:
        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        chosen_losses = 1 - F.sigmoid(self.beta * (chosen_logratios - kl))
        chosen_rewards = self.beta * chosen_logratios.detach()

    # Rejected losses (if any rejected in batch)
    if len(policy_rejected_logps) > 0:
        rejected_logratios = policy_rejected_logps - reference_rejected_logps
        rejected_losses = 1 - F.sigmoid(self.beta * (kl - rejected_logratios))
        rejected_rewards = self.beta * rejected_logratios.detach()

    # Combine with class weights
    losses = torch.cat([
        self.desirable_weight * chosen_losses,
        self.undesirable_weight * rejected_losses
    ])

    return losses.mean(), chosen_rewards, rejected_rewards, kl
```

**Code location:** `kto_trainer.py:1126-1201`

---

## Handling Class Imbalance

### Desirable/Undesirable Weights

Real-world data often has imbalanced desirable/undesirable examples. KTO handles this with loss weights:

```python
config = KTOConfig(
    desirable_weight=1.0,     # Weight for desirable examples
    undesirable_weight=1.0,   # Weight for undesirable examples
)
```

**Optimal weight selection (from KTO paper):**

If `num_desirable != num_undesirable`, the paper recommends:

```
desirable_weight ∈ [λ_l, λ_u] where:
  λ_l = (num_undesirable / num_desirable) * undesirable_weight
  λ_u = λ_l * 1.33

OR

undesirable_weight ∈ [λ_l / 1.33, λ_l] where:
  λ_l = (num_desirable / num_undesirable) * desirable_weight
```

**Code location:** `kto_trainer.py:740-758`

KTOTrainer will **warn you** if your weights are outside the recommended range.

---

## Memory Optimization: Precomputed Reference Logprobs

### The Problem

KTO requires both policy and reference models in memory, doubling VRAM usage.

### The Solution

**Precompute reference log probabilities** once, then train without loading the reference model:

```python
config = KTOConfig(
    precompute_ref_log_probs=True,  # Compute once, cache in dataset
)

trainer = KTOTrainer(
    model=model,
    ref_model=None,  # Reference model optional when precomputing
    args=config,
    train_dataset=dataset,
)

# On first epoch, reference logprobs are computed and cached
# Subsequent epochs use cached values
trainer.train()
```

**How it works:**

1. Before training starts, compute ref_logprobs for entire dataset
2. Add as new column to dataset: `dataset["reference_logps"]`
3. During training, use cached values instead of forward pass through ref_model

**Code location:** `kto_trainer.py:845-887` (train), `kto_trainer.py:889-942` (eval)

**Memory savings:** ~50% VRAM reduction (store only policy model)

**Drawback:** Reference logprobs are static - if you update the reference model, must recompute.

---

## Training Configuration

### KTOConfig

```python
from trl import KTOConfig

config = KTOConfig(
    # Core KTO parameters
    beta=0.1,                      # Temperature (higher = less deviation from ref)
    desirable_weight=1.0,          # Weight for desirable examples
    undesirable_weight=1.0,        # Weight for undesirable examples

    # Loss variant
    loss_type="kto",               # "kto" or "apo_zero_unpaired"

    # Sequence lengths
    max_length=1024,               # Max total length (prompt + completion)
    max_prompt_length=512,         # Max prompt length

    # Memory optimization
    precompute_ref_log_probs=False,  # Cache reference logprobs?
    gradient_checkpointing=True,     # Save memory during backward

    # Training
    learning_rate=1e-6,            # Lower than SFT (default)
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,

    # Evaluation
    eval_strategy="steps",
    eval_steps=500,
    generate_during_eval=False,    # Generate samples during eval?

    # Optimization
    bf16=True,                     # Mixed precision

    output_dir="./kto_model",
)
```

**Code location:** `trl/trainer/kto_config.py`

---

## Production Example

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import KTOTrainer, KTOConfig
from peft import LoraConfig

# Load model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")

# Load dataset (binary labels: True/False)
dataset = load_dataset("your-org/kto-dataset", split="train")

# Configure training
config = KTOConfig(
    output_dir="./kto_output",
    beta=0.1,
    desirable_weight=1.0,
    undesirable_weight=1.0,
    max_length=1024,
    max_prompt_length=512,
    learning_rate=5e-7,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    gradient_checkpointing=True,
    bf16=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=500,
)

# Optional: LoRA for efficiency
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

# Initialize trainer
trainer = KTOTrainer(
    model=model,
    ref_model=None,  # Will be created automatically from model
    args=config,
    train_dataset=dataset,
    eval_dataset=dataset.select(range(1000)),
    processing_class=tokenizer,
    peft_config=peft_config,
)

# Train
trainer.train()

# Save
trainer.save_model("./kto_final")
```

---

## APO-Zero Variant

KTO also supports **APO-zero** (Anchored Preference Optimization), an unpaired variant:

```python
config = KTOConfig(
    loss_type="apo_zero_unpaired",  # Instead of "kto"
    beta=0.1,
)
```

**APO-zero loss:**
```python
# Chosen
chosen_losses = 1 - σ(β * log_ratio)

# Rejected
rejected_losses = σ(β * log_ratio)
```

**Key difference:** No KL term - simpler but potentially less stable.

**When to use:**
- When you believe chosen outputs are better than model's **default** behavior (not relative to KL-calibrated baseline)
- When you want simpler loss without KL estimation overhead

**Code location:** `kto_trainer.py:1169-1188`

---

## Evaluation Metrics

KTO tracks several metrics during training:

```python
{
    "rewards/chosen": 2.3,           # Average reward for desirable examples
    "rewards/rejected": -1.8,        # Average reward for undesirable examples
    "rewards/margins": 4.1,          # chosen - rejected margin
    "logps/chosen": -45.2,           # Log prob of chosen under policy
    "logps/rejected": -89.4,         # Log prob of rejected under policy
    "kl": 0.05,                      # Estimated KL divergence
    "loss": 0.42,                    # Total KTO loss
}
```

**Interpretation:**
- **Higher margins** = model better distinguishes desirable/undesirable
- **KL near 0** = policy close to reference (may need lower β to allow more deviation)
- **KL very high** = policy diverging too much (may need higher β or lower LR)

---

## Common Issues

### Issue 1: Unbalanced classes

**Problem:** 90% desirable, 10% undesirable (or vice versa)

**Solution:** Adjust weights according to KTO paper recommendations:
```python
config = KTOConfig(
    desirable_weight=1.0,
    undesirable_weight=9.0,  # Upweight minority class
)
```

### Issue 2: KL divergence is 0

**Problem:** KL stays at 0 throughout training

**Causes:**
1. Batch size too small (not enough diversity in KL dataset)
2. Model and reference are identical (early in training)
3. Data has low diversity

**Solutions:**
- Increase batch size
- Wait a few steps (KL will increase as model trains)
- Check data diversity

### Issue 3: High memory usage

**Problem:** Can't fit both policy and reference model

**Solutions:**
1. Use `precompute_ref_log_probs=True`
2. Use LoRA/PEFT (reduces trainable parameters)
3. Use gradient checkpointing
4. Reduce batch size

---

## Comparison: KTO vs DPO vs Reward Modeling

| Feature | KTO | DPO | Reward Model |
|---------|-----|-----|--------------|
| **Data format** | Binary (👍/👎) | Pairwise (A vs B) | Pairwise |
| **Reference model** | Required | Required | Not required |
| **Training stages** | 1 (direct) | 1 (direct) | 2 (RM → RL) |
| **KL estimation** | Required | No | No |
| **Memory** | Medium-High | Medium | Medium |
| **Data collection** | Easier | Harder | Harder |
| **Asymmetric loss** | Yes | No | N/A |

**When to use KTO:**
- You have binary feedback (thumbs up/down, good/bad)
- Data collection budget is limited (no need for pairwise comparisons)
- You want direct optimization (avoid reward model + RL)

**When NOT to use:**
- You only have pairwise data (use DPO instead - more direct)
- You need explicit reward scores (use reward modeling)
- You have very small batches (KL estimation unreliable)

---

## Memory Requirements

**7B model (assuming bf16):**
- **Full fine-tuning:** ~30-35 GB (policy + reference models)
- **With precompute_ref_log_probs:** ~18-20 GB (policy only)
- **LoRA + precompute + gradient checkpointing:** ~14-16 GB

**Comparison to DPO:**
KTO uses ~same memory as DPO (both need policy + reference), but KTO additionally computes KL dataset forward passes.

---

## Summary

**KTOTrainer = Binary Preferences → Aligned Model**

**Key formulas:**
```
L_chosen = 1 - σ(β * (log_ratio - KL))
L_rejected = 1 - σ(β * (KL - log_ratio))
```

**Advantages:**
- Easier data collection (binary labels)
- Theoretically grounded (prospect theory)
- Asymmetric loss (matches human psychology)

**Use cases:**
- Training on user feedback (thumbs up/down)
- When pairwise comparisons unavailable
- Direct preference optimization without reward models

**Output:**
An aligned model that better matches human preferences, using only binary feedback data.
