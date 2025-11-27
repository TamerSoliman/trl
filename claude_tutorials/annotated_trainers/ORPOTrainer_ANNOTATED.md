# ORPOTrainer: Odds Ratio Preference Optimization

**Source:** `trl/trainer/orpo_trainer.py` (900+ lines)
**Purpose:** Align models using preference data WITHOUT a reference model

---

## Overview

ORPOTrainer implements **Odds Ratio Preference Optimization**, a monolithic preference optimization method that eliminates the need for a reference model. This is ORPO's key innovation - it combines SFT and preference learning in a single stage.

### The ORPO Advantage

**Traditional pipeline (DPO/KTO):**
```
1. SFT → Base model
2. DPO/KTO → Aligned model (requires reference model)
```

**ORPO pipeline:**
```
1. ORPO → Aligned model (no reference model needed!)
```

**Benefits:**
- **50% memory savings** (no reference model)
- **Single-stage training** (SFT + alignment together)
- **Faster** (one forward pass instead of two)

---

## Core Concept: Odds Ratio Loss

### Mathematical Foundation

ORPO uses the **odds ratio** between chosen and rejected outputs instead of comparing to a reference model.

**Odds (not odds ratio yet):**
```
Odds of chosen:   P(y_w|x) / (1 - P(y_w|x))
Odds of rejected: P(y_l|x) / (1 - P(y_l|x))
```

**Odds ratio:**
```
OR(y_w, y_l | x) = [P(y_w|x) / (1-P(y_w|x))] / [P(y_l|x) / (1-P(y_l|x))]
```

**In log space:**
```
log OR = [log P(y_w|x) - log(1-P(y_w|x))] - [log P(y_l|x) - log(1-P(y_l|x))]
```

**Code location:** `orpo_trainer.py:655-685`

### ORPO Loss Function

**Full ORPO loss:**
```
L_ORPO = L_SFT - λ * E[log sigmoid(log OR)]
       = L_NLL(y_w) - λ * E[log σ(log OR)]

where:
  L_NLL = negative log-likelihood (standard SFT loss)
  λ = beta parameter (controls preference vs. SFT tradeoff)
  σ = sigmoid function
```

**Code:**
```python
def odds_ratio_loss(policy_chosen_logps, policy_rejected_logps):
    # Compute log odds ratio
    log_odds = (
        (policy_chosen_logps - policy_rejected_logps) -
        (torch.log1p(-torch.exp(policy_chosen_logps)) -
         torch.log1p(-torch.exp(policy_rejected_logps)))
    )

    # Preference loss (we want to MAXIMIZE log_odds, so MINIMIZE negative)
    ratio = F.logsigmoid(log_odds)
    preference_loss = self.beta * ratio  # Will be subtracted from NLL

    return preference_loss

# Full loss
loss = nll_loss - preference_loss.mean()
```

**Code location:** `orpo_trainer.py:655-685`, `orpo_trainer.py:835`

**Intuition:**
- **NLL term:** Teaches model to generate chosen outputs (like SFT)
- **Odds ratio term:** Amplifies difference between chosen and rejected
- **Combined:** Model learns both to generate well AND prefer better outputs

---

## Why No Reference Model?

### The DPO Comparison

**DPO loss:**
```
L_DPO = -log σ(β * [log π/π_ref(y_w) - log π/π_ref(y_l)])
      = -log σ(β * [(log π(y_w) - log π_ref(y_w)) - (log π(y_l) - log π_ref(y_l))])
```

**Key:** DPO compares current policy to a **fixed reference** π_ref.

### ORPO's Alternative

**ORPO loss:**
```
L_ORPO = L_NLL(y_w) - λ * log σ(log_odds(y_w, y_l))
```

**Key:** ORPO compares chosen to rejected **directly**, using the **same model**.

**The trade:**
- ✅ No reference model needed (saves memory)
- ✅ Single-stage training (faster)
- ⚠️ Must include NLL term to prevent collapse (otherwise model could make both probabilities tiny)

---

## Two-Component Loss

### Component 1: SFT Loss (NLL)

```python
def cross_entropy_loss(logits, labels):
    logits = logits[..., :-1, :].contiguous()  # Shift
    labels = labels[..., 1:].contiguous()

    loss_fct = nn.CrossEntropyLoss()
    return loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))

# Compute only on chosen outputs
nll_loss = cross_entropy_loss(chosen_logits, chosen_labels)
```

**Code location:** `orpo_trainer.py:765-786`

**Purpose:** Standard supervised learning - teaches model to generate the chosen responses.

**Why only chosen?** We want the model to learn to generate preferred outputs, not rejected ones!

### Component 2: Odds Ratio Loss

```python
# Get log probabilities for both chosen and rejected
chosen_logps = get_batch_logps(chosen_logits, chosen_labels, average_log_prob=True)
rejected_logps = get_batch_logps(rejected_logits, rejected_labels, average_log_prob=True)

# Compute odds ratio
log_odds = (
    (chosen_logps - rejected_logps) -
    (torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps)))
)

preference_loss = self.beta * F.logsigmoid(log_odds)
```

**Code location:** `orpo_trainer.py:675-680`

**Purpose:** Amplify the gap between chosen and rejected probabilities.

### Combined Loss

```python
loss = nll_loss - preference_loss.mean()
```

**The minus sign:** We want to MAXIMIZE preference_loss (more positive = better odds ratio), so we MINIMIZE its negative.

**Code location:** `orpo_trainer.py:835`

---

## Implementation Details

### Concatenated Forward Pass

Like DPO, ORPO uses a single forward pass for efficiency:

```python
def concatenated_forward(model, batch):
    # Concatenate chosen and rejected
    concatenated_batch = {
        "input_ids": torch.cat([chosen_input_ids, rejected_input_ids]),
        "attention_mask": torch.cat([chosen_attention_mask, rejected_attention_mask]),
    }

    # Single forward pass
    outputs = model(**concatenated_batch)

    # Split outputs
    len_chosen = chosen_input_ids.shape[0]
    chosen_logits = outputs.logits[:len_chosen]
    rejected_logits = outputs.logits[len_chosen:]

    return chosen_logits, rejected_logits
```

**Code location:** `orpo_trainer.py:730-809`

**Benefit:** FSDP/DDP friendly - single communication round instead of two.

### Log Probability Computation

```python
def get_batch_logps(logits, labels, average_log_prob=False):
    # Shift for causal LM
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]

    # Mask padding
    loss_mask = (labels != label_pad_token_id)
    labels = torch.where(labels == label_pad_token_id, 0, labels)

    # Compute per-token log probs
    per_token_logps = selective_log_softmax(logits, labels)

    # Average or sum
    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)
```

**Code location:** `orpo_trainer.py:687-728`

**Note:** ORPO uses **average** log prob (per-token), not sum like DPO. This normalizes for sequence length.

---

## Training Configuration

### ORPOConfig

```python
from trl import ORPOConfig

config = ORPOConfig(
    # Core ORPO parameter
    beta=0.1,                      # λ in paper (odds ratio weight)

    # Sequence lengths
    max_length=1024,               # Max total length
    max_prompt_length=512,         # Max prompt length

    # Training
    learning_rate=1e-6,            # Lower than SFT
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,            # Can train longer than DPO

    # Memory
    gradient_checkpointing=True,
    bf16=True,

    # Evaluation
    eval_strategy="steps",
    eval_steps=500,
    generate_during_eval=False,

    output_dir="./orpo_model",
)
```

**Code location:** `trl/trainer/orpo_config.py`

### Key Hyperparameters

**Beta (λ):**
- **Range:** 0.05 - 0.5
- **Default:** 0.1
- **Higher β:** More weight on preference learning
- **Lower β:** More weight on SFT (generation quality)

**Learning Rate:**
- **Default:** 1e-6 (same as DPO/KTO)
- **Lower than SFT** due to combined objective

**Number of Epochs:**
- ORPO can train for **longer** than DPO (3-5 epochs)
- Since it's also doing SFT, needs more iterations

---

## Dataset Format

Same as DPO - pairwise preference data:

```python
{
    "prompt": "What is the capital of France?",
    "chosen": "The capital of France is Paris.",
    "rejected": "I don't know."
}
```

Or conversational format:

```python
{
    "chosen": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"}
    ],
    "rejected": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "5"}
    ]
}
```

---

## Production Example

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import ORPOTrainer, ORPOConfig

# Load model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")

# Load dataset
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

# Configure training
config = ORPOConfig(
    output_dir="./orpo_output",
    beta=0.1,
    max_length=1024,
    max_prompt_length=512,
    learning_rate=8e-7,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    gradient_checkpointing=True,
    bf16=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=500,
)

# Optional: LoRA
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

# Initialize trainer
trainer = ORPOTrainer(
    model=model,
    args=config,
    train_dataset=dataset,
    eval_dataset=dataset.select(range(1000)),
    processing_class=tokenizer,
    peft_config=peft_config,
)

# Train
trainer.train()

# Save
trainer.save_model("./orpo_final")
```

---

## Evaluation Metrics

```python
{
    "nll_loss": 2.1,                  # SFT component
    "log_odds_ratio": 1.5,            # Preference component
    "log_odds_chosen": 0.8,           # Log sigmoid of odds
    "rewards/chosen": -45.2,          # Average log prob
    "rewards/rejected": -89.4,
    "rewards/margins": 44.2,          # Chosen - rejected
    "rewards/accuracies": 0.73,       # Fraction where chosen > rejected
}
```

**Healthy training:**
- `nll_loss` decreasing (model learning to generate)
- `log_odds_ratio` increasing (preference gap widening)
- `rewards/margins` increasing
- `rewards/accuracies` > 0.6

---

## Comparison: ORPO vs DPO vs KTO

| Feature | ORPO | DPO | KTO |
|---------|------|-----|-----|
| **Reference model** | ❌ No | ✅ Yes | ✅ Yes |
| **Memory** | Low (1 model) | High (2 models) | High (2 models) |
| **Training stages** | 1 (monolithic) | 2 (SFT + DPO) | 2 (SFT + KTO) |
| **Data format** | Pairwise | Pairwise | Binary |
| **Loss components** | NLL + Odds Ratio | Relative Preference | KL-calibrated Preference |
| **Training epochs** | 3-5 | 1-2 | 1 |
| **Speed** | Fast | Medium | Medium |

**When to use ORPO:**
- Memory-constrained environments (single GPU)
- Want single-stage training
- Have high-quality pairwise data
- Starting from base model (not SFT'd model)

**When NOT to use:**
- Already have SFT'd model (use DPO instead)
- Need explicit reference model for evaluation
- Want maximum alignment quality (DPO/PPO often better)

---

## Memory Requirements

**7B model (bf16):**
- **ORPO:** ~14-16 GB (single model)
- **DPO:** ~28-30 GB (policy + reference)
- **Savings:** ~50%

**With LoRA + gradient checkpointing:**
- **ORPO:** ~12-14 GB (fits RTX 4090!)

---

## Common Issues

### Issue 1: Model collapse

**Symptoms:** Both chosen and rejected probabilities drop to near-zero

**Cause:** Beta too high, overwhelming the NLL term

**Solution:**
- Reduce beta (try 0.05)
- Ensure NLL loss is not too small
- Check that chosen responses are reasonable quality

### Issue 2: No preference learning

**Symptoms:** Margins not increasing, accuracies stuck at ~50%

**Cause:** Beta too low, NLL term dominating

**Solution:**
- Increase beta (try 0.2-0.3)
- Check data quality (are preferences clear?)
- Train longer (ORPO needs more epochs)

### Issue 3: High NLL but good odds ratio

**Symptoms:** Model has good preferences but poor generation quality

**Cause:** Beta too high, model not learning to generate well

**Solution:**
- Lower beta to balance SFT and preference
- Train longer to allow NLL to decrease
- Check if chosen responses are high quality

---

## Advanced: Odds Ratio Interpretation

### What is an "Odds Ratio"?

**Odds** (single outcome):
```
Odds = P(event) / P(not event) = P / (1-P)
```

**Example:** If P(chosen) = 0.8, then Odds = 0.8/0.2 = 4

**Odds Ratio** (comparison):
```
OR = Odds_A / Odds_B
```

**Example:** If Odds_chosen = 4 and Odds_rejected = 0.25, then OR = 4/0.25 = 16

**Interpretation:** Chosen output is 16x more "likely" (in odds terms) than rejected.

### Why Odds Ratio, Not Probability Ratio?

**Probability ratio:**
```
P(y_w) / P(y_l)
```

**Problem:** Asymmetric! If y_w is 2x more likely, ratio = 2. But if y_l is 2x more likely, ratio = 0.5 (not -2).

**Odds ratio:**
```
OR(y_w, y_l) = 1/OR(y_l, y_w)
```

**Symmetric in log space:** log OR(y_w, y_l) = -log OR(y_l, y_w)

This makes optimization more stable.

---

## Summary

**ORPOTrainer = SFT + Preference Learning, No Reference Needed**

**Key formulas:**
```
L_ORPO = L_NLL(chosen) - λ * E[log sigmoid(log_odds_ratio)]

log_odds_ratio = [log P(y_w) - log(1-P(y_w))] - [log P(y_l) - log(1-P(y_l))]
```

**Main advantage:**
- **Memory efficient** (no reference model)
- **Single-stage** training

**Trade-offs:**
- Combines two objectives (may need careful tuning)
- Usually needs more epochs than DPO

**Use cases:**
- Training from scratch with preference data
- Memory-constrained environments
- When you want monolithic training

**Output:**
An aligned model that has learned both to generate well and to prefer better outputs, using only a single model!
