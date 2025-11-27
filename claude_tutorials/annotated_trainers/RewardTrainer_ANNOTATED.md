# RewardTrainer: Training Reward Models for RLHF

**Source:** `trl/trainer/reward_trainer.py` (606 lines)
**Purpose:** Train reward models from human preference data - essential first step for PPO/RLHF

---

## Overview

RewardTrainer trains **outcome reward models (ORM)** that predict human preferences. These models are used in PPO/RLOO/GRPO to provide learning signals.

### The RLHF Pipeline

```
1. SFT → Instruction-following model
2. RewardTrainer → Train reward model on preferences ← WE ARE HERE
3. PPO/GRPO → Optimize policy using reward model
```

### What is a Reward Model?

A reward model is a **sequence classification model** that outputs a scalar score for (prompt, response) pairs:

```python
reward = reward_model(prompt + response)  # Returns scalar
```

Higher scores = better responses according to human preferences.

---

## Core Concept: Bradley-Terry Model

### Mathematical Foundation

Given preference data where humans chose response A over B, the Bradley-Terry model assumes:

```
P(y_w > y_l | x) = σ(r(x, y_w) - r(x, y_l))

where:
  y_w = chosen (winning) response
  y_l = rejected (losing) response
  r(x, y) = reward model score
  σ = sigmoid function
```

### Loss Function

```
L = -log σ(r_w - r_l)
  = -log(1 / (1 + exp(-(r_w - r_l))))
  = log(1 + exp(r_l - r_w))
```

**Code:** `reward_trainer.py:400-450` (inherited from base Trainer)

**Intuition:** Maximize the probability that the reward model ranks chosen responses higher than rejected ones.

---

## Implementation Details

### Data Collator

**DataCollatorForPreference** (`reward_trainer.py:82-169`)

**What it does:**
Converts preference pairs into a batch suitable for reward model training.

**Input format:**
```python
examples = [
    {
        "chosen_input_ids": [1, 2, 3, 4],      # Prompt + chosen response
        "rejected_input_ids": [1, 2, 5, 6],    # Prompt + rejected response
        "margin": 0.5  # Optional: preference strength
    },
]
```

**Output format:**
```python
{
    "input_ids": [
        [1, 2, 3, 4],  # Chosen (first half of batch)
        [1, 2, 5, 6],  # Rejected (second half of batch)
    ],
    "attention_mask": [[1,1,1,1], [1,1,1,1]],
    "margin": [0.5]  # Optional
}
```

**Key insight:** Batch structure is `[chosen_1, chosen_2, ..., rejected_1, rejected_2, ...]`

**Code:**
```python
def torch_call(self, examples):
    chosen_input_ids = [torch.tensor(ex["chosen_input_ids"]) for ex in examples]
    rejected_input_ids = [torch.tensor(ex["rejected_input_ids"]) for ex in examples]

    # Concatenate: [chosen_1, chosen_2, rejected_1, rejected_2]
    input_ids = chosen_input_ids + rejected_input_ids

    # Pad to max length in batch
    output["input_ids"] = pad(input_ids, padding_value=pad_token_id)
    output["attention_mask"] = pad(attention_masks, padding_value=0)

    return output
```

### Model Architecture

**Base:** Any causal LM (e.g., Llama, Qwen) + **scalar head**

```python
# Loading automatically adds sequence classification head
model = AutoModelForSequenceClassification.from_pretrained(
    "Qwen/Qwen2-0.5B",
    num_labels=1,  # Single scalar output
)
```

**Architecture:**
```
Input tokens → Transformer → Hidden states → Score head → Scalar reward
                                              [Linear layer]
```

**Code location:** `reward_trainer.py:250-280`

### Compute Loss

The loss is computed in `compute_loss()` method:

**Steps:**
1. Forward pass on batch (chosen + rejected)
2. Extract scores
3. Split into chosen/rejected halves
4. Compute ranking loss

**Code:**
```python
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    # Forward pass
    rewards = model(**inputs, return_dict=True).logits  # [batch_size, 1]

    # Split batch into chosen and rejected halves
    batch_size = rewards.size(0) // 2
    chosen_rewards = rewards[:batch_size]
    rejected_rewards = rewards[batch_size:]

    # Bradley-Terry loss
    loss = -torch.nn.functional.logsigmoid(chosen_rewards - rejected_rewards).mean()

    return (loss, outputs) if return_outputs else loss
```

**Location:** Inherited from BaseTrainer, customized in `reward_trainer.py:400-450`

### Optional: Margin Loss

Some datasets include preference **strength** (how much better is A than B?):

```python
# Standard loss
loss = -log σ(r_w - r_l)

# With margin m
loss = -log σ(r_w - r_l - m)
```

**When to use:** If your preference data includes confidence scores.

---

## Training Configuration

### RewardConfig

**Key parameters:**

```python
@dataclass
class RewardConfig(TrainingArguments):
    max_length: int = 512           # Max sequence length
    gradient_checkpointing: bool = True  # Save memory
    remove_unused_columns: bool = False  # Keep all columns
```

**Typical hyperparameters:**

```python
config = RewardConfig(
    output_dir="./reward_model",
    learning_rate=1e-5,              # Lower than SFT
    per_device_train_batch_size=4,   # Pairs per device
    num_train_epochs=1,              # Usually 1 epoch sufficient
    gradient_checkpointing=True,
    bf16=True,
    max_length=512,                  # Truncate long sequences
)
```

### Dataset Requirements

**Must have columns:**
- `chosen`: Chosen response (can be string or messages)
- `rejected`: Rejected response (can be string or messages)
- `prompt`: (optional) If not included, inferred from chosen/rejected

**Supported formats:**

1. **Standard:**
```python
{
    "prompt": "What is 2+2?",
    "chosen": "2+2 equals 4.",
    "rejected": "I don't know."
}
```

2. **Conversational:**
```python
{
    "chosen": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."}
    ],
    "rejected": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "I don't know."}
    ]
}
```

---

## Production Example

```python
from datasets import load_dataset
from trl import RewardTrainer, RewardConfig
from peft import LoraConfig

# Load preference dataset
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

# Configure training
config = RewardConfig(
    output_dir="./reward_model",
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=1000,
    gradient_checkpointing=True,
    bf16=True,
    max_length=1024,
)

# Optional: Use LoRA for efficiency
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="SEQ_CLS",  # Sequence classification
)

# Initialize trainer
trainer = RewardTrainer(
    model="Qwen/Qwen2-0.5B",
    args=config,
    train_dataset=dataset,
    eval_dataset=dataset.select(range(1000)),  # Small eval set
    peft_config=peft_config,
)

# Train
trainer.train()

# Save
trainer.save_model("./reward_model_final")
```

---

## Evaluation Metrics

**Accuracy:** Fraction of pairs where chosen > rejected

```python
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # predictions shape: [batch_size * 2, 1]

    batch_size = predictions.shape[0] // 2
    chosen_rewards = predictions[:batch_size]
    rejected_rewards = predictions[batch_size:]

    accuracy = (chosen_rewards > rejected_rewards).mean()

    return {"accuracy": accuracy}

trainer = RewardTrainer(
    ...,
    compute_metrics=compute_metrics,
)
```

**Typical values:**
- Random: 50%
- Good reward model: 65-75%
- Excellent: 75-85%

---

## Common Issues

### Issue 1: Reward hacking in downstream RL

**Problem:** PPO exploits reward model weaknesses.

**Solutions:**
- Train on diverse data
- Use ensemble of reward models
- Add KL penalty in PPO
- Regular human evaluation

### Issue 2: Overfitting

**Symptoms:** Training accuracy >> eval accuracy

**Solutions:**
- More data
- Regularization (dropout, weight decay)
- LoRA (implicit regularization)
- Early stopping

### Issue 3: Reward collapse

**Symptoms:** All responses get similar scores

**Solutions:**
- Check data quality (real preferences?)
- Increase learning rate
- Reduce regularization
- Check model capacity

---

## Memory Requirements

**7B model:**
- Full fine-tuning: ~14 GB (model) + ~10 GB (activations) = **24 GB**
- LoRA: ~14 GB (model) + ~2 GB (activations) = **16 GB**
- LoRA + gradient checkpointing: **12-14 GB** (fits RTX 4090)

---

## Summary

**RewardTrainer = Preference → Scores**

**Key formula:**
```
loss = -log σ(reward_chosen - reward_rejected)
```

**Use cases:**
- Train reward models for PPO/GRPO
- Quality filtering (score generated outputs)
- Preference evaluation

**Outputs:**
A model that scores any (prompt, response) pair, enabling RL-based alignment.
