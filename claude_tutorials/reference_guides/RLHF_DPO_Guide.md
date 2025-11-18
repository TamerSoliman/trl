# RLHF & DPO: Complete Data Flow and Lifecycle Guide

**Purpose**: This guide traces the complete lifecycle of preference data through the DPO training pipeline, from raw dataset to trained model.

---

## Table of Contents

1. [Overview of the Pipeline](#overview-of-the-pipeline)
2. [Input Data Format](#input-data-format)
3. [Data Transformations](#data-transformations)
4. [Complete Training Lifecycle](#complete-training-lifecycle)
5. [Memory and Computation Flow](#memory-and-computation-flow)
6. [Comparison: DPO vs Traditional RLHF](#comparison-dpo-vs-traditional-rlhf)

---

## Overview of the Pipeline

### High-Level Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                    DPO TRAINING PIPELINE                      │
└───────────────────────────────────────────────────────────────┘

INPUT: Preference Dataset
├── Format: {prompt, chosen, rejected}
├── Source: Human annotations, AI feedback, or synthetic
└── Size: Typically 10K-1M pairs

                        ↓

PREPROCESSING
├── Tokenization: Convert text to token IDs
├── Chat template application (if conversational)
└── Truncation and padding

                        ↓

BATCHING & COLLATION
├── Group examples into batches
├── Pad to same length
└── Create attention masks

                        ↓

FORWARD PASSES
├── Policy model: Get log probabilities
└── Reference model: Get reference log probabilities

                        ↓

LOSS COMPUTATION
├── Compute log probability ratios
├── Apply DPO loss function
└── Calculate metrics

                        ↓

OPTIMIZATION
├── Backpropagation through policy model
├── Gradient accumulation
└── Parameter update

                        ↓

OUTPUT: Aligned Policy Model
├── Prefers chosen over rejected responses
├── Stays close to reference model (KL constraint)
└── Ready for deployment or further fine-tuning
```

---

## Input Data Format

### Standard Format

DPO expects preference data in one of two formats:

#### Format 1: Explicit Prompts (Recommended)

```python
{
    "prompt": str,      # The instruction/question
    "chosen": str,      # Preferred response
    "rejected": str     # Dispreferred response
}
```

**Example**:
```python
{
    "prompt": "Explain photosynthesis in simple terms.",
    "chosen": "Photosynthesis is the process where plants convert sunlight into energy...",
    "rejected": "I don't know about photosynthesis."
}
```

#### Format 2: Conversational (Messages)

```python
{
    "chosen": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ],
    "rejected": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
}
```

The trainer will automatically extract the prompt (everything except the last assistant message).

---

### Popular Preference Datasets

| Dataset | Size | Domain | Format |
|---------|------|--------|--------|
| `trl-lib/ultrafeedback_binarized` | ~64K | General | Explicit prompt |
| `Anthropic/hh-rlhf` | ~161K | Helpful & Harmless | Conversational |
| `argilla/ultrafeedback-binarized-preferences` | ~64K | General | Explicit prompt |
| `Intel/orca_dpo_pairs` | ~13K | Reasoning | Conversational |

---

## Data Transformations

### Transformation 1: Prompt Extraction

**Code**: `maybe_extract_prompt()` in `data_utils.py`

**Input**:
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

**Output**:
```python
{
    "prompt": [{"role": "user", "content": "What is 2+2?"}],
    "chosen": [{"role": "assistant", "content": "2+2 equals 4."}],
    "rejected": [{"role": "assistant", "content": "I don't know."}]
}
```

**Why**: Separates the shared context (prompt) from the differing responses.

---

### Transformation 2: Chat Template Application

**Code**: `maybe_apply_chat_template()` in `data_utils.py`

**Input** (from Transformation 1):
```python
{
    "prompt": [{"role": "user", "content": "What is 2+2?"}],
    "chosen": [{"role": "assistant", "content": "2+2 equals 4."}],
    "rejected": [{"role": "assistant", "content": "I don't know."}]
}
```

**Output** (depends on model's chat template, e.g., ChatML):
```python
{
    "prompt": "<|im_start|>user\nWhat is 2+2?<|im_end|>\n",
    "chosen": "<|im_start|>assistant\n2+2 equals 4.<|im_end|>\n",
    "rejected": "<|im_start|>assistant\nI don't know.<|im_end|>\n"
}
```

**Why**: Different models expect different conversation formats (ChatML, Llama, Mistral, etc.).

---

### Transformation 3: Tokenization

**Code**: `DPOTrainer.tokenize_row()` at line 685

**Input** (from Transformation 2):
```python
{
    "prompt": "<|im_start|>user\nWhat is 2+2?<|im_end|>\n",
    "chosen": "<|im_start|>assistant\n2+2 equals 4.<|im_end|>\n",
    "rejected": "<|im_start|>assistant\nI don't know.<|im_end|>\n"
}
```

**Processing**:
```python
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt_ids = tokenizer("What is 2+2?", add_special_tokens=False)["input_ids"]
# → [1841, 374, 220, 17, 10, 17, 30]

chosen_ids = tokenizer("2+2 equals 4.", add_special_tokens=False)["input_ids"]
# → [17, 10, 17, 17239, 220, 19, 13]
chosen_ids = chosen_ids + [tokenizer.eos_token_id]  # Add EOS
# → [17, 10, 17, 17239, 220, 19, 13, 50256]

rejected_ids = tokenizer("I don't know.", add_special_tokens=False)["input_ids"]
# → [40, 1541, 956, 1440, 13]
rejected_ids = rejected_ids + [tokenizer.eos_token_id]
# → [40, 1541, 956, 1440, 13, 50256]
```

**Output**:
```python
{
    "prompt_input_ids": [1841, 374, 220, 17, 10, 17, 30],
    "chosen_input_ids": [17, 10, 17, 17239, 220, 19, 13, 50256],
    "rejected_input_ids": [40, 1541, 956, 1440, 13, 50256]
}
```

**Truncation** (if needed):
- Prompt: Truncate from LEFT (keep most recent context)
- Completions: Truncate from RIGHT (keep beginning)

---

### Transformation 4: Batching and Padding

**Code**: `DataCollatorForPreference` at line 108

**Input**: List of tokenized examples
```python
[
    {
        "prompt_input_ids": [1, 2, 3],
        "chosen_input_ids": [4, 5],
        "rejected_input_ids": [6]
    },
    {
        "prompt_input_ids": [7, 8],
        "chosen_input_ids": [9, 10],
        "rejected_input_ids": [11, 12, 13]
    }
]
```

**Output**: Padded batch
```python
{
    "prompt_input_ids": tensor([
        [0, 1, 2, 3],  # Left-padded
        [0, 0, 7, 8]
    ]),
    "prompt_attention_mask": tensor([
        [0, 1, 1, 1],
        [0, 0, 1, 1]
    ]),
    "chosen_input_ids": tensor([
        [4, 5, 0],     # Right-padded
        [9, 10, 0]
    ]),
    "chosen_attention_mask": tensor([
        [1, 1, 0],
        [1, 1, 0]
    ]),
    "rejected_input_ids": tensor([
        [6, 0, 0],
        [11, 12, 13]
    ]),
    "rejected_attention_mask": tensor([
        [1, 0, 0],
        [1, 1, 1]
    ])
}
```

**Padding Strategy**:
- **Prompts**: Left-padded (for causal LM efficiency)
- **Completions**: Right-padded (standard)

---

### Transformation 5: Concatenation

**Code**: `DPOTrainer.concatenated_inputs()` at line 946

**Purpose**: Combine chosen and rejected into single batch for efficient forward pass

**Input** (from Transformation 4, batch_size=2):
```python
{
    "prompt_input_ids": [[P1], [P2]],
    "chosen_input_ids": [[C1], [C2]],
    "rejected_input_ids": [[R1], [R2]]
}
```

**Output**:
```python
{
    "prompt_input_ids": [[P1], [P2], [P1], [P2]],  # Doubled
    "completion_input_ids": [[C1], [C2], [R1], [R2]]  # Chosen first, rejected second
}
```

**Batch Structure**:
```
Index 0: Prompt 1 + Chosen 1
Index 1: Prompt 2 + Chosen 2
Index 2: Prompt 1 + Rejected 1  ← Same prompts repeated
Index 3: Prompt 2 + Rejected 2
```

**Why**: Process both chosen and rejected in a single forward pass (2x faster).

---

## Complete Training Lifecycle

### Phase 1: Initialization

```python
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load pretrained model (usually SFT model)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

# 2. Create reference model (frozen copy)
ref_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
# This is π_ref, frozen during training

# 3. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

# 4. Load preference dataset
from datasets import load_dataset
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
# Format: {prompt, chosen, rejected}

# 5. Configure trainer
config = DPOConfig(
    output_dir="./dpo_output",
    beta=0.1,  # Temperature for DPO loss
    learning_rate=5e-7,
    per_device_train_batch_size=2,
    max_length=512
)

# 6. Initialize trainer
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=config,
    train_dataset=dataset,
    processing_class=tokenizer
)
```

**What happens in `__init__`**:
1. Model and reference model prepared for distributed training
2. Dataset is preprocessed (tokenization happens here)
3. Data collator is created
4. Optimizer and scheduler are set up

---

### Phase 2: Dataset Preprocessing (During `__init__`)

**Step-by-step**:

1. **Extract prompts** (if conversational):
   ```python
   dataset = dataset.map(maybe_extract_prompt)
   ```

2. **Apply chat template**:
   ```python
   dataset = dataset.map(
       maybe_apply_chat_template,
       fn_kwargs={"tokenizer": tokenizer}
   )
   ```

3. **Tokenize**:
   ```python
   dataset = dataset.map(
       DPOTrainer.tokenize_row,
       fn_kwargs={"processing_class": tokenizer, "max_prompt_length": 512}
   )
   ```

After preprocessing, each example looks like:
```python
{
    "prompt_input_ids": [token_ids...],
    "chosen_input_ids": [token_ids...],
    "rejected_input_ids": [token_ids...]
}
```

---

### Phase 3: Training Loop

```python
trainer.train()
```

**What happens in each training step**:

#### Step 3.1: Get Batch

```python
for batch in dataloader:
    # batch = {
    #     "prompt_input_ids": tensor([...]),
    #     "chosen_input_ids": tensor([...]),
    #     "rejected_input_ids": tensor([...]),
    #     "prompt_attention_mask": tensor([...]),
    #     ...
    # }
```

#### Step 3.2: Concatenate Inputs

```python
concatenated_batch = self.concatenated_inputs(batch)
# Doubles batch size: [chosen_1, chosen_2, ..., rejected_1, rejected_2, ...]
```

#### Step 3.3: Policy Model Forward Pass

```python
# Concatenate prompt + completion for each example
input_ids = torch.cat([prompt_ids, completion_ids], dim=1)

# Forward pass
outputs = policy_model(input_ids, attention_mask=attention_mask)
logits = outputs.logits[:, :-1, :]  # Shape: [2*batch_size, seq_len-1, vocab_size]

# Compute log probabilities
labels = input_ids[:, 1:]
log_probs = selective_log_softmax(logits, labels)

# Sum over completion tokens only (not prompt)
all_logps_sum = (log_probs * loss_mask).sum(dim=-1)  # Shape: [2*batch_size]

# Split back into chosen and rejected
chosen_logps = all_logps_sum[:batch_size]
rejected_logps = all_logps_sum[batch_size:]
```

#### Step 3.4: Reference Model Forward Pass

```python
with torch.no_grad():  # No gradients for reference model
    if ref_model is not None:
        ref_outputs = ref_model(input_ids, attention_mask=attention_mask)
    else:
        # Use policy model with adapters disabled (for PEFT)
        with model.disable_adapter():
            ref_outputs = policy_model(input_ids, attention_mask=attention_mask)

    # Same process as policy model
    ref_logps_sum = (ref_log_probs * loss_mask).sum(dim=-1)
    ref_chosen_logps = ref_logps_sum[:batch_size]
    ref_rejected_logps = ref_logps_sum[batch_size:]
```

#### Step 3.5: Compute DPO Loss

```python
# Compute log ratios
chosen_logratios = chosen_logps - ref_chosen_logps
rejected_logratios = rejected_logps - ref_rejected_logps

# Compute logits
logratios = chosen_logps - rejected_logps
ref_logratios = ref_chosen_logps - ref_rejected_logps
logits = logratios - ref_logratios

# Compute loss (sigmoid variant)
losses = -F.logsigmoid(beta * logits)
loss = losses.mean()  # Average over batch
```

#### Step 3.6: Backpropagation

```python
loss.backward()  # Compute gradients
optimizer.step()  # Update policy model parameters
optimizer.zero_grad()  # Reset gradients

# Reference model is NOT updated (frozen)
```

#### Step 3.7: Logging

```python
# Compute metrics
chosen_rewards = beta * chosen_logratios.detach()
rejected_rewards = beta * rejected_logratios.detach()
reward_margin = (chosen_rewards - rejected_rewards).mean()
accuracy = (chosen_rewards > rejected_rewards).float().mean()

# Log to wandb/tensorboard
log({
    "loss": loss.item(),
    "reward_margin": reward_margin.item(),
    "accuracy": accuracy.item(),
    "learning_rate": scheduler.get_last_lr()[0]
})
```

---

## Memory and Computation Flow

### Memory Requirements

For a single training step with:
- Model: 7B parameters (14GB in fp16)
- Batch size: 4
- Sequence length: 512
- Gradient accumulation: 4

**Memory breakdown**:

| Component | Memory |
|-----------|--------|
| Policy model (fp16) | ~14 GB |
| Reference model (fp16) | ~14 GB |
| Optimizer states (Adam) | ~28 GB |
| Gradients | ~14 GB |
| Activations (batch=4, seq=512) | ~8 GB |
| **Total** | **~78 GB** |

**Optimization strategies**:
1. **Gradient checkpointing**: Reduces activations to ~2GB (-6GB)
2. **Precompute reference log probs**: Removes need for ref model in memory (-14GB)
3. **PEFT (LoRA)**: Only store adapter weights (~100MB)
4. **Quantization**: Load model in 4-bit/8-bit (~4-7GB)

With all optimizations: **~20GB** (fits on single A6000/4090)

---

### Computational Flow

**FLOPs per training step**:

```
Forward pass (policy):     2 * N * L * B  (N=params, L=seq_len, B=batch_size)
Forward pass (reference):  2 * N * L * B
Backward pass:             6 * N * L * B  (2x forward for gradients + 4x for optimizer)
──────────────────────────────────────────
Total:                     12 * N * L * B
```

For 7B model, 512 tokens, batch=4:
```
12 * 7e9 * 512 * 4 ≈ 172 trillion FLOPs per step
```

On an A100 (312 TFLOPS fp16):
```
172e12 / 312e12 ≈ 0.55 seconds per step (theoretical)
```

In practice: ~2-3 seconds/step due to memory bandwidth and overhead.

---

## Comparison: DPO vs Traditional RLHF

### Traditional RLHF Pipeline

```
┌────────────┐      ┌──────────────┐      ┌──────────────┐
│    SFT     │  →   │ Reward Model │  →   │  PPO/REINFORCE│
│            │      │   Training   │      │   Training    │
└────────────┘      └──────────────┘      └──────────────┘
  Instruct           Bradley-Terry         Online RL
   tuning            preference model      optimization

Required:            Required:              Required:
- Base model        - Preference data      - SFT model
- Instruct data     - SFT model            - Reward model
                                           - Value model
                                           - Reference model

Output:              Output:                Output:
- SFT model         - Reward model         - Aligned model
```

**Total**: 3 stages, 4 models needed

---

### DPO Pipeline

```
┌────────────┐      ┌──────────────┐
│    SFT     │  →   │     DPO      │
│            │      │   Training   │
└────────────┘      └──────────────┘
  Instruct           Direct preference
   tuning            optimization

Required:            Required:
- Base model        - Preference data
- Instruct data     - SFT model (becomes ref)

Output:              Output:
- SFT model         - Aligned model
```

**Total**: 2 stages, 2 models needed

---

### Side-by-Side Comparison

| Aspect | Traditional RLHF | DPO |
|--------|-----------------|-----|
| **Stages** | 3 (SFT → RM → RL) | 2 (SFT → DPO) |
| **Models needed** | 4 (policy, ref, value, reward) | 2 (policy, ref) |
| **Reward model** | Explicit (trained separately) | Implicit (in loss function) |
| **Training stability** | Can be unstable (RL) | More stable (supervised) |
| **Hyperparameter sensitivity** | High (many RL params) | Low (mainly β) |
| **Computational cost** | High (4 models, online generation) | Medium (2 models, offline) |
| **Memory requirement** | ~80-100GB | ~40-60GB |
| **Sample efficiency** | Lower (online exploration) | Higher (offline data reuse) |
| **Data efficiency** | Needs more data | Works with less data |
| **Best for** | Complex reasoning, safety | General alignment, faster iteration |

---

### When to Use Which?

**Use DPO when**:
- You have good preference data
- Want faster iteration
- Have limited compute
- Need stable training
- Doing general instruction following

**Use RLHF when**:
- Need online exploration
- Have complex reward function
- Doing safety-critical applications
- Can afford computational cost
- Need fine-grained reward shaping

**Hybrid approaches**:
- Start with DPO for quick alignment
- Fine-tune with RLHF for specific behaviors
- Use Online DPO for best of both worlds

---

## Summary

### Key Data Transformations

```
Raw text (prompt, chosen, rejected)
  ↓ maybe_extract_prompt()
Separated components
  ↓ maybe_apply_chat_template()
Formatted strings
  ↓ tokenize_row()
Token IDs
  ↓ DataCollator
Padded batches
  ↓ concatenated_inputs()
Concatenated batches
  ↓ concatenated_forward()
Log probabilities
  ↓ dpo_loss()
Loss value
  ↓ backward()
Gradients
  ↓ optimizer.step()
Updated policy model
```

### Critical Insights

1. **Efficiency**: Concatenation trick processes chosen+rejected together
2. **Stability**: No need for online RL, just supervised learning
3. **Simplicity**: Only 2 models needed vs 4 in traditional RLHF
4. **Flexibility**: 15+ loss variants for different scenarios
5. **Scalability**: Works with standard distributed training tools

### Resource Requirements

**Minimum**:
- GPU: 1x A6000 (48GB) with gradient checkpointing + PEFT
- Data: ~10K preference pairs
- Time: ~6-12 hours for 7B model, 1 epoch

**Recommended**:
- GPU: 4x A100 (80GB) for efficient distributed training
- Data: ~50K-100K preference pairs for robust alignment
- Time: ~2-4 hours for 7B model, 3 epochs

---

## Next Steps

For deeper understanding:
- See `DPO_Loss_Reference.md` for mathematical derivation
- See `annotated_trainers/DPOTrainer_ANNOTATED.md` for line-by-line code
- See example scripts in `annotated_examples/` for complete training setups
