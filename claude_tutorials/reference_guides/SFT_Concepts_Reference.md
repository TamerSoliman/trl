# SFT Concepts Reference: From Theory to Code

**Purpose:** Mathematical foundations and conceptual explanations for Supervised Fine-Tuning

---

## Table of Contents

1. [What is SFT?](#what-is-sft)
2. [The Loss Function](#the-loss-function)
3. [Packing Algorithms](#packing-algorithms)
4. [Padding-Free Training](#padding-free-training)
5. [Completion-Only Loss](#completion-only-loss)
6. [Chat Templates](#chat-templates)
7. [Performance Optimizations](#performance-optimizations)

---

## What is SFT?

### The Big Picture

**Supervised Fine-Tuning (SFT)** is the process of teaching a pretrained language model to follow instructions.

```
Pretrained Model (e.g., Llama-3-8B-Base)
    ↓ [trained on raw internet text]
Can complete: "The capital of France is" → "Paris"
Cannot do: "What is the capital of France?" → "The capital of France is Paris."

    ↓ [SFT on instruction data]

Instruction-Tuned Model (e.g., Llama-3-8B-Instruct)
Can do: "What is the capital of France?" → "The capital of France is Paris."
```

### Why SFT is the foundation

**All alignment techniques require SFT first:**

1. **SFT**: Pretrained → Instruction-following
2. **RLHF/DPO**: Instruction-following → Aligned with preferences
3. **PPO/GRPO**: Further refinement with reinforcement learning

**You cannot skip SFT.** Even DPO requires an SFT model as initialization (and often as the reference model).

### Training data format

SFT uses **instruction-response pairs**:

```json
{
  "prompt": "What is machine learning?",
  "completion": "Machine learning is a subset of AI that enables systems to learn from data without being explicitly programmed."
}
```

Or in conversational format:

```json
{
  "messages": [
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is a subset of AI..."}
  ]
}
```

---

## The Loss Function

### Standard Language Modeling

SFT uses **next-token prediction** (also called **causal language modeling**):

**Objective:** Maximize the probability of the correct next token.

**Mathematical formulation:**

Given a sequence of tokens `x = [x₁, x₂, ..., xₙ]`, the model learns to predict each token given all previous tokens:

```
p(x) = p(x₁) · p(x₂|x₁) · p(x₃|x₁,x₂) · ... · p(xₙ|x₁,...,xₙ₋₁)
```

**Loss function (Negative Log-Likelihood):**

```
L = -log p(x) = -Σᵢ log p(xᵢ | x₁, ..., xᵢ₋₁)
```

**In practice (PyTorch):**

```python
# Model forward pass
logits = model(input_ids)  # Shape: [batch, seq_len, vocab_size]

# Shift for next-token prediction
shift_logits = logits[:, :-1, :]  # Predict tokens 1 to n
shift_labels = labels[:, 1:]       # Target tokens are 1 to n

# Compute cross-entropy loss
loss = F.cross_entropy(
    shift_logits.reshape(-1, vocab_size),
    shift_labels.reshape(-1),
    ignore_index=-100  # Ignore masked tokens
)
```

**Why shift?**

```
Input:    [The, cat, sat, on, mat]
Logits:   [L0,  L1,  L2,  L3, L4, L5]
           ↓    ↓    ↓    ↓   ↓   ↓
Predicts: [?,  The, cat, sat, on, mat]

We want L1 to predict "cat", L2 to predict "sat", etc.
So we compare logits[:-1] with labels[1:]
```

### Code mapping: Where loss is computed

**In SFTTrainer:**

1. `compute_loss()` calls parent class (Transformers Trainer)
2. Transformers Trainer calls `model.forward()` which internally computes loss
3. If `compute_loss_func` is provided, it overrides the default

**Location in model (for CausalLM):**

```python
# transformers/models/llama/modeling_llama.py (example)

class LlamaForCausalLM:
    def forward(self, input_ids, labels=None, ...):
        # Get logits
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift logits and labels
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Compute loss
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )

        return CausalLMOutputWithPast(loss=loss, logits=logits, ...)
```

### DFT Loss (Alternative)

**Dynamic Fine-Tuning** modifies the loss to upweight uncertain predictions:

**Standard NLL:**
```
L = -Σᵢ log p(yᵢ | x)
```

**DFT Loss:**
```
L = -Σᵢ p(yᵢ | x) · log p(yᵢ | x)
```

**Code:** `sft_trainer.py:465-479`

```python
def dft_loss(outputs, labels, num_items_in_batch=None):
    # Get log probabilities
    logprobs = selective_log_softmax(outputs.logits, shift_labels)

    # Weight by probability (detached to prevent gradient flow)
    per_token_loss = -logprobs.exp().detach() * logprobs

    # Average over non-masked tokens
    loss = (per_token_loss * loss_mask).sum() / num_items_in_batch
    return loss
```

**Why detach?**

We want to weight by probability, but not backpropagate through the weighting term. Otherwise, the model would learn to minimize loss by reducing p(y), which is the opposite of what we want!

**When to use DFT:**
- When you have noisy or ambiguous labels
- When you want to focus learning on hard examples
- Generally provides slight quality improvements at the cost of slower convergence

---

## Packing Algorithms

### The padding problem

**Without packing:**

```
Sequence 1: [1, 2, 3, 0, 0, 0, 0, 0]     # 3 tokens, 5 padding
Sequence 2: [4, 5, 6, 7, 0, 0, 0, 0]     # 4 tokens, 4 padding
Sequence 3: [8, 9, 0, 0, 0, 0, 0, 0]     # 2 tokens, 6 padding

Efficiency: 9 tokens / 24 total = 37.5%
Waste: 62.5% of computation is on padding!
```

**With packing:**

```
Packed 1: [1, 2, 3, 4, 5, 6, 7, 0]       # seq1 + seq2, 1 padding
Packed 2: [8, 9, 0, 0, 0, 0, 0, 0]       # seq3, 6 padding

Efficiency: 9 tokens / 16 total = 56.25%
Speedup: 1.5x
```

For real datasets with very short sequences (e.g., average 300 tokens, max 2048), packing can give **2-3x speedup**!

### BFD (Best-Fit Decreasing)

**Algorithm:**

1. Sort sequences by length (descending)
2. For each sequence, find the bin with most remaining space that can fit it
3. If no bin fits, create a new bin

**Example:**

```python
# Sequences (length): [3, 4, 2, 1, 5, 2]
# Max length: 8

# Step 1: Sort descending
sorted_seqs = [5, 4, 3, 2, 2, 1]

# Step 2: Pack
bins = []

# seq of length 5
bins[0] = [5]  # remaining: 3

# seq of length 4
bins[1] = [4]  # remaining: 4 (doesn't fit in bin 0)

# seq of length 3
bins[0] = [5, 3]  # remaining: 0 (perfect fit!)

# seq of length 2
bins[1] = [4, 2]  # remaining: 2

# seq of length 2
bins[1] = [4, 2, 2]  # remaining: 0 (perfect fit!)

# seq of length 1
bins[2] = [1]  # no space in other bins

# Result: 3 bins instead of 6 sequences (2x efficiency)
```

**Pros:**
- Optimal packing (minimizes bins)
- Maximum efficiency

**Cons:**
- Requires sorting entire dataset (not streaming-friendly)
- Requires buffering sequences

**Code:** `trl/data_utils.py:pack_dataset()`

### Wrapped Packing

**Algorithm:**

1. Start with empty bin
2. Add sequences until bin is full
3. Start new bin

**Example:**

```python
# Sequences (in order): [3, 4, 2, 1, 5, 2]
# Max length: 8

bins[0] = [3, 4]     # 7 tokens, remaining: 1
bins[1] = [2, 1, 5]  # 8 tokens, remaining: 0
bins[2] = [2]        # 2 tokens, remaining: 6
```

**Pros:**
- Simple
- Works with streaming datasets
- No sorting or buffering needed

**Cons:**
- Less efficient than BFD
- Last bin often has much padding

**When to use:**
- Streaming datasets
- When you want simplicity
- When sequences have similar lengths (BFD doesn't help much)

### Position IDs for packed sequences

**Problem:** When you pack `[seq1, seq2]` into one sequence, the model needs to know they're separate documents.

**Solution:** Reset position IDs at document boundaries.

```python
# Packed: [A, B, C, X, Y, Z]
#         ^^^^^^^ seq1  ^^^^^^^ seq2

# Wrong position IDs (treats as single sequence):
position_ids = [0, 1, 2, 3, 4, 5]

# Correct position IDs (separate sequences):
position_ids = [0, 1, 2, 0, 1, 2]
                ^^^^^^^ seq1  ^^^^^^^ seq2
```

**Why this matters:**

1. **Attention masking**: FlashAttention uses position_ids to prevent tokens from seq1 attending to seq2
2. **Positional embeddings**: RoPE and other position encodings need correct positions

**Code:** `sft_trainer.py:226-250` (`get_position_ids_from_packed_seq_lengths`)

**The trick:**

```python
position_ids = torch.ones(total_length)
position_ids[0] = 0

# Set negative values at document boundaries
position_ids[cumsum(seq_lengths[:-1])] = -(seq_lengths[:-1] - 1)

# Cumsum resets at negative values!
position_ids = position_ids.cumsum(0)
```

**Example:**

```python
seq_lengths = [3, 2, 4]  # 3 docs packed together

# After setting boundaries:
position_ids = [0, 1, 1, -1, 1, 1, 1, -2, 1]
                ^^^^^^^ doc1  ^^^ doc2  ^^^^^^ doc3

# After cumsum:
position_ids = [0, 1, 2,  0, 1, 2, 3,  0, 1]
                ^^^^^^^ doc1  ^^^^^^^ doc2  ^^^ doc3
```

**Magic!** The cumsum automatically resets at boundaries.

---

## Padding-Free Training

### Concept

Instead of padding sequences to the same length, **flatten everything into a single sequence**.

**Standard training:**

```
Batch of 3:
seq1: [A, B, C, 0, 0]
seq2: [X, Y, 0, 0, 0]
seq3: [P, Q, R, S, 0]

Shape: [3, 5]
Padding tokens: 7/15 = 47% waste
```

**Padding-free training:**

```
Flattened batch:
[A, B, C, X, Y, P, Q, R, S]

Shape: [1, 9]
Padding tokens: 0
Waste: 0%
```

**But wait... how does attention work?**

Use **position_ids** and **FlashAttention**!

```python
input_ids =    [A, B, C, X, Y, P, Q, R, S]
position_ids = [0, 1, 2, 0, 1, 0, 1, 2, 3]
                ^^^^^^^ seq1 ^^^ seq2 ^^^^^^^ seq3
```

FlashAttention uses position_ids to:
1. Determine sequence boundaries
2. Prevent cross-sequence attention
3. Apply causal masking correctly

### Requirements

**Padding-free ONLY works with:**
- FlashAttention 2 or 3
- Padding-free mode must be explicitly enabled
- No custom attention masks (uses position_ids instead)

**Why FlashAttention?**

Standard PyTorch attention:
```python
# Expects explicit attention_mask
attn_weights = Q @ K.T
attn_weights = attn_weights + attention_mask  # -inf for masked positions
```

FlashAttention:
```python
# Uses position_ids to infer attention pattern
# Handles variable-length sequences in a batch efficiently
# No need for explicit mask!
```

### Code mapping

**Where padding-free is implemented:**

1. **Collator flattens sequences:** `sft_trainer.py:185-192`
```python
if self.padding_free:
    input_ids = [torch.cat(input_ids, dim=0)]
    position_ids = [torch.cat(position_ids, dim=0)]
```

2. **Model uses position_ids for attention:** Handled by FlashAttention kernel

3. **Loss computation handles flattened batch:** `sft_trainer.py:1121-1122`
```python
elif "position_ids" in inputs:
    entropy = torch.mean(per_token_entropy)  # No masking needed!
```

### Memory savings

**Example: 7B model, batch_size=4, seq_len=2048**

Without padding-free:
```
Sequences: [2048, 1024, 512, 256] tokens
Padded to: [2048, 2048, 2048, 2048]
Total tokens processed: 8192
Activations: ~16 GB
```

With padding-free:
```
Sequences: [2048, 1024, 512, 256] tokens
Flattened: [3840] tokens
Total tokens processed: 3840
Activations: ~7.5 GB
Memory saved: ~8.5 GB (53%)
```

**Combined with packing, you can get 3-4x efficiency!**

---

## Completion-Only Loss

### Motivation

For instruction tuning, we don't want the model to learn to predict prompts. We only want it to learn to generate good completions.

**Bad:**
```
Train on: "What is 2+2? The answer is 4."
Model learns to predict: "What" → "is", "is" → "2", etc.
```

**Good:**
```
Train on: "What is 2+2? The answer is 4."
          ^^^^^^^^^^^ masked   ^^^^^^^^^^^^^^ trained
Model learns to predict: "The" → "answer", "answer" → "is", etc.
```

### Implementation

**Using completion masks:**

```python
# Example
prompt = "What is 2+2?"
completion = " The answer is 4."

input_ids =        [1045, 374, 220, 17, 10, 17, 30, 791, 4320, 374, 220, 19, 13]
                    What  is   [2]  +  [2] ?   The  answer is   [4]  .
completion_mask =  [   0,   0,   0,  0,   0,  0,  1,   1,    1,   1,  1,  1,  1]

# Create labels
labels = input_ids.clone()
labels[completion_mask == 0] = -100

labels =           [-100, -100, -100, -100, -100, -100, 791, 4320, 374, 220, 19, 13]
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ masked (no gradient)
```

**Loss computation (CrossEntropyLoss with ignore_index=-100):**

```python
loss = F.cross_entropy(
    logits.view(-1, vocab_size),
    labels.view(-1),
    ignore_index=-100  # These positions contribute 0 to loss
)
```

**Code:** `sft_trainer.py:217`

```python
output["labels"][completion_mask == 0] = -100
```

### Assistant-only loss (for multi-turn conversations)

For conversations with multiple turns, you want loss only on assistant responses:

```python
messages = [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!"},
    {"role": "user", "content": "How are you?"},
    {"role": "assistant", "content": "I'm good, thanks!"},
]

# After tokenization:
input_ids = [USER_TOKEN, "Hi", ASST_TOKEN, "Hello", USER_TOKEN, "How", "are", "you", ASST_TOKEN, "I'm", "good", "thanks"]
assistant_mask = [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1]
                              ^^^ train here       ^^^^^^^ and here
```

**Code:** Tokenizer's `apply_chat_template` with `return_assistant_tokens_mask=True`

**Why this is better than completion-only:**

- **Completion-only**: Masks the first user message, trains on everything after
- **Assistant-only**: Masks ALL user messages, trains only on assistant responses

This teaches the model to respond to prompts without learning to generate prompts.

---

## Chat Templates

### What is a chat template?

A **Jinja template** that converts structured messages into a tokenizable string.

**Input (structured):**
```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "The answer is 4."}
]
```

**Output (formatted string):**
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
The answer is 4.<|im_end|>
```

### Example templates

**Llama-3:**
```jinja
{% for message in messages %}
    {% if message['role'] == 'system' %}
        {{ '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}
    {% elif message['role'] == 'user' %}
        {{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}
    {% elif message['role'] == 'assistant' %}
        {{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}
    {% endif %}
{% endfor %}
{% if add_generation_prompt %}
    {{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{% endif %}
```

**ChatML (Qwen, etc.):**
```jinja
{% for message in messages %}
    {{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}
{% endfor %}
{% if add_generation_prompt %}
    {{ '<|im_start|>assistant\n' }}
{% endif %}
```

### Special tokens

**Common special tokens:**
- `<|begin_of_text|>`: Start of sequence (BOS)
- `<|end_of_text|>`: End of sequence (EOS)
- `<|im_start|>`, `<|im_end|>`: Message delimiters
- `<|start_header_id|>`, `<|end_header_id|>`: Role markers

**These must be in the tokenizer vocabulary!**

```python
# Check if token exists
tokenizer.convert_tokens_to_ids("<|im_end|>")  # Returns int or None

# Add new tokens (requires model resizing!)
tokenizer.add_special_tokens({"additional_special_tokens": ["<|im_end|>"]})
model.resize_token_embeddings(len(tokenizer))
```

### add_generation_prompt

**What it does:** Adds the assistant turn prefix without the completion.

**Example:**

```python
messages = [{"role": "user", "content": "What is 2+2?"}]

# Without add_generation_prompt:
text = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n"

# With add_generation_prompt=True:
text = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
                                                     ^^^^^^^^^^^^^^^^^^^^^^^^ ready for generation!
```

**When to use:**
- `add_generation_prompt=True`: When preparing prompts for generation
- `add_generation_prompt=False`: When preparing full conversations for training

### Code mapping

**Where chat templates are applied:**

1. **During dataset preparation:** `sft_trainer.py:959-975`
```python
prompt_ids = processing_class.apply_chat_template(
    example["prompt"],
    tokenize=True,
    add_generation_prompt=True,  # Add assistant prefix
)
```

2. **For full conversations:** `sft_trainer.py:1011-1018`
```python
processed = processing_class.apply_chat_template(
    example["messages"],
    return_dict=True,
    tokenize=True,
    return_assistant_tokens_mask=assistant_only_loss,
)
```

### Custom chat templates

**Option 1: Jinja file**

```python
config = SFTConfig(
    chat_template_path="./my_template.jinja"
)
```

**Option 2: Clone from another model**

```python
config = SFTConfig(
    chat_template_path="Qwen/Qwen2.5-7B-Instruct"  # Use Qwen's template
)
```

This will:
1. Load the chat template from Qwen
2. Add any new special tokens to the tokenizer
3. Resize the model's embedding layer
4. Make new tokens trainable (if using PEFT)

**Code:** `sft_trainer.py:645-647`

---

## Performance Optimizations

### Memory optimizations

**1. Gradient checkpointing**

Trades compute for memory by recomputing activations during backward pass.

```python
config = SFTConfig(
    gradient_checkpointing=True  # Default in SFTTrainer
)
```

**Memory saved:** ~40-50% of activation memory
**Speed cost:** ~20-30% slower training

**2. Activation offloading**

Offloads activations to CPU during forward pass, loads back during backward.

```python
config = SFTConfig(
    activation_offloading=True
)
```

**Memory saved:** ~60-70% of activation memory
**Speed cost:** ~50-80% slower (significant!)

**3. Mixed precision (bf16)**

Uses 16-bit floats instead of 32-bit for most computations.

```python
config = SFTConfig(
    bf16=True  # Default in SFTTrainer
)
```

**Memory saved:** ~50% of model memory (not activations)
**Speed gain:** ~2x faster on Ampere+ GPUs

**4. Quantization**

Loads model in 8-bit or 4-bit precision.

```python
config = SFTConfig(
    model_init_kwargs={
        "load_in_8bit": True,  # Or load_in_4bit
        "device_map": "auto",
    }
)
```

**Memory saved:**
- 8-bit: ~50% of model memory
- 4-bit: ~75% of model memory

**Quality cost:**
- 8-bit: Minimal (<1% degradation)
- 4-bit: Noticeable but acceptable (~2-5% degradation)

### Speed optimizations

**1. FlashAttention**

Faster and more memory-efficient attention implementation.

```python
config = SFTConfig(
    model_init_kwargs={
        "attn_implementation": "flash_attention_2"
    }
)
```

**Speed gain:** 2-3x faster for long sequences (>512 tokens)
**Memory saved:** ~20-30% for attention

**2. Packing**

Already covered above. 2-3x speedup for short sequences.

**3. Larger batch sizes**

More compute-efficient due to better GPU utilization.

```python
config = SFTConfig(
    per_device_train_batch_size=8,  # As large as fits in memory
    gradient_accumulation_steps=2,  # Effective batch size: 16
)
```

**Sweet spot:** Batch size that utilizes 80-90% of GPU memory.

**4. Multiprocessing for data loading**

```python
config = SFTConfig(
    dataset_num_proc=8,  # Use 8 CPU cores for preprocessing
    dataloader_num_workers=4,  # Use 4 workers for loading
)
```

**Speed gain:** 1.5-2x faster preprocessing and loading

### Combined example

**Maximum efficiency for 7B model on A100 80GB:**

```python
config = SFTConfig(
    # Memory optimizations
    bf16=True,
    gradient_checkpointing=True,
    model_init_kwargs={
        "attn_implementation": "flash_attention_2",
    },

    # Data efficiency
    packing=True,
    packing_strategy="bfd",
    padding_free=True,

    # Batch size (maximum that fits)
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,  # Effective: 32

    # Data loading
    dataset_num_proc=16,
    dataloader_num_workers=8,

    # Other
    max_length=2048,
)
```

**Expected performance:**
- Throughput: ~8000-10000 tokens/sec on A100
- Memory usage: ~60-70 GB
- Can train with sequences up to 2048 tokens

**For smaller GPUs (e.g., RTX 4090 24GB):**

```python
config = SFTConfig(
    bf16=True,
    gradient_checkpointing=True,
    model_init_kwargs={
        "attn_implementation": "flash_attention_2",
        "load_in_4bit": True,  # Quantize to 4-bit
        "bnb_4bit_compute_dtype": torch.bfloat16,
    },
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,  # Effective: 32
    packing=True,
    max_length=1024,  # Shorter sequences
)

peft_config = LoraConfig(r=16, lora_alpha=32)  # Use LoRA
```

**Expected performance:**
- Throughput: ~2000-3000 tokens/sec
- Memory usage: ~20-22 GB
- Quality: Very close to full fine-tuning

---

## Summary

### Key concepts

1. **SFT = Next-token prediction on instruction data**
   - Loss: `-Σ log p(token|previous tokens)`
   - Trains model to follow instructions

2. **Packing = Efficiency**
   - Combines short sequences to reduce padding
   - BFD (optimal) vs Wrapped (simple)
   - Requires FlashAttention for correctness

3. **Padding-free = More efficiency**
   - Eliminates all padding by flattening batch
   - Requires FlashAttention
   - Combined with packing: 3-4x speedup!

4. **Completion-only loss = Faster learning**
   - Only train on completions, not prompts
   - Uses mask: labels[prompt] = -100

5. **Chat templates = Format control**
   - Converts messages to strings
   - Model-specific (Llama, Qwen, etc.)
   - Must match between training and inference!

### Performance hierarchy

```
Baseline (no optimizations):        1x speed,  100% memory

+ bf16:                             2x speed,   50% memory
+ FlashAttention:                   4x speed,   40% memory
+ Gradient checkpointing:           3x speed,   20% memory
+ Packing:                          9x speed,   20% memory
+ Padding-free:                    12x speed,   15% memory
+ LoRA:                            10x speed,    8% memory
+ 4-bit quantization:               8x speed,    4% memory

(Approximate values for 7B model on A100)
```

The key is **combining** optimizations strategically based on your hardware!
