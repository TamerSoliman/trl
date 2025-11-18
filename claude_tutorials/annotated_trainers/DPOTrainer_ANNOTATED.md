# DPOTrainer: Heavily Annotated Implementation Deep Dive

**File**: `trl/trainer/dpo_trainer.py`
**Lines**: ~2000
**Purpose**: Direct Preference Optimization without explicit reward modeling

This document provides a **line-by-line** deep dive into the critical sections of DPOTrainer, explaining the **What**, **How**, and **Why** of every major operation.

---

## Table of Contents

1. [Overview and Architecture](#overview-and-architecture)
2. [Data Flow](#data-flow)
3. [Core Methods Breakdown](#core-methods-breakdown)
   - [Tokenization: `tokenize_row()`](#tokenization-tokenize_row)
   - [Data Collation: `DataCollatorForPreference`](#data-collation-datacollatorforpreference)
   - [Forward Pass: `concatenated_forward()`](#forward-pass-concatenated_forward)
   - [DPO Loss Calculation: `dpo_loss()`](#dpo-loss-calculation-dpo_loss)
   - [Training Step: `compute_loss()`](#training-step-compute_loss)
4. [Mathematical Foundations](#mathematical-foundations)
5. [Code-to-Math Mapping](#code-to-math-mapping)

---

## Overview and Architecture

### What is DPO?

**Direct Preference Optimization** (DPO) aligns language models to human preferences **without training a separate reward model**. Instead of the traditional RLHF pipeline (SFT → Reward Model → RL), DPO directly optimizes the policy using preference data.

### Key Innovation

DPO reparameterizes the reward function implicitly through the policy itself:

```
r(x, y) = β * log(π_θ(y|x) / π_ref(y|x)) + Z(x)
```

This allows direct optimization of the policy without ever explicitly computing rewards.

### Components Required

1. **Policy Model** (`π_θ`): The model being trained
2. **Reference Model** (`π_ref`): Frozen copy of the initial policy (or PEFT base model)
3. **Preference Data**: Triples of `(prompt, chosen, rejected)`

**Why no reward model?** The preference signal is encoded directly in the loss function through log probability ratios.

---

## Data Flow

```
Input Dataset (prompt, chosen, rejected)
           ↓
    tokenize_row() - Tokenize each component separately
           ↓
    DataCollator - Pad and batch
           ↓
    concatenated_inputs() - Concatenate chosen/rejected for efficiency
           ↓
    concatenated_forward() - Single forward pass for both
           ↓
    Extract log probabilities for chosen and rejected
           ↓
    dpo_loss() - Compute DPO loss from log probabilities
           ↓
    Backpropagate and update policy model
```

---

## Core Methods Breakdown

### Tokenization: `tokenize_row()`

**Location**: Lines 685-752
**Purpose**: Convert raw text into token IDs for prompt, chosen, and rejected responses

#### Code with Line-by-Line Annotation:

```python
@staticmethod
def tokenize_row(
    features: dict[str, str],
    processing_class: PreTrainedTokenizerBase,
    max_prompt_length: int | None = None,
    max_completion_length: int | None = None,
    add_special_tokens: bool = True,
) -> dict[str, list[int]]:
    """
    Tokenize a single row of preference data.

    Args:
        features: Dict with keys 'prompt', 'chosen', 'rejected'
        processing_class: Tokenizer
        max_prompt_length: Max tokens for prompt (truncate if needed)
        max_completion_length: Max tokens for completions
        add_special_tokens: Whether to add BOS/EOS tokens (for enc-dec models)

    Returns:
        Dict with 'prompt_input_ids', 'chosen_input_ids', 'rejected_input_ids'
    """

    # ═══════════════════════════════════════════════════════════════
    # STEP 1: Tokenize each component independently
    # ═══════════════════════════════════════════════════════════════
    # WHY: We need separate token sequences to compute log probabilities
    # for each response independently during the forward pass

    tokenizer = processing_class  # Alias for clarity

    # Tokenize the prompt (query/instruction)
    prompt_input_ids = tokenizer(features["prompt"], add_special_tokens=False)["input_ids"]
    # WHY add_special_tokens=False: We'll add them manually to have full control

    # Tokenize the chosen (preferred) response
    chosen_input_ids = tokenizer(features["chosen"], add_special_tokens=False)["input_ids"]

    # Tokenize the rejected (dispreferred) response
    rejected_input_ids = tokenizer(features["rejected"], add_special_tokens=False)["input_ids"]

    # ═══════════════════════════════════════════════════════════════
    # STEP 2: Add special tokens (encoder-decoder models only)
    # ═══════════════════════════════════════════════════════════════
    # For models like T5/BART that use separate encoder/decoder
    # WHY: Encoder-decoder models need explicit start/end markers

    if add_special_tokens:
        # Add BOS token to prompt if available
        if tokenizer.bos_token_id is not None:
            prompt_input_ids = [tokenizer.bos_token_id] + prompt_input_ids
            # WHY: Marks beginning of encoder input

        # Add EOS token to prompt if available
        if tokenizer.eos_token_id is not None:
            prompt_input_ids = prompt_input_ids + [tokenizer.eos_token_id]
            # WHY: Marks end of encoder input

    # Always add EOS to completions (both chosen and rejected)
    chosen_input_ids = chosen_input_ids + [tokenizer.eos_token_id]
    rejected_input_ids = rejected_input_ids + [tokenizer.eos_token_id]
    # WHY: Models need to learn when to stop generating

    # ═══════════════════════════════════════════════════════════════
    # STEP 3: Truncate if sequences exceed maximum length
    # ═══════════════════════════════════════════════════════════════
    # WHY: Prevent OOM and ensure consistent batch shapes

    if max_prompt_length is not None:
        # Truncate from the LEFT (keep most recent context)
        prompt_input_ids = prompt_input_ids[-max_prompt_length:]
        # WHY -max_prompt_length: For dialogue, recent turns matter most

    if max_completion_length is not None:
        # Truncate from the RIGHT (keep beginning of response)
        chosen_input_ids = chosen_input_ids[:max_completion_length]
        rejected_input_ids = rejected_input_ids[:max_completion_length]
        # WHY [:max_completion_length]: Completions should start coherently

    # ═══════════════════════════════════════════════════════════════
    # RETURN: Three separate token sequences
    # ═══════════════════════════════════════════════════════════════
    return {
        "prompt_input_ids": prompt_input_ids,
        "chosen_input_ids": chosen_input_ids,
        "rejected_input_ids": rejected_input_ids,
    }
```

#### Example Execution:

```python
# Input
features = {
    "prompt": "What is 2+2?",
    "chosen": "2+2 equals 4.",
    "rejected": "I don't know."
}

# After tokenization (example token IDs)
{
    "prompt_input_ids": [1841, 374, 220, 17, 10, 17, 30],  # "What is 2+2?"
    "chosen_input_ids": [17, 10, 17, 17239, 220, 19, 13, 50256],  # "2+2 equals 4." + EOS
    "rejected_input_ids": [40, 1541, 956, 1440, 13, 50256]  # "I don't know." + EOS
}
```

---

### Data Collation: `DataCollatorForPreference`

**Location**: Lines 108-186
**Purpose**: Pad tokenized sequences to same length and batch them efficiently

#### Code with Annotation:

```python
@dataclass
class DataCollatorForPreference(DataCollatorMixin):
    """
    Batches preference data by padding sequences to the same length.

    KEY INSIGHT: We keep prompt, chosen, and rejected SEPARATE in the batch.
    They will be concatenated later in concatenated_inputs() for efficiency.
    """

    pad_token_id: int  # Token to use for padding (usually 0 or tokenizer.pad_token_id)
    return_tensors: str = "pt"  # Always PyTorch tensors

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Args:
            examples: List of tokenized examples from the dataset
                Each example has: prompt_input_ids, chosen_input_ids, rejected_input_ids

        Returns:
            Batched and padded tensors ready for model
        """

        # ═══════════════════════════════════════════════════════════════
        # STEP 1: Convert lists to PyTorch tensors
        # ═══════════════════════════════════════════════════════════════

        prompt_input_ids = [torch.tensor(ex["prompt_input_ids"]) for ex in examples]
        chosen_input_ids = [torch.tensor(ex["chosen_input_ids"]) for ex in examples]
        rejected_input_ids = [torch.tensor(ex["rejected_input_ids"]) for ex in examples]

        # Create attention masks (1 for real tokens, 0 for padding)
        prompt_attention_mask = [torch.ones_like(ids) for ids in prompt_input_ids]
        chosen_attention_mask = [torch.ones_like(ids) for ids in chosen_input_ids]
        rejected_attention_mask = [torch.ones_like(ids) for ids in rejected_input_ids]
        # WHY: Attention masks tell the model which tokens are real vs padding

        # Handle vision model data (images) if present
        if "pixel_values" in examples[0]:
            pixel_values = [torch.tensor(ex["pixel_values"]) for ex in examples]
        if "ref_chosen_logps" in examples[0]:
            # Precomputed reference log probabilities (optimization)
            ref_chosen_logps = torch.tensor([ex["ref_chosen_logps"] for ex in examples])
            ref_rejected_logps = torch.tensor([ex["ref_rejected_logps"] for ex in examples])
            # WHY: If ref model is expensive, precompute these once

        # ═══════════════════════════════════════════════════════════════
        # STEP 2: Pad sequences to the same length within each batch
        # ═══════════════════════════════════════════════════════════════
        # Prompts: padded on the LEFT (for causal LM efficiency)
        # Completions: padded on the RIGHT (standard)

        output = {}
        output["prompt_input_ids"] = pad(
            prompt_input_ids,
            padding_value=self.pad_token_id,
            padding_side="left"
        )
        # WHY left padding for prompts: In causal LM, we want the actual prompt
        # tokens to be at the end, so attention flows left-to-right naturally
        # Example: [PAD, PAD, "What", "is", "2+2", "?"]

        output["prompt_attention_mask"] = pad(
            prompt_attention_mask,
            padding_value=0,  # 0 = ignore this position
            padding_side="left"
        )

        # Chosen completions: pad on right
        output["chosen_input_ids"] = pad(
            chosen_input_ids,
            padding_value=self.pad_token_id,
            padding_side="right"  # Standard padding
        )
        output["chosen_attention_mask"] = pad(
            chosen_attention_mask,
            padding_value=0,
            padding_side="right"
        )

        # Rejected completions: pad on right
        output["rejected_input_ids"] = pad(
            rejected_input_ids,
            padding_value=self.pad_token_id,
            padding_side="right"
        )
        output["rejected_attention_mask"] = pad(
            rejected_attention_mask,
            padding_value=0,
            padding_side="right"
        )

        # Add vision/reference data if present
        if "pixel_values" in examples[0]:
            output["pixel_values"] = pad(pixel_values, padding_value=0.0)
        if "ref_chosen_logps" in examples[0]:
            output["ref_chosen_logps"] = ref_chosen_logps
            output["ref_rejected_logps"] = ref_rejected_logps

        return output
```

#### Example Batch Formation:

```python
# Input: 2 examples with different lengths
examples = [
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

# Output: Padded batch
{
    "prompt_input_ids": tensor([
        [0, 1, 2, 3],  # Left-padded
        [0, 0, 7, 8]   # Left-padded
    ]),
    "prompt_attention_mask": tensor([
        [0, 1, 1, 1],
        [0, 0, 1, 1]
    ]),
    "chosen_input_ids": tensor([
        [4, 5, 0],  # Right-padded
        [9, 10, 0]  # Right-padded
    ]),
    "chosen_attention_mask": tensor([
        [1, 1, 0],
        [1, 1, 0]
    ]),
    "rejected_input_ids": tensor([
        [6, 0, 0],     # Right-padded
        [11, 12, 13]   # No padding needed
    ]),
    "rejected_attention_mask": tensor([
        [1, 0, 0],
        [1, 1, 1]
    ])
}
```

---

### Forward Pass: `concatenated_forward()`

**Location**: Lines 1479-1725
**Purpose**: Efficiently compute log probabilities for both chosen and rejected in a single forward pass

#### Why Concatenate?

**Problem**: We need log probabilities for both chosen AND rejected responses.
**Naive approach**: Two forward passes (one for chosen, one for rejected).
**DPO approach**: Concatenate them vertically, do ONE forward pass, then split the outputs.

**Efficiency Gain**: 2x faster, especially important with FSDP (Fully Sharded Data Parallel).

#### Code with Deep Annotation:

```python
def concatenated_forward(
    self,
    model: nn.Module,
    batch: dict[str, list | torch.LongTensor],
    is_ref_model: bool = False
) -> dict[str, torch.Tensor]:
    """
    Run model on BOTH chosen and rejected responses in a single forward pass.

    Args:
        model: The policy or reference model
        batch: Collated batch from DataCollator
        is_ref_model: If True, this is the reference model (frozen)

    Returns:
        Dict containing:
            - chosen_logps: Log probabilities of chosen responses
            - rejected_logps: Log probabilities of rejected responses
            - (and more for metrics)
    """

    # ═══════════════════════════════════════════════════════════════
    # STEP 1: Concatenate chosen and rejected into single batch
    # ═══════════════════════════════════════════════════════════════
    # This is the CRITICAL efficiency trick

    num_examples = batch["prompt_input_ids"].shape[0]  # Batch size

    concatenated_batch = self.concatenated_inputs(batch, padding_value=self.pad_token_id)
    # This method (lines 946-1022) does:
    # 1. Doubles the prompts: [prompt_1, prompt_2] -> [prompt_1, prompt_2, prompt_1, prompt_2]
    # 2. Concatenates completions: [chosen_1, chosen_2, rejected_1, rejected_2]
    # Result: Batch of size 2*N where first N are chosen, last N are rejected

    # ═══════════════════════════════════════════════════════════════
    # STEP 2: Prepare inputs for the model
    # ═══════════════════════════════════════════════════════════════

    model_kwargs = {"use_cache": False}
    # WHY use_cache=False: We're training, not generating. Caching is for generation.

    # Extract components from concatenated batch
    prompt_input_ids = concatenated_batch["prompt_input_ids"]
    prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
    completion_input_ids = concatenated_batch["completion_input_ids"]
    completion_attention_mask = concatenated_batch["completion_attention_mask"]

    if not self.is_encoder_decoder:
        # ═══════════════════════════════════════════════════════════════
        # CAUSAL LM PATH (GPT-style models)
        # ═══════════════════════════════════════════════════════════════

        # Concatenate prompt + completion into single sequence
        input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
        attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
        # Result shape: [2*batch_size, prompt_len + completion_len]

        # Create loss mask: 1 for completion tokens, 0 for prompt tokens
        loss_mask = torch.cat(
            (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
            dim=1,
        )
        # WHY: We only compute loss on completion tokens, not the prompt
        # Example: [0, 0, 0, 0, 1, 1, 1, 1] = ignore first 4, compute loss on last 4

        # Handle truncation if needed
        if self.max_length is not None and attention_mask.size(1) > self.max_length:
            # Flush left to remove padding, then truncate
            attention_mask, input_ids, loss_mask = flush_left(
                attention_mask, input_ids, loss_mask
            )
            # Truncate to max_length
            input_ids = input_ids[:, :self.max_length]
            attention_mask = attention_mask[:, :self.max_length]
            loss_mask = loss_mask[:, :self.max_length]

        # ═══════════════════════════════════════════════════════════════
        # STEP 3: Forward pass through the model
        # ═══════════════════════════════════════════════════════════════

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **model_kwargs,
        )
        # outputs.logits shape: [2*batch_size, seq_len, vocab_size]
        # These are the raw logits BEFORE softmax

        logits = outputs.logits
        # Get the logits for next token prediction (shift by 1)
        logits = logits[:, :-1, :]  # Remove last position (no next token to predict)
        # Shape: [2*batch_size, seq_len-1, vocab_size]

        # Shift labels (input_ids) by 1 to align with logits
        labels = input_ids[:, 1:].clone()
        # WHY: In causal LM, we predict token at position t+1 given tokens up to t

        # Apply loss mask to labels
        loss_mask = loss_mask[:, :-1].bool()  # Also shift by 1
        labels[~loss_mask] = self.label_pad_token_id  # Mark ignored positions
        # WHY: We set non-completion tokens to -100 (PyTorch's ignore_index)

    # ═══════════════════════════════════════════════════════════════
    # STEP 4: Compute log probabilities for each token
    # ═══════════════════════════════════════════════════════════════
    # This is WHERE THE MAGIC HAPPENS for DPO

    # Get log probabilities using selective_log_softmax
    # This function computes: log P(label | context) for each position
    all_logps = selective_log_softmax(logits, labels)
    # Shape: [2*batch_size, seq_len-1]
    # Each entry is log P(correct_token | context)

    # Sum log probabilities across the sequence (product in prob space)
    # log P(y|x) = sum_t log P(y_t | y_<t, x)
    all_logps_sum = (all_logps * loss_mask).sum(dim=-1)
    # WHY multiply by loss_mask: Only sum over completion tokens, not prompt
    # Shape: [2*batch_size]

    # ═══════════════════════════════════════════════════════════════
    # STEP 5: Split concatenated outputs back into chosen/rejected
    # ═══════════════════════════════════════════════════════════════

    # First half of batch = chosen responses
    # Second half of batch = rejected responses
    chosen_logps = all_logps_sum[:num_examples]
    rejected_logps = all_logps_sum[num_examples:]
    # Shape: [batch_size] each

    # Also split logits for metrics
    chosen_logits = logits[:num_examples]
    rejected_logits = logits[num_examples:]

    # ═══════════════════════════════════════════════════════════════
    # RETURN: Log probabilities and metrics
    # ═══════════════════════════════════════════════════════════════

    return {
        "chosen_logps": chosen_logps,      # log π(chosen | prompt)
        "rejected_logps": rejected_logps,  # log π(rejected | prompt)
        "chosen_logits": chosen_logits,    # Raw logits for analysis
        "rejected_logits": rejected_logits,
        "mean_chosen_logits": chosen_logits.detach().mean(),  # Diagnostic metric
        "mean_rejected_logits": rejected_logits.detach().mean(),
    }
```

#### Visual Representation:

```
Input Batch (size=2):
┌─────────────────┬──────────┬──────────┐
│ Prompt │ Chosen  │ Rejected │
├────────┼─────────┼──────────┤
│ P1     │ C1      │ R1       │
│ P2     │ C2      │ R2       │
└────────┴─────────┴──────────┘

After concatenated_inputs():
┌────────┬────────┐
│ Prompt │ Completion │
├────────┼────────────┤
│ P1     │ C1         │  ← Chosen examples
│ P2     │ C2         │
│ P1     │ R1         │  ← Rejected examples
│ P2     │ R2         │
└────────┴────────────┘

Forward Pass (single call):
   ↓
[All computed together]
   ↓
Split outputs:
chosen_logps  = [log P(C1|P1), log P(C2|P2)]
rejected_logps = [log P(R1|P1), log P(R2|P2)]
```

---

### DPO Loss Calculation: `dpo_loss()`

**Location**: Lines 1024-1245
**Purpose**: THE HEART OF DPO - Convert log probabilities into the DPO loss

This is the most important method in the entire trainer. Let's break down every line.

#### Mathematical Foundation (Bradley-Terry Model):

The DPO loss derives from the Bradley-Terry preference model:

```
P(y_chosen ≻ y_rejected | x) = σ(r(x, y_chosen) - r(x, y_rejected))
```

Where `σ` is the sigmoid function and `r` is the reward.

DPO reparameterizes the reward as:

```
r(x, y) = β * log(π_θ(y|x) / π_ref(y|x))
```

Substituting this into Bradley-Terry and taking the negative log likelihood gives the DPO loss.

#### Code with Complete Annotation:

```python
def dpo_loss(
    self,
    chosen_logps: torch.FloatTensor,      # log π_θ(y_chosen | x)
    rejected_logps: torch.FloatTensor,    # log π_θ(y_rejected | x)
    ref_chosen_logps: torch.FloatTensor,  # log π_ref(y_chosen | x)
    ref_rejected_logps: torch.FloatTensor, # log π_ref(y_rejected | x)
    loss_type: str = "sigmoid",
    model_output: dict = None,
) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """
    Compute the DPO loss from log probabilities.

    This function implements the CORE MATHEMATICAL INSIGHT of DPO:
    We can optimize preferences directly using the implicit reward formulation.

    Args:
        chosen_logps: Shape [batch_size], log probabilities from policy model for chosen responses
        rejected_logps: Shape [batch_size], log probabilities from policy model for rejected responses
        ref_chosen_logps: Shape [batch_size], log probabilities from reference model for chosen
        ref_rejected_logps: Shape [batch_size], log probabilities from reference model for rejected
        loss_type: Which DPO loss variant to use (sigmoid, IPO, hinge, etc.)

    Returns:
        Tuple of (losses, chosen_rewards, rejected_rewards)
    """

    device = self.accelerator.device

    # ═══════════════════════════════════════════════════════════════
    # STEP 1: Compute log ratios (policy / reference)
    # ═══════════════════════════════════════════════════════════════
    # This is π_θ / π_ref in log space = log π_θ - log π_ref

    chosen_logratios = chosen_logps.to(device) - (not self.reference_free) * ref_chosen_logps.to(device)
    # If reference_free=True, the second term becomes 0 (no reference model)
    # Otherwise: log π_θ(y_chosen|x) - log π_ref(y_chosen|x) = log(π_θ/π_ref)

    rejected_logratios = rejected_logps.to(device) - (not self.reference_free) * ref_rejected_logps.to(device)

    # ═══════════════════════════════════════════════════════════════
    # STEP 2: Compute the "logits" for the preference model
    # ═══════════════════════════════════════════════════════════════
    # In DPO, the "logits" are the differences in log ratios

    # Compute: log(π_θ(y_chosen|x)/π_ref(y_chosen|x)) - log(π_θ(y_rejected|x)/π_ref(y_rejected|x))
    logratios = chosen_logps - rejected_logps  # Policy log prob difference

    if not self.reference_free:
        ref_logratios = ref_chosen_logps - ref_rejected_logps  # Reference log prob difference
        logits = logratios - ref_logratios
        # This equals:
        # [log π_θ(chosen) - log π_θ(rejected)] - [log π_ref(chosen) - log π_ref(rejected)]
        # = log[π_θ(chosen)/π_θ(rejected)] - log[π_ref(chosen)/π_ref(rejected)]
    else:
        logits = logratios

    # ═══════════════════════════════════════════════════════════════
    # STEP 3: Compute loss based on loss_type (DEFAULT: sigmoid)
    # ═══════════════════════════════════════════════════════════════
    # The SIGMOID loss is the classic DPO loss from the paper

    if loss_type == "sigmoid":
        # DPO Loss (from paper, Equation 7):
        # L = -log σ(β * logits)
        #
        # Breaking it down:
        # - β (beta): Temperature parameter (controls sharpness, typically 0.1-0.5)
        # - logits: Preference signal (how much more likely is chosen vs rejected?)
        # - σ: Sigmoid function maps to probability
        # - -log: Convert to loss (minimize negative log likelihood)

        losses = (
            -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
            - F.logsigmoid(-self.beta * logits) * self.label_smoothing
        )

        # ── MATHEMATICAL EXPLANATION ──
        #
        # Part 1: -F.logsigmoid(β * logits) * (1 - ε)
        #   This is the main loss term: -log σ(β * logits)
        #   Interpretation: Maximize σ(β * logits), which means:
        #     - When chosen is better, logits > 0 → σ(β*logits) close to 1 → loss close to 0 ✓
        #     - When rejected is better, logits < 0 → σ(β*logits) close to 0 → loss is large ✗
        #   Weighted by (1-ε) where ε is label_smoothing
        #
        # Part 2: -F.logsigmoid(-β * logits) * ε
        #   This is label smoothing: -log σ(-β * logits) = -log(1 - σ(β * logits))
        #   Interpretation: Add a small penalty assuming labels might be wrong
        #   Weighted by ε (typically 0 or small like 0.05)
        #
        # ── WHY LABEL SMOOTHING? ──
        #   Human preferences are noisy. Label smoothing prevents overconfidence.
        #   When ε=0: Pure DPO loss
        #   When ε>0: Regularized DPO loss

    elif loss_type == "ipo":
        # IPO Loss (https://arxiv.org/abs/2310.12036)
        # L = (logits - 1/(2β))²
        #
        # Different from DPO: Uses MSE instead of log-sigmoid
        # WHY: More robust to outliers, less sensitive to β
        losses = (logits - 1 / (2 * self.beta)) ** 2

    elif loss_type == "hinge":
        # Hinge Loss (from SLiC paper)
        # L = max(0, 1 - β * logits)
        #
        # Interpretation: Margin-based loss (similar to SVM)
        # Only penalize when margin < 1
        losses = torch.relu(1 - self.beta * logits)

    # ... other loss variants omitted for brevity ...
    # (See lines 1126-1240 for 15+ other loss types)

    # ═══════════════════════════════════════════════════════════════
    # STEP 4: Compute implicit rewards for monitoring
    # ═══════════════════════════════════════════════════════════════
    # These are NOT used in the loss, just for logging/analysis

    # Implicit reward: r(x,y) = β * log(π_θ(y|x) / π_ref(y|x))
    chosen_rewards = self.beta * (chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = self.beta * (rejected_logps - ref_rejected_logps).detach()
    # WHY .detach(): These are metrics, not part of the computation graph

    # ═══════════════════════════════════════════════════════════════
    # RETURN: Loss and rewards
    # ═══════════════════════════════════════════════════════════════

    return losses, chosen_rewards, rejected_rewards
    # losses: Shape [batch_size], one loss value per example
    # chosen_rewards: Implicit reward for chosen responses (for logging)
    # rejected_rewards: Implicit reward for rejected responses (for logging)
```

#### Line-by-Line Example:

Let's trace through a concrete example with numbers:

```python
# Assume batch_size = 2, β = 0.5

# From forward pass:
chosen_logps    = torch.tensor([-2.3, -3.1])  # log π_θ(y_chosen | x)
rejected_logps  = torch.tensor([-4.2, -5.0])  # log π_θ(y_rejected | x)
ref_chosen_logps = torch.tensor([-2.5, -3.0]) # log π_ref(y_chosen | x)
ref_rejected_logps = torch.tensor([-3.8, -4.5]) # log π_ref(y_rejected | x)

# STEP 1: Compute log ratios
chosen_logratios = chosen_logps - ref_chosen_logps
                 = [-2.3, -3.1] - [-2.5, -3.0]
                 = [0.2, -0.1]
# Interpretation: For example 1, policy gives slightly higher prob than ref
#                 For example 2, policy gives slightly lower prob than ref

rejected_logratios = rejected_logps - ref_rejected_logps
                   = [-4.2, -5.0] - [-3.8, -4.5]
                   = [-0.4, -0.5]
# Interpretation: Policy gives lower prob to rejected for both examples ✓

# STEP 2: Compute logits
logratios = chosen_logps - rejected_logps
          = [-2.3, -3.1] - [-4.2, -5.0]
          = [1.9, 1.9]

ref_logratios = ref_chosen_logps - ref_rejected_logps
              = [-2.5, -3.0] - [-3.8, -4.5]
              = [1.3, 1.5]

logits = logratios - ref_logratios
       = [1.9, 1.9] - [1.3, 1.5]
       = [0.6, 0.4]
# Interpretation: Policy separates chosen/rejected more than reference ✓

# STEP 3: Compute loss (sigmoid variant, no label smoothing)
beta_logits = 0.5 * [0.6, 0.4] = [0.3, 0.2]

sigmoid(beta_logits) = [0.574, 0.550]  # σ(0.3) ≈ 0.574, σ(0.2) ≈ 0.550

logsigmoid(beta_logits) = [-0.554, -0.598]

losses = -logsigmoid(beta_logits)
       = [0.554, 0.598]
# Interpretation: Both losses are relatively low (both < 1) because policy
#                 correctly prefers chosen over rejected ✓

# STEP 4: Compute rewards
chosen_rewards = 0.5 * ([-2.3, -3.1] - [-2.5, -3.0])
               = 0.5 * [0.2, -0.1]
               = [0.1, -0.05]

rejected_rewards = 0.5 * ([-4.2, -5.0] - [-3.8, -4.5])
                 = 0.5 * [-0.4, -0.5]
                 = [-0.2, -0.25]
# Interpretation: Chosen has higher reward than rejected ✓
#                 (Though example 2's chosen has slightly negative reward,
#                  it's still better than rejected's -0.25)
```

---

### Training Step: `compute_loss()`

**Location**: Lines 1815-1836
**Purpose**: Main training loop entry point - orchestrates the full DPO training step

#### Code with Annotation:

```python
def compute_loss(
    self,
    model: PreTrainedModel | nn.Module,
    inputs: dict[str, torch.Tensor | Any],
    return_outputs=False,
    num_items_in_batch=None,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    """
    Main training step called by Hugging Face Trainer on each batch.

    This orchestrates:
    1. Forward pass (policy + reference models)
    2. Loss computation
    3. Metric collection

    Args:
        model: The policy model being trained
        inputs: Batch from DataLoader (after collation)
        return_outputs: Whether to return metrics dict
        num_items_in_batch: For gradient accumulation

    Returns:
        loss: Scalar loss value for backprop
        metrics: (optional) Dict of metrics for logging
    """

    # ═══════════════════════════════════════════════════════════════
    # STEP 1: Compute loss and metrics for this batch
    # ═══════════════════════════════════════════════════════════════

    loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")
    # get_batch_loss_metrics (lines 1727-1813) does:
    # 1. Call concatenated_forward() for policy model
    # 2. Call concatenated_forward() for reference model (if separate)
    # 3. Call dpo_loss() to compute loss
    # 4. Collect metrics (rewards, accuracy, margins, etc.)

    # ═══════════════════════════════════════════════════════════════
    # STEP 2: Move loss to correct device
    # ═══════════════════════════════════════════════════════════════

    loss = loss.to(self.args.device)
    # WHY: In distributed training, ensure loss is on the right device

    # ═══════════════════════════════════════════════════════════════
    # STEP 3: Store metrics for logging
    # ═══════════════════════════════════════════════════════════════

    self.store_metrics(metrics, train_eval="train")
    # These will be averaged and logged at the end of the epoch

    # ═══════════════════════════════════════════════════════════════
    # RETURN: Loss (and optionally metrics)
    # ═══════════════════════════════════════════════════════════════

    if return_outputs:
        return loss, metrics
    return loss
    # The loss is used by Trainer for:
    # 1. loss.backward() - compute gradients
    # 2. optimizer.step() - update weights
```

---

## Mathematical Foundations

### DPO Objective Derivation

Starting from the Bradley-Terry model for preferences:

```
P(y_w ≻ y_l | x) = σ(r(x, y_w) - r(x, y_l))
```

Where:
- `y_w` = winning/chosen response
- `y_l` = losing/rejected response
- `σ` = sigmoid function
- `r` = reward function

The maximum likelihood objective is:

```
max_r E_{(x, y_w, y_l) ~ D} [log σ(r(x, y_w) - r(x, y_l))]
```

DPO's key insight: Reparameterize the reward using the policy:

```
r(x, y) = β log(π_θ(y|x) / π_ref(y|x)) + β log Z(x)
```

Since `Z(x)` doesn't depend on `y`, it cancels in the reward difference:

```
r(x, y_w) - r(x, y_l) = β log(π_θ(y_w|x) / π_ref(y_w|x)) - β log(π_θ(y_l|x) / π_ref(y_l|x))
```

Substituting back:

```
P(y_w ≻ y_l | x) = σ(β log(π_θ(y_w|x) / π_θ(y_l|x)) - β log(π_ref(y_w|x) / π_ref(y_l|x)))
```

The negative log likelihood loss:

```
L_DPO = -E[log σ(β (log π_θ(y_w|x) - log π_θ(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x)))]
```

This is exactly what's computed in `dpo_loss()`!

---

## Code-to-Math Mapping

| Math Symbol | Code Variable | Location | Shape |
|-------------|---------------|----------|-------|
| `π_θ(y_chosen\|x)` | `chosen_logps` | concatenated_forward output | `[batch_size]` |
| `π_θ(y_rejected\|x)` | `rejected_logps` | concatenated_forward output | `[batch_size]` |
| `π_ref(y_chosen\|x)` | `ref_chosen_logps` | from ref model or precomputed | `[batch_size]` |
| `π_ref(y_rejected\|x)` | `ref_rejected_logps` | from ref model or precomputed | `[batch_size]` |
| `β` | `self.beta` | Config parameter | scalar |
| Log ratio | `chosen_logratios` | `chosen_logps - ref_chosen_logps` | `[batch_size]` |
| Logits | `logits` | `logratios - ref_logratios` | `[batch_size]` |
| `σ(β * logits)` | `F.logsigmoid(self.beta * logits)` | In dpo_loss() | `[batch_size]` |
| Loss | `losses` | `-F.logsigmoid(...)` | `[batch_size]` |

---

## Complete Training Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING LOOP                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 1. DATASET LOADING                                          │
│    Raw: {prompt, chosen, rejected}                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ↓ tokenize_row()
┌─────────────────────────────────────────────────────────────┐
│ 2. TOKENIZATION                                             │
│    {prompt_input_ids, chosen_input_ids, rejected_input_ids} │
└─────────────────────────────────────────────────────────────┘
                              │
                              ↓ DataCollator
┌─────────────────────────────────────────────────────────────┐
│ 3. BATCHING & PADDING                                       │
│    Batched tensors with attention masks                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ↓ concatenated_inputs()
┌─────────────────────────────────────────────────────────────┐
│ 4. CONCATENATION                                            │
│    [prompt+chosen, prompt+chosen, ...,                      │
│     prompt+rejected, prompt+rejected, ...]                  │
└─────────────────────────────────────────────────────────────┘
                              │
                     ┌────────┴────────┐
                     │                 │
                     ↓                 ↓
┌──────────────────────────┐  ┌──────────────────────────┐
│ 5a. POLICY FORWARD       │  │ 5b. REFERENCE FORWARD    │
│     concatenated_forward │  │     (if not precomputed) │
│     → chosen_logps       │  │     → ref_chosen_logps   │
│     → rejected_logps     │  │     → ref_rejected_logps │
└──────────────────────────┘  └──────────────────────────┘
                     │                 │
                     └────────┬────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. DPO LOSS COMPUTATION                                     │
│    loss = -log σ(β * (logratios - ref_logratios))          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. BACKPROPAGATION                                          │
│    loss.backward() → Update π_θ only (π_ref frozen)         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 8. METRICS LOGGING                                          │
│    - chosen_rewards vs rejected_rewards                     │
│    - accuracy (chosen_rewards > rejected_rewards?)          │
│    - reward margin                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Takeaways

1. **Efficiency**: DPO computes both chosen and rejected in a single forward pass using concatenation
2. **No Reward Model**: Rewards are implicit in the log probability ratios
3. **Reference Model**: Essential for stability; can be separate model or PEFT base
4. **Loss Variants**: 15+ different loss types for different use cases
5. **Metrics**: Implicit rewards and accuracy provide training insights

---

## See Also

- `reference_guides/DPO_Loss_Reference.md` - Mathematical deep dive
- `reference_guides/RLHF_DPO_Guide.md` - Full data lifecycle
- `annotated_examples/dpo_example_ANNOTATED.md` - Complete training script walkthrough
