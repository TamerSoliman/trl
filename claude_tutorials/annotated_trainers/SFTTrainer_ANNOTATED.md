# SFTTrainer: Complete Implementation Deep Dive

**Source:** `trl/trainer/sft_trainer.py` (1,213 lines)
**Purpose:** Supervised Fine-Tuning - The foundation of all alignment techniques

---

## Table of Contents

1. [Overview](#overview)
2. [Data Collators](#data-collators)
3. [Initialization](#initialization)
4. [Dataset Preparation](#dataset-preparation)
5. [Training Loop](#training-loop)
6. [Configuration](#configuration)

---

## Overview

SFT (Supervised Fine-Tuning) is the **first step** in the alignment pipeline. Before you can do DPO, PPO, or any preference optimization, you need an SFT model.

**What SFT does:**
- Takes a pretrained base model (e.g., Llama-3-8B)
- Fine-tunes it on instruction-following data
- Teaches the model to follow the chat format and respond helpfully

**Key features of TRL's SFTTrainer:**
1. **Packing**: Concatenate multiple short sequences to reduce padding waste
2. **Padding-free training**: Flatten batches for maximum efficiency with FlashAttention
3. **Completion-only loss**: Only compute loss on assistant responses, not prompts
4. **Vision-language support**: Handle multimodal inputs (text + images)
5. **Multiple dataset formats**: Standard text, conversational, prompt-completion

---

## Data Collators

### DataCollatorForLanguageModeling

**Location:** `sft_trainer.py:86-251`

This is the **heart of SFT's data preparation**. It handles batching, padding, and masking.

#### What: Core functionality

```python
@dataclass
class DataCollatorForLanguageModeling(DataCollatorMixin):
    pad_token_id: int
    completion_only_loss: bool = True      # Only compute loss on completions
    padding_free: bool = False             # Flatten batches (no padding)
    pad_to_multiple_of: int | None = None  # For tensor core efficiency
```

**The collator receives:** A list of examples, each containing `input_ids` and optionally:
- `completion_mask`: Binary mask (0 = prompt, 1 = completion)
- `assistant_masks`: Binary mask for assistant turns in conversations
- `seq_lengths`: For packed sequences (multiple docs in one sequence)

**The collator returns:** A batch dictionary with:
- `input_ids`: Padded token IDs
- `labels`: Token IDs with -100 for masked positions
- `attention_mask` (standard mode) OR `position_ids` (padding-free mode)

#### How: The collation process

**Step 1: Convert to tensors (lines 160-181)**

```python
def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
    # Convert to tensor
    input_ids = [torch.tensor(example["input_ids"]) for example in examples]
    if "labels" in examples[0]:
        labels = [torch.tensor(example["labels"]) for example in examples]
    else:
        labels = [torch.tensor(example["input_ids"]) for example in examples]
```

**Why:** Labels default to input_ids for language modeling (predict next token).

**Step 2: Handle padding-free mode (lines 169-176)**

```python
if self.padding_free:
    if "seq_lengths" in examples[0]:
        # For PACKED sequences: compute position_ids from seq_lengths
        position_ids = self.get_position_ids_from_packed_seq_lengths(
            [example["seq_lengths"] for example in examples]
        )
    else:
        # For regular sequences: simple incrementing position_ids
        position_ids = [torch.arange(len(ids)) for ids in input_ids]
else:
    attention_mask = [torch.ones_like(ids) for ids in input_ids]
```

**Why:** FlashAttention uses `position_ids` to know where each sequence starts/ends when multiple sequences are concatenated.

**Example with padding-free packing:**
```
# Two sequences: "Hello world" (2 tokens) + "How are you" (3 tokens)
# Normal mode (with padding):
input_ids:     [[1, 2, 0],      # "Hello world" + PAD
                [3, 4, 5]]      # "How are you"
attention_mask:[[1, 1, 0],
                [1, 1, 1]]

# Padding-free mode (flattened):
input_ids:     [[1, 2, 3, 4, 5]]  # Concatenated!
position_ids:  [[0, 1, 0, 1, 2]]  # Reset at each sequence
seq_lengths:   [[2, 3]]           # Lengths of each sub-sequence
```

**Step 3: Flatten for padding-free (lines 185-192)**

```python
if self.padding_free:
    input_ids = [torch.cat(input_ids, dim=0)]     # Flatten to single sequence
    labels = [torch.cat(labels, dim=0)]
    position_ids = [torch.cat(position_ids, dim=0)]
    if self.completion_only_loss and "completion_mask" in examples[0]:
        completion_mask = [torch.cat(completion_mask, dim=0)]
```

**Why:** All sequences in the batch become ONE long sequence. This eliminates padding waste.

**Step 4: Pad sequences (lines 195-212)**

```python
output["input_ids"] = pad(
    input_ids,
    padding_value=self.pad_token_id,
    padding_side="right",
    pad_to_multiple_of=self.pad_to_multiple_of,
)
output["labels"] = pad(
    labels,
    padding_value=-100,  # IMPORTANT: -100 is ignored in CrossEntropyLoss
    padding_side="right",
    pad_to_multiple_of=self.pad_to_multiple_of
)
```

**Why -100?** PyTorch's `CrossEntropyLoss` ignores targets with value -100. This lets us mask padding tokens.

**Step 5: Apply completion masks (lines 213-222)**

```python
if self.completion_only_loss and "completion_mask" in examples[0]:
    completion_mask = pad(
        completion_mask, padding_value=0, padding_side="right"
    )
    output["labels"][completion_mask == 0] = -100  # Mask prompts!

if "assistant_masks" in examples[0]:
    assistant_masks = pad(assistant_masks, padding_value=0, padding_side="right")
    output["labels"][assistant_masks == 0] = -100  # Mask non-assistant turns!
```

**Why:** This is **completion-only loss**. We don't want the model to learn to predict prompts, only responses.

**Example:**
```python
# Prompt: "What is 2+2?" → Completion: "The answer is 4."
input_ids:        [1045, 374, 220, 17, 10, 17,  30, 791, 4320, 374, 220, 19, 13]
                   What  is   [2]  +  [2] ?   [The answer is  [4]  .]
completion_mask:  [   0,   0,   0,  0,   0,  0,    1,   1,   1,  1,   1, 1,  1]
labels (before):  [1045, 374, 220, 17, 10, 17,  30, 791, 4320, 374, 220, 19, 13]
labels (after):   [-100,-100,-100,-100,-100,-100, 30, 791, 4320, 374, 220, 19, 13]
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ masked!
```

**The model only learns from the completion tokens!**

#### Why: Position IDs for packed sequences

**Method:** `get_position_ids_from_packed_seq_lengths` (lines 226-250)

When you pack multiple documents into one sequence, you need to reset position IDs at each document boundary.

```python
@staticmethod
def get_position_ids_from_packed_seq_lengths(batch_seq_lengths: list[list[int]]):
    """
    Example input: [[3, 2], [4, 1]]  # Batch of 2, first has 2 docs (len 3,2), second has 2 docs (len 4,1)
    Expected output: [
        [0, 1, 2, 0, 1],     # First example: positions for doc1 (3) + doc2 (2)
        [0, 1, 2, 3, 0]      # Second example: positions for doc1 (4) + doc2 (1)
    ]
    """
    # Flatten: [3, 2, 4, 1]
    batch_seq_lengths = torch.tensor(
        [seq_length for seq_lengths in batch_seq_lengths for seq_length in seq_lengths]
    )

    position_ids = torch.ones(sum(example_lengths), dtype=batch_seq_lengths.dtype)
    position_ids[0] = 0  # First position is 0

    # KEY INSIGHT: Set to negative values at document boundaries
    # This causes cumsum to reset the counter!
    position_ids[batch_seq_lengths[:-1].cumsum(0)] = -(batch_seq_lengths[:-1] - 1)

    position_ids = position_ids.cumsum(0)
    return list(position_ids.split(example_lengths))
```

**Concrete example:**
```python
# Packing 2 docs: "Hi there" (2 tokens) + "How are you?" (3 tokens)
seq_lengths = [2, 3]

position_ids (initial): [1, 1, 1, 1, 1]
position_ids[0] = 0:    [0, 1, 1, 1, 1]

# At position 2 (boundary), set to -(2-1) = -1
position_ids[2] = -1:   [0, 1, -1, 1, 1]

# Cumulative sum resets the counter!
cumsum:                 [0, 1, 0, 1, 2]
                         ^^^^ doc1  ^^^^ doc2
```

**Why this matters:** FlashAttention uses position_ids to compute attention correctly for packed sequences. Without proper position_ids, tokens from different documents would attend to each other!

---

### DataCollatorForVisionLanguageModeling

**Location:** `sft_trainer.py:254-462`

Handles vision-language models (VLMs) like Qwen2-VL, LLaVA, etc.

#### What: Key differences from text-only

1. **On-the-fly processing**: Images are converted to pixel values during collation (not preprocessed)
2. **Processor instead of tokenizer**: Uses `ProcessorMixin` which handles both text and images
3. **Two dataset types**:
   - Language modeling: `{"images": [...], "messages": [...]}`
   - Prompt-completion: `{"images": [...], "prompt": ..., "completion": ...}`

#### How: Language modeling collation

**Method:** `_collate_language_modeling` (lines 351-385)

```python
def _collate_language_modeling(self, examples: list[dict[str, Any]]):
    images = [example["images"] for example in examples]

    # Handle conversational data
    if "messages" in examples[0]:
        messages = [
            prepare_multimodal_messages(example["messages"], example["images"])
            for example in examples
        ]
        texts = self.processor.apply_chat_template(messages)
    elif self.dataset_text_field in examples[0]:
        texts = [example[self.dataset_text_field] for example in examples]

    # Process images + text together
    output = self.processor(
        images=images,
        text=texts,
        padding=True,
        padding_side="right",
        truncation=self.max_length is not None,
        max_length=self.max_length,
        return_tensors="pt",
        add_special_tokens=False,  # Chat template already added them
    )

    # Create labels (mask padding only)
    labels = output["input_ids"].clone()
    labels[output["attention_mask"] == 0] = -100
    output["labels"] = labels
    return output
```

**Why `add_special_tokens=False`?** The chat template already added BOS/EOS tokens. Adding them again would create duplicates!

#### How: Prompt-completion collation

**Method:** `_collate_prompt_completion` (lines 387-462)

This is more complex because we need to:
1. Process prompts with images (left padding)
2. Process completions without images (right padding)
3. Concatenate them
4. Apply completion-only masking

```python
def _collate_prompt_completion(self, examples):
    images = [example["images"] for example in examples]
    prompts = [example["prompt"] for example in examples]
    completions = [example["completion"] for example in examples]

    # Process prompts (with images, left-padded)
    processed_prompts = self.processor(
        images=images,
        text=prompts,
        padding=True,
        padding_side="left",  # Left padding for prompts!
        return_tensors="pt",
    )

    # Process completions (no images, right-padded)
    processed_completions = self.processor(
        text=completions,
        padding=True,
        padding_side="right",  # Right padding for completions!
        return_tensors="pt",
    )

    # Concatenate
    prompt_ids = processed_prompts["input_ids"]
    completion_ids = processed_completions["input_ids"]
    input_ids = torch.cat((prompt_ids, completion_ids), dim=1)

    # Create completion mask
    prompt_mask = processed_prompts["attention_mask"]
    completion_mask = processed_completions["attention_mask"]
    attention_mask = torch.cat((prompt_mask, completion_mask), dim=1)
    completion_mask = torch.cat((torch.zeros_like(prompt_mask), completion_mask), dim=1)

    # Flush left to reduce padding
    attention_mask, input_ids, completion_mask = flush_left(
        attention_mask, input_ids, completion_mask
    )

    # Create labels with completion-only masking
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    if self.completion_only_loss:
        labels[completion_mask == 0] = -100

    output = processed_prompts  # Contains pixel_values
    output["input_ids"] = input_ids
    output["attention_mask"] = attention_mask
    output["labels"] = labels
    return output
```

**Why left-pad prompts, right-pad completions?**
- **Left padding prompts**: Aligns the last prompt token with the first completion token
- **Right padding completions**: Standard for generation

**Concrete example:**
```python
# Batch of 2:
# 1. Prompt: "What" (1 token), Completion: "A cat" (2 tokens)
# 2. Prompt: "Describe this" (2 tokens), Completion: "A dog" (2 tokens)

# After processing:
prompt_ids:     [[0, 1],      # [PAD, "What"]  (left-padded)
                 [2, 3]]      # ["Describe", "this"]

completion_ids: [[4, 5],      # ["A", "cat"]  (right-padded)
                 [6, 7]]      # ["A", "dog"]

# After concatenation:
input_ids:      [[0, 1, 4, 5],
                 [2, 3, 6, 7]]

completion_mask:[[0, 0, 1, 1],  # Only completions are 1
                 [0, 0, 1, 1]]

# After flush_left (remove leading padding):
input_ids:      [[1, 4, 5, 0],  # Moved [PAD] to right
                 [2, 3, 6, 7]]

attention_mask: [[1, 1, 1, 0],
                 [1, 1, 1, 1]]

completion_mask:[[0, 1, 1, 0],
                 [0, 0, 1, 1]]
```

---

## Initialization

### SFTTrainer.__init__

**Location:** `sft_trainer.py:576-857`

The initialization does **a lot**: model loading, tokenizer setup, dataset preprocessing, PEFT wrapping, etc.

#### Step 1: Config setup (lines 594-602)

```python
if args is None:
    model_name = model if isinstance(model, str) else get_config_model_id(model.config)
    model_name = model_name.split("/")[-1]
    args = SFTConfig(f"{model_name}-SFT")
elif isinstance(args, TrainingArguments) and not isinstance(args, SFTConfig):
    # Convert TrainingArguments → SFTConfig
    dict_args = args.to_dict()
    args = SFTConfig(**dict_args)
```

**Why:** SFTTrainer needs SFTConfig (which extends TrainingArguments with SFT-specific params).

#### Step 2: Model loading (lines 604-612)

```python
if isinstance(model, str):
    model = create_model_from_path(model, **args.model_init_kwargs or {})
else:
    if args.model_init_kwargs is not None:
        logger.warning(
            "You passed `model_init_kwargs` to the `SFTConfig`, but your model is already "
            "instantiated. The `model_init_kwargs` will be ignored."
        )
```

**Why:** Support both string paths and pre-instantiated models for flexibility.

#### Step 3: Processing class setup (lines 614-626)

```python
if processing_class is None:
    processing_class = AutoProcessor.from_pretrained(get_config_model_id(model.config))

# Determine if it's a VLM or text-only model
if isinstance(processing_class, ProcessorMixin):
    tokenizer = processing_class.tokenizer
    self._is_vlm = True
elif isinstance(processing_class, PreTrainedTokenizerBase):
    tokenizer = processing_class
    self._is_vlm = False
else:
    raise TypeError("The `processing_class` must be either a tokenizer or processor")
```

**Why:** VLMs use Processor (handles images + text), LLMs use Tokenizer (text only).

#### Step 4: EOS token handling (lines 628-637)

```python
if args.eos_token is not None:
    eos_token = args.eos_token
    eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
    if eos_token_id is None:
        raise ValueError(f"The specified `eos_token` ('{eos_token}') is not in vocabulary")
    tokenizer.eos_token_id = eos_token_id
```

**Why:** Some models (e.g., Qwen) use special EOS tokens like `<|im_end|>` instead of standard `</s>`.

#### Step 5: Chat template (lines 639-649)

```python
if args.chat_template_path is not None:
    if os.path.isfile(args.chat_template_path) and args.chat_template_path.endswith((".jinja", ".j2")):
        # Load from Jinja file
        with open(args.chat_template_path, encoding="utf-8") as f:
            processing_class.chat_template = f.read()
        added_tokens = []
    else:
        # Clone from another model
        model, processing_class, added_tokens = clone_chat_template(
            model, processing_class, args.chat_template_path
        )
```

**Why:** Chat templates control how messages are formatted. Different models use different formats.

#### Step 6: VLM validation (lines 651-665)

```python
if self._is_vlm and args.packing:
    raise ValueError("Packing is not supported for VLMs.")
if self._is_vlm and args.padding_free:
    raise ValueError("Padding-free training not supported for VLMs.")
if self._is_vlm and args.assistant_only_loss:
    raise ValueError("Assistant-only loss not supported for VLMs.")
```

**Why:** VLMs have variable-length image tokens that break packing/padding-free assumptions.

#### Step 7: PEFT wrapping (lines 667-700)

```python
if peft_config is not None:
    if added_tokens:
        # Make new tokens trainable
        if peft_config.trainable_token_indices is None:
            peft_config.trainable_token_indices = {"embed_tokens": added_tokens}

        # Make lm_head trainable
        if peft_config.modules_to_save is None:
            peft_config.modules_to_save = ["lm_head"]
        else:
            peft_config.modules_to_save.append("lm_head")

if peft_config is not None or isinstance(model, PeftModel):
    model = prepare_peft_model(model, peft_config, args)
    if model.active_adapter in model.peft_config:
        peft_model_config = model.peft_config[model.active_adapter]
        self.num_virtual_tokens = getattr(peft_model_config, "num_virtual_tokens", 0)
```

**Why:** LoRA adapters need to include new token embeddings and lm_head to learn the new tokens.

**Important:** `num_virtual_tokens` is for Prompt Tuning (adds learnable prefix tokens). We skip these when computing accuracy.

#### Step 8: Data collator setup (lines 702-772)

```python
# Padding-free required for BFD packing
self.padding_free = args.padding_free or (args.packing and args.packing_strategy == "bfd")

# Validate FlashAttention usage
use_flash_attention = model.config._attn_implementation in FLASH_ATTENTION_VARIANTS
if self.padding_free and not use_flash_attention:
    logger.warning(
        "Padding-free training is enabled, but attention is not FlashAttention. "
        "This may lead to unexpected behavior."
    )

# Auto-detect completion-only loss
dataset_sample = next(iter(train_dataset))
if args.completion_only_loss is None:
    self.completion_only_loss = "prompt" in dataset_sample and "completion" in dataset_sample
else:
    self.completion_only_loss = args.completion_only_loss

# Create collator
if data_collator is None and not self._is_vision_dataset:
    pad_token = args.pad_token or tokenizer.pad_token or tokenizer.eos_token
    pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
    data_collator = DataCollatorForLanguageModeling(
        pad_token_id=pad_token_id,
        completion_only_loss=self.completion_only_loss,
        padding_free=self.padding_free,
        pad_to_multiple_of=args.pad_to_multiple_of,
    )
elif data_collator is None and self._is_vision_dataset:
    data_collator = DataCollatorForVisionLanguageModeling(
        processor=processing_class,
        max_length=args.max_length,
        completion_only_loss=self.completion_only_loss,
        pad_to_multiple_of=args.pad_to_multiple_of,
    )
```

**Why auto-detect completion_only_loss?**
- Prompt-completion datasets should use completion-only loss (obvious)
- Language modeling datasets should use full sequence loss (standard next-token prediction)

#### Step 9: Dataset preparation (lines 789-818)

```python
skip_prepare_dataset = (
    args.dataset_kwargs.get("skip_prepare_dataset", False) or self._is_vision_dataset
)

if not skip_prepare_dataset:
    train_dataset = self._prepare_dataset(
        train_dataset, processing_class, args, args.packing, formatting_func, "train"
    )
    if eval_dataset is not None:
        packing = args.packing if args.eval_packing is None else args.eval_packing
        if isinstance(eval_dataset, dict):
            eval_dataset = {
                key: self._prepare_dataset(dataset, processing_class, args, packing, formatting_func, key)
                for key, dataset in eval_dataset.items()
            }
        else:
            eval_dataset = self._prepare_dataset(
                eval_dataset, processing_class, args, packing, formatting_func, "eval"
            )
```

**Why skip for VLMs?** Image processing is expensive; doing it upfront for the entire dataset would be costly. Better to do it on-the-fly.

#### Step 10: Loss function (lines 820-832)

```python
if args.loss_type == "nll":
    pass  # Default CrossEntropyLoss from Transformers
elif args.loss_type == "dft":
    if compute_loss_func is not None:
        raise ValueError("Can't pass both `compute_loss_func` and `loss_type='dft'`")
    compute_loss_func = dft_loss
else:
    raise ValueError(f"Invalid `loss_type` {args.loss_type}")
```

**DFT loss** (Dynamic Fine-Tuning, lines 465-479):
```python
def dft_loss(outputs, labels, num_items_in_batch=None):
    """
    Instead of standard -log p(correct), uses -p(correct) * log p(correct)

    Standard NLL:  -log p(y)
    DFT:           -p(y) * log p(y)  (entropy-weighted)
    """
    logprobs = selective_log_softmax(outputs.logits, shift_labels)
    per_token_loss = -logprobs.exp().detach() * logprobs  # Detach p(y)!
    loss = (per_token_loss * loss_mask).sum() / num_items_in_batch
    return loss
```

**Why DFT?** It upweights uncertain predictions (low probability), downweights confident predictions. This can improve generalization.

---

## Dataset Preparation

### _prepare_dataset

**Location:** `sft_trainer.py:871-1072`

This is where the magic happens: converting raw text/conversations → tokenized sequences ready for training.

#### Pipeline overview

```
Raw dataset
    ↓
1. Apply formatting_func (if any)
    ↓
2. Convert to ChatML (if conversational)
    ↓
3. Apply chat template / add EOS
    ↓
4. Tokenize
    ↓
5. Pack or truncate
    ↓
Ready for training!
```

#### Step 1: Remove None values (lines 882-883)

```python
if isinstance(dataset, Dataset):
    dataset = dataset.with_transform(remove_none_values)
```

**Why?** Parquet/Arrow backends insert `None` for missing nested keys. This breaks tokenization.

#### Step 2: Check if already preprocessed (lines 886-887)

```python
column_names = get_dataset_column_names(dataset)
is_processed = "input_ids" in column_names
```

**Why?** Skip tokenization if the dataset is already tokenized.

#### Step 3: Apply formatting function (lines 896-910)

```python
if formatting_func is not None and not is_processed:
    def _func(example):
        return {"text": formatting_func(example)}

    dataset = dataset.map(_func, batched=False, **map_kwargs)
```

**Example formatting function:**
```python
def format_dolly(example):
    instruction = example["instruction"]
    context = example["context"]
    response = example["response"]

    prompt = f"### Instruction:\n{instruction}\n"
    if context:
        prompt += f"### Context:\n{context}\n"
    prompt += "### Response:\n"

    return prompt + response
```

**Why?** Converts arbitrary dataset structures to text.

#### Step 4: Convert to ChatML (lines 912-923)

```python
first_example = next(iter(dataset))
if is_conversational_from_value(first_example):
    dataset = dataset.map(
        maybe_convert_to_chatml,
        remove_columns="conversations" if "conversations" in column_names else None,
    )
```

**ChatML format:**
```python
# Before (ShareGPT format):
{
    "conversations": [
        {"from": "human", "value": "What is 2+2?"},
        {"from": "gpt", "value": "4"}
    ]
}

# After (ChatML format):
{
    "messages": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"}
    ]
}
```

#### Step 5: Apply chat template or add EOS (lines 925-943)

```python
first_example = next(iter(dataset))
if not is_conversational(first_example):
    # Standard text: just add EOS
    def add_eos(example, eos_token):
        if "text" in example and not example["text"].endswith(eos_token):
            example["text"] = example["text"] + eos_token
        elif "completion" in example and not example["completion"].endswith(eos_token):
            example["completion"] = example["completion"] + eos_token
        return example

    dataset = dataset.map(
        add_eos,
        fn_kwargs={"eos_token": processing_class.eos_token},
        remove_columns="messages" if "messages" in column_names else None,
    )
```

**Why add EOS?** The model needs to learn when to stop generating.

**For conversational data**, the chat template handles EOS internally.

#### Step 6: Tokenize (lines 945-1043)

This is the **longest and most complex** part. It handles 4 cases:
1. Prompt-completion, conversational
2. Prompt-completion, standard text
3. Language modeling, conversational
4. Language modeling, standard text

**Case 1: Prompt-completion + conversational (lines 950-986)**

```python
def tokenize_fn(example, processing_class, dataset_text_field, assistant_only_loss):
    if "prompt" in example:
        if is_conversational(example):
            # Tokenize prompt with generation_prompt=True
            prompt_ids = processing_class.apply_chat_template(
                example["prompt"],
                tokenize=True,
                add_generation_prompt=True,  # Adds "Assistant: " at end
                tools=example.get("tools"),
            )

            # Tokenize prompt + completion together
            prompt_completion_processed = processing_class.apply_chat_template(
                example["prompt"] + example["completion"],
                return_dict=True,
                tokenize=True,
                return_assistant_tokens_mask=assistant_only_loss,
                tools=example.get("tools"),
            )
            prompt_completion_ids = prompt_completion_processed["input_ids"]

            # Create completion mask
            completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids))

            return {
                "input_ids": prompt_completion_ids,
                "completion_mask": completion_mask,
                "assistant_masks": prompt_completion_processed.get("assistant_masks"),
            }
```

**Example:**
```python
prompt = [
    {"role": "user", "content": "What is 2+2?"}
]
completion = [
    {"role": "assistant", "content": "The answer is 4."}
]

# After apply_chat_template(prompt, add_generation_prompt=True):
# "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
prompt_ids = [151644, 872, 198, 3838, 374, 220, 17, 10, 17, 30, 151645, 198, 151644, 77091, 198]

# After apply_chat_template(prompt + completion):
# "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\nThe answer is 4.<|im_end|>"
prompt_completion_ids = [151644, 872, 198, 3838, 374, 220, 17, 10, 17, 30, 151645, 198,
                          151644, 77091, 198, 791, 4320, 374, 220, 19, 13, 151645]

# completion_mask: [0]*15 + [1]*9
completion_mask = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1]
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ prompt
                                                     ^^^^^^^^^^^^^^^^^ completion
```

**Case 2: Prompt-completion + standard text (lines 986-1003)**

Much simpler - just tokenize strings:

```python
else:  # Not conversational
    prompt_ids = processing_class(text=example["prompt"])["input_ids"]
    prompt_completion_ids = processing_class(
        text=example["prompt"] + example["completion"]
    )["input_ids"]

    # Sanity check
    if not prompt_completion_ids[:len(prompt_ids)] == prompt_ids:
        logger.warning(
            "Tokenized prompt doesn't match start of tokenized prompt+completion. "
            "This may be due to whitespace or special token handling."
        )

    completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids))
    return {
        "input_ids": prompt_completion_ids,
        "completion_mask": completion_mask,
    }
```

**Case 3: Language modeling + conversational (lines 1005-1023)**

```python
else:  # Language modeling (no prompt/completion split)
    if is_conversational(example):
        processed = processing_class.apply_chat_template(
            example["messages"],
            return_dict=True,
            tokenize=True,
            return_assistant_tokens_mask=assistant_only_loss,
            tools=example.get("tools"),
        )
        return {
            "input_ids": processed["input_ids"],
            "assistant_masks": processed.get("assistant_masks"),
        }
```

**What is assistant_masks?**

For multi-turn conversations, you often want loss only on assistant responses:

```python
messages = [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!"},
    {"role": "user", "content": "How are you?"},
    {"role": "assistant", "content": "I'm good!"},
]

# Tokenized (simplified):
input_ids:      [USER, "Hi", ASST, "Hello", USER, "How", "are", "you", ASST, "I'm", "good"]
assistant_mask: [0,    0,    0,    1,      0,    0,     0,     0,     0,    1,     1]
                                   ^^^^^^ loss here         ^^^^^^^^^^^^^^^ and here
```

**Case 4: Language modeling + standard text (lines 1023-1024)**

```python
    else:  # Standard text
        return {"input_ids": processing_class(text=example[dataset_text_field])["input_ids"]}
```

Simple next-token prediction on the entire text.

#### Step 7: Pack or truncate (lines 1045-1070)

```python
if packing:
    if args.max_length is None:
        raise ValueError("When packing is enabled, `max_length` can't be `None`.")

    columns = ["input_ids"]
    if "completion_mask" in get_dataset_column_names(dataset):
        columns.append("completion_mask")
    if "assistant_masks" in get_dataset_column_names(dataset):
        columns.append("assistant_masks")

    dataset = dataset.select_columns(columns)
    dataset = pack_dataset(dataset, args.max_length, args.packing_strategy, map_kwargs)

elif args.max_length is not None:
    dataset = truncate_dataset(dataset, args.max_length, map_kwargs)
```

**Packing strategies:**

1. **BFD (Best-Fit Decreasing)**: Sort sequences by length (descending), use bin-packing algorithm
   - Pro: Minimizes padding waste
   - Con: Requires sorting (not suitable for streaming)

2. **Wrapped**: Concatenate sequences until hitting max_length, then start new sequence
   - Pro: Simple, works with streaming
   - Con: More padding waste

**Example packing:**
```python
# Before packing (3 sequences, max_length=10):
seq1: [1, 2, 3]           # length 3
seq2: [4, 5, 6, 7]        # length 4
seq3: [8, 9]              # length 2

# After packing with BFD:
packed1: [1, 2, 3, 4, 5, 6, 7]  # seq1 + seq2 (length 7, fits!)
packed2: [8, 9]                  # seq3 (length 2)

# Stored with seq_lengths for position_ids:
{
    "input_ids": [1, 2, 3, 4, 5, 6, 7],
    "seq_lengths": [3, 4],  # First doc is 3 tokens, second is 4
}
```

---

## Training Loop

### compute_loss

**Location:** `sft_trainer.py:1085-1185`

This method is called for every batch during training and evaluation.

#### What it computes

1. **Loss**: Standard next-token prediction loss (or DFT loss if configured)
2. **Entropy**: Measure of model uncertainty
3. **Token accuracy**: How often the model predicts the correct next token
4. **Auxiliary loss**: For MoE models (load balancing)

#### Step 1: Setup (lines 1095-1105)

```python
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    mode = "train" if self.model.training else "eval"

    # Save labels (they get dropped if using custom loss)
    labels = inputs["labels"]

    # Disable cache (incompatible with gradient checkpointing)
    inputs["use_cache"] = False

    # Call parent class compute_loss
    (loss, outputs) = super().compute_loss(
        model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
    )
```

**Why save labels?** Custom loss functions in Transformers sometimes drop labels from outputs.

**Why `use_cache=False`?** KV-cache is for inference. During training with gradient checkpointing, it causes errors.

#### Step 2: Compute entropy (lines 1107-1126)

```python
if not self.args.use_liger_kernel:  # Liger doesn't return logits
    with torch.no_grad():
        per_token_entropy = entropy_from_logits(outputs.logits)

        # Skip virtual tokens (for Prompt Tuning)
        if self.num_virtual_tokens > 0 and not PREFIX_TUNING:
            per_token_entropy = per_token_entropy[:, self.num_virtual_tokens:]

        # Average over non-padding tokens
        if "attention_mask" in inputs:
            attention_mask = inputs["attention_mask"]
            entropy = torch.sum(per_token_entropy * attention_mask) / attention_mask.sum()
        elif "position_ids" in inputs:  # Padding-free mode
            entropy = torch.mean(per_token_entropy)

        entropy = self.accelerator.gather_for_metrics(entropy).mean().item()
    self._metrics[mode]["entropy"].append(entropy)
```

**What is entropy?**
```
Entropy = -Σ p(token) * log p(token)

Low entropy:  Model is confident (e.g., p=0.9 for one token, p=0.1 for others)
High entropy: Model is uncertain (e.g., p=0.25 for four tokens)
```

**Why track entropy?** It's a good indicator of training progress:
- Early training: High entropy (model is guessing randomly)
- Later training: Low entropy (model is confident)
- If entropy increases: Possible overfitting or divergence

#### Step 3: Track token counts (lines 1128-1139)

```python
if mode == "train":
    if "attention_mask" in inputs:
        num_tokens_in_batch = self.accelerator.gather_for_metrics(
            inputs["attention_mask"].sum()
        ).sum().item()
    elif "position_ids" in inputs:
        local_num_tokens = torch.tensor(inputs["position_ids"].size(1))
        num_tokens_in_batch = self.accelerator.gather_for_metrics(local_num_tokens).sum().item()

    self._total_train_tokens += num_tokens_in_batch
self._metrics[mode]["num_tokens"] = [self._total_train_tokens]
```

**Why track tokens?** Throughput metric (tokens/sec) is more meaningful than batches/sec.

#### Step 4: Compute token accuracy (lines 1141-1183)

```python
if not self.args.use_liger_kernel:
    with torch.no_grad():
        # Handle context parallelism (labels are pre-shifted)
        if "shift_labels" in inputs:
            shift_logits = outputs.logits.contiguous()
            shift_labels = inputs["shift_labels"]
        else:
            # Standard: shift logits and labels
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

        # Skip virtual tokens
        if self.num_virtual_tokens > 0 and not PREFIX_TUNING:
            shift_logits = shift_logits[:, self.num_virtual_tokens:, :]

        # Get predictions
        predictions = shift_logits.argmax(dim=-1)

        # Mask padding tokens
        mask = shift_labels != -100

        # Count correct predictions
        correct_predictions = (predictions == shift_labels) & mask
        total_tokens = mask.sum()
        correct_tokens = correct_predictions.sum()

        # Gather across devices
        correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
        total_tokens = self.accelerator.gather_for_metrics(total_tokens)

        # Compute accuracy
        total_sum = total_tokens.sum()
        accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
        self._metrics[mode]["mean_token_accuracy"].append(accuracy)
```

**Why shift logits and labels?**

Language modeling predicts the **next** token:
```
Input:  [The, cat, sat]
Logits: [logits_0, logits_1, logits_2, logits_3]
         ^^^^^^^^^  ^^^^^^^^^  ^^^^^^^^^  ^^^^^^^^^
         predicts:  "The"      "cat"      "sat"      <EOS>

Labels: [The, cat, sat, <EOS>]

# We want:
logits_0 → predicts "The" (but we don't have the token before "The", so skip)
logits_1 → predicts "cat"  ✓
logits_2 → predicts "sat"  ✓
logits_3 → predicts <EOS>  ✓

# Solution: shift
shift_logits = logits[:-1]  = [logits_0, logits_1, logits_2]
shift_labels = labels[1:]   = ["cat", "sat", <EOS>]
```

**Auxiliary loss for MoE (lines 1180-1183):**
```python
if self.aux_loss_enabled:
    aux_loss = outputs.aux_loss
    aux_loss = self.accelerator.gather_for_metrics(aux_loss).mean().item()
    self._metrics[mode]["aux_loss"].append(aux_loss)
```

MoE models (Mixture of Experts) have a load-balancing loss to ensure all experts are used.

---

## Configuration

### SFTConfig

**Location:** `sft_config.py:22-273`

Extends `TrainingArguments` with SFT-specific parameters.

#### Key parameters

**Model parameters:**
```python
model_init_kwargs: dict[str, Any] | None = None
# Example: {"torch_dtype": torch.bfloat16, "attn_implementation": "flash_attention_2"}

chat_template_path: str | None = None
# Path to Jinja template or model with chat template
```

**Data preprocessing:**
```python
dataset_text_field: str = "text"
# Which column contains the text

max_length: int | None = 1024
# Maximum sequence length

packing: bool = False
# Pack multiple sequences into one to reduce padding

packing_strategy: str = "bfd"
# "bfd" (best-fit decreasing) or "wrapped"

padding_free: bool = False
# Flatten batches (requires FlashAttention)

pad_to_multiple_of: int | None = None
# For tensor core efficiency (e.g., 8 for mixed precision)
```

**Training:**
```python
completion_only_loss: bool | None = None
# If None, auto-detect based on dataset format

assistant_only_loss: bool = False
# Only compute loss on assistant responses (conversational data)

loss_type: str = "nll"
# "nll" (standard) or "dft" (dynamic fine-tuning)

activation_offloading: bool = False
# Offload activations to CPU (saves memory)
```

**Default values (different from TrainingArguments):**
```python
learning_rate: float = 2e-5        # Lower than default 5e-5
logging_steps: float = 10
gradient_checkpointing: bool = True  # Enabled by default
bf16: bool = True                    # If fp16 not set
```

---

## Complete Training Example

Putting it all together:

```python
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

# 1. Load dataset
dataset = load_dataset("trl-lib/Capybara", split="train")

# 2. Configure training
config = SFTConfig(
    output_dir="./llama-3-8b-sft",

    # Data processing
    max_length=2048,
    packing=True,
    packing_strategy="bfd",
    padding_free=True,  # Automatic with BFD packing

    # Training
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,  # Effective batch size: 16
    learning_rate=2e-5,

    # Efficiency
    bf16=True,
    gradient_checkpointing=True,

    # Logging
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=500,
)

# 3. Configure LoRA
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

# 4. Initialize trainer
trainer = SFTTrainer(
    model="meta-llama/Llama-3-8B",
    args=config,
    train_dataset=dataset,
    peft_config=peft_config,
)

# 5. Train!
trainer.train()

# 6. Save
trainer.save_model("./llama-3-8b-sft-final")
```

**Memory estimate (Llama-3-8B with LoRA):**
- Base model (bf16): 16 GB
- Gradients (LoRA only): ~200 MB
- Optimizer states (LoRA): ~400 MB
- Activations (batch_size=2, seq_len=2048, gradient_checkpointing): ~8 GB

**Total: ~25 GB (fits on A100 40GB or RTX 6000 Ada)**

---

## Common Issues and Solutions

### Issue 1: OOM (Out of Memory)

**Symptoms:** CUDA out of memory error

**Solutions:**
1. Enable gradient checkpointing: `gradient_checkpointing=True`
2. Reduce batch size: `per_device_train_batch_size=1`
3. Enable padding-free: `padding_free=True, packing=True`
4. Use activation offloading: `activation_offloading=True`
5. Reduce max_length: `max_length=1024` instead of 2048
6. Use LoRA instead of full fine-tuning
7. Use quantization: `load_in_8bit=True` or `load_in_4bit=True`

### Issue 2: Training is slow

**Symptoms:** Low tokens/sec throughput

**Solutions:**
1. Enable packing: `packing=True` (2-3x speedup for short sequences)
2. Enable FlashAttention: `attn_implementation="flash_attention_2"`
3. Enable padding-free: `padding_free=True`
4. Increase batch size: `per_device_train_batch_size=4`
5. Use bf16: `bf16=True` (faster than fp16 on Ampere+ GPUs)

### Issue 3: Loss not decreasing

**Symptoms:** Loss stays constant or increases

**Diagnostics:**
1. Check token accuracy: Should increase over time
2. Check entropy: Should decrease over time
3. Check if completion_mask is correct: Are you actually training on completions?

**Solutions:**
1. Lower learning rate: Try `5e-6` instead of `2e-5`
2. Check data quality: Are completions correct?
3. Check masking: `completion_only_loss=True` for prompt-completion data
4. Increase training steps: Model might need more time

### Issue 4: Model outputs garbage after training

**Symptoms:** Model generates nonsense or repeats tokens

**Causes:**
1. **Missing EOS tokens**: Model never learned to stop
2. **Wrong chat template**: Format mismatch between training and inference
3. **Overfitting**: Trained too long on small dataset

**Solutions:**
1. Ensure EOS tokens in dataset: `eos_token="<|im_end|>"`
2. Use same chat template for training and inference
3. Add early stopping: `eval_strategy="steps", load_best_model_at_end=True`
4. Reduce epochs: Try `num_train_epochs=0.5`

### Issue 5: Packing breaks attention

**Symptoms:** Model attends across document boundaries

**Cause:** Using packing without FlashAttention

**Solution:**
```python
config = SFTConfig(
    packing=True,
    packing_strategy="bfd",
    model_init_kwargs={
        "attn_implementation": "flash_attention_2"
    }
)
```

FlashAttention uses position_ids to prevent cross-document attention.

---

## Summary

**SFTTrainer workflow:**
1. Load model and tokenizer
2. Prepare dataset (tokenize, pack/truncate)
3. Create data collator (handles batching and masking)
4. Train with next-token prediction loss
5. Track metrics (loss, accuracy, entropy)

**Key innovations:**
- **Packing**: Reduces padding waste (2-3x speedup)
- **Padding-free**: Eliminates all padding (additional 1.5x speedup)
- **Completion-only loss**: Faster convergence for instruction tuning
- **Assistant-only loss**: Multi-turn conversation support
- **VLM support**: Unified interface for vision-language models

**Comparison with standard Trainer:**

| Feature | Transformers Trainer | SFTTrainer |
|---------|---------------------|------------|
| Packing | ❌ | ✅ |
| Padding-free | ❌ | ✅ |
| Completion masking | ❌ | ✅ |
| Chat template support | Partial | Full |
| VLM support | ❌ | ✅ |
| Token accuracy tracking | ❌ | ✅ |

SFT is the foundation. Master it, and DPO/PPO/GRPO will make much more sense!
