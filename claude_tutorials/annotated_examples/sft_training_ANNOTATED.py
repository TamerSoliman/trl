#!/usr/bin/env python3
"""
SFT Training Script - Production-Ready with Comprehensive Annotations

This script demonstrates supervised fine-tuning (SFT) using TRL's SFTTrainer.
Every step is annotated with What, How, and Why explanations.

USAGE:
    # Full fine-tuning
    python sft_training_ANNOTATED.py

    # With LoRA (recommended for <48GB GPU)
    python sft_training_ANNOTATED.py --use_lora

    # With 4-bit quantization (for 24GB GPU)
    python sft_training_ANNOTATED.py --use_lora --load_in_4bit

REQUIREMENTS:
    pip install trl transformers datasets peft accelerate bitsandbytes flash-attn

HARDWARE:
    - Full FT (7B):  A100 80GB or H100
    - LoRA (7B):     A100 40GB or RTX 6000 Ada
    - LoRA+4bit:     RTX 4090 24GB or A10 24GB
"""

import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer


# ==============================================================================
# STEP 1: ARGUMENT PARSING
# ==============================================================================
# WHAT: Parse command-line arguments for configurability
# WHY:  Allows easy experimentation without changing code
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="SFT Training with TRL")

    # Model selection
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2-0.5B-Instruct",
        help="HuggingFace model ID or local path. Examples: "
             "'meta-llama/Llama-3-8B', 'Qwen/Qwen2-7B', 'mistralai/Mistral-7B-v0.3'"
    )
    # EXPLANATION: We use Qwen2-0.5B by default for fast testing.
    # For production, use 7B+ models: Llama-3-8B, Qwen2-7B, etc.

    # Dataset selection
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="trl-lib/Capybara",
        help="Dataset from HuggingFace Hub. Should contain conversational data with 'messages' column."
    )
    # EXPLANATION: Capybara is a high-quality instruction dataset.
    # Alternatives: HuggingFaceH4/ultrachat_200k, OpenAssistant/oasst1, etc.

    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train[:1%]",
        help="Dataset split to use. Use slicing for quick tests: 'train[:100]' or 'train[:1%]'"
    )
    # WHY: Using 1% by default for fast iteration. For production, use 'train'.

    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./sft_output",
        help="Directory to save checkpoints and final model"
    )

    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="Push to HuggingFace Hub under this name (e.g., 'username/model-name')"
    )

    # Training hyperparameters
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of training epochs. For large datasets, 0.5-2 is typical."
    )
    # EXPLANATION: SFT doesn't need many epochs. Even 0.5 epochs can work!
    # Too many epochs → overfitting, repetitive outputs

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate. Range: 5e-6 (safe) to 5e-5 (fast but risky)"
    )
    # EXPLANATION: Lower than pretraining (typically 3e-4).
    # - Full FT: 1e-5 to 5e-5
    # - LoRA: 1e-4 to 5e-4 (10x higher because fewer parameters)

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Batch size per GPU. Increase until OOM, then reduce."
    )
    # WHY: Larger batch = better GPU utilization, but uses more memory.
    # Typical range: 1-8 depending on sequence length and GPU memory.

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Accumulate gradients over N steps. Effective batch = batch_size * accum_steps * num_gpus"
    )
    # EXPLANATION: Simulates larger batches without using more memory.
    # Example: batch_size=2, accum=8, 4 GPUs → effective batch = 64

    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length. Longer = more context but more memory."
    )
    # EXPLANATION: Should match your data's typical length.
    # - Short QA: 512-1024
    # - Long conversations: 2048-4096
    # - Code: 4096-8192

    # Efficiency options
    parser.add_argument(
        "--packing",
        action="store_true",
        default=True,
        help="Pack multiple sequences into one to reduce padding waste (2-3x speedup)"
    )
    # WHY: Most datasets have variable-length sequences. Packing dramatically
    # improves efficiency by concatenating short sequences.

    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        default=True,
        help="Use FlashAttention 2 for faster and more memory-efficient attention"
    )
    # WHY: FlashAttention is 2-3x faster and uses less memory, especially for long sequences.
    # REQUIRED for padding-free training!

    # PEFT (Parameter-Efficient Fine-Tuning) options
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning"
    )
    # EXPLANATION: LoRA trains <1% of parameters (adapters) instead of the full model.
    # - Trains faster
    # - Uses less memory
    # - Quality is 95-98% of full fine-tuning
    # RECOMMENDED for GPUs with <48GB memory

    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank. Higher = more capacity but more memory. Range: 8-64"
    )
    # EXPLANATION:
    # - r=8:  Very parameter-efficient, good for small datasets
    # - r=16: Good balance (default)
    # - r=32-64: More capacity, use for complex tasks or large datasets

    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha (scaling factor). Typically 2x lora_r"
    )
    # EXPLANATION: alpha/r controls the magnitude of LoRA updates.
    # - Higher ratio = stronger adaptation
    # - Lower ratio = more conservative
    # Rule of thumb: alpha = 2 * r

    # Quantization options
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4-bit precision (requires bitsandbytes). 75% memory reduction!"
    )
    # WHY: Enables training large models on smaller GPUs.
    # Example: Train Llama-3-70B on a single A100 80GB!
    # Quality loss: ~2-3% (acceptable for most use cases)

    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8-bit precision. 50% memory reduction, minimal quality loss."
    )
    # EXPLANATION: 8-bit is safer than 4-bit (better quality) but uses more memory.

    # Evaluation
    parser.add_argument(
        "--eval_strategy",
        type=str,
        default="steps",
        choices=["no", "steps", "epoch"],
        help="When to run evaluation. 'steps' = every N steps, 'epoch' = every epoch"
    )

    parser.add_argument(
        "--eval_steps",
        type=int,
        default=100,
        help="Run evaluation every N steps (if eval_strategy='steps')"
    )

    # Checkpointing
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="steps",
        choices=["no", "steps", "epoch"],
        help="When to save checkpoints"
    )

    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps (if save_strategy='steps')"
    )

    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
        help="Maximum number of checkpoints to keep. Older checkpoints are deleted."
    )
    # WHY: Prevents disk space from filling up. Keep 2-3 recent checkpoints.

    return parser.parse_args()


# ==============================================================================
# STEP 2: MODEL AND TOKENIZER LOADING
# ==============================================================================
# WHAT: Load the pretrained model and tokenizer
# HOW:  Use HuggingFace transformers with optional quantization
# WHY:  We start from a pretrained model and fine-tune it on our data
# ==============================================================================

def load_model_and_tokenizer(args):
    """
    Load model with appropriate settings based on arguments.

    Returns:
        tuple: (model_name_or_instance, tokenizer, model_kwargs)
    """

    # ----------------------------------------------------------------------
    # 2.1: Build model_kwargs for from_pretrained()
    # ----------------------------------------------------------------------
    model_kwargs = {
        "torch_dtype": torch.bfloat16,  # Use bf16 for efficiency (on Ampere+ GPUs)
        "device_map": "auto",            # Automatically distribute model across GPUs
    }
    # EXPLANATION:
    # - bf16: Faster and more memory-efficient than fp32, negligible quality loss
    # - device_map="auto": Handles multi-GPU setups automatically

    # FlashAttention
    if args.use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        # WHY: FlashAttention 2 is 2-3x faster and uses less memory.
        # See https://arxiv.org/abs/2307.08691 for details.

    # Quantization
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in bf16 (faster)
            bnb_4bit_use_double_quant=True,          # Double quantization (extra memory savings)
            bnb_4bit_quant_type="nf4",               # NF4 quantization (best for LLMs)
        )
        # EXPLANATION:
        # - load_in_4bit: Weights stored as 4-bit, dequantized on-the-fly
        # - bnb_4bit_compute_dtype: Actual computation happens in bf16 (quality + speed)
        # - use_double_quant: Quantize the quantization constants (extra ~2GB saved)
        # - nf4: Normalized Float 4 (optimized for neural networks)

        print("✓ Using 4-bit quantization (NF4)")

    elif args.load_in_8bit:
        model_kwargs["load_in_8bit"] = True
        print("✓ Using 8-bit quantization")

    # ----------------------------------------------------------------------
    # 2.2: Load tokenizer
    # ----------------------------------------------------------------------
    print(f"Loading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Set padding token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # WHY: Some models (e.g., Llama) don't have a pad token by default.
        # We use EOS as padding (will be masked in loss anyway).

    print(f"✓ Tokenizer loaded. Vocabulary size: {len(tokenizer)}")
    print(f"  - EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    print(f"  - PAD token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")

    # ----------------------------------------------------------------------
    # 2.3: Return model path (not loaded yet - SFTTrainer will load it)
    # ----------------------------------------------------------------------
    # WHY NOT load the model here? Because SFTTrainer needs to:
    # 1. Load the model
    # 2. Apply PEFT if needed
    # 3. Prepare for distributed training
    # It's cleaner to let SFTTrainer handle this.

    return args.model_name, tokenizer, model_kwargs


# ==============================================================================
# STEP 3: DATASET LOADING AND VALIDATION
# ==============================================================================
# WHAT: Load dataset and validate its format
# HOW:  Use HuggingFace datasets library
# WHY:  SFTTrainer expects specific column names ("messages" or "text")
# ==============================================================================

def load_and_validate_dataset(args):
    """
    Load dataset and validate it has the correct format.

    SFTTrainer expects datasets in one of these formats:
    1. Conversational: {"messages": [{"role": "user", "content": "..."}, ...]}
    2. Standard: {"text": "..."}
    3. Prompt-completion: {"prompt": "...", "completion": "..."}
    """

    print(f"\nLoading dataset: {args.dataset_name}...")
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    print(f"✓ Loaded {len(dataset)} examples")

    # ----------------------------------------------------------------------
    # 3.1: Inspect first example
    # ----------------------------------------------------------------------
    first_example = dataset[0]
    print(f"\nDataset format check:")
    print(f"  Columns: {dataset.column_names}")
    print(f"  First example keys: {list(first_example.keys())}")

    # ----------------------------------------------------------------------
    # 3.2: Validate format
    # ----------------------------------------------------------------------
    has_messages = "messages" in first_example
    has_text = "text" in first_example
    has_prompt_completion = "prompt" in first_example and "completion" in first_example

    if has_messages:
        print(f"  ✓ Format: Conversational (messages)")
        print(f"  Example message: {first_example['messages'][0]}")
        # EXPLANATION: This is the preferred format for chat models.
        # SFTTrainer will apply the chat template automatically.

    elif has_text:
        print(f"  ✓ Format: Standard (text)")
        print(f"  Example text (first 100 chars): {first_example['text'][:100]}...")
        # EXPLANATION: Simple text format. Good for continued pretraining or
        # language modeling on unstructured text.

    elif has_prompt_completion:
        print(f"  ✓ Format: Prompt-completion")
        print(f"  Example prompt: {first_example['prompt'][:100]}...")
        # EXPLANATION: Useful when you want explicit control over prompt/completion
        # boundaries. SFTTrainer will use completion-only loss automatically.

    else:
        raise ValueError(
            f"Dataset format not recognized. Expected 'messages', 'text', or 'prompt'+'completion', "
            f"but got columns: {dataset.column_names}"
        )

    # ----------------------------------------------------------------------
    # 3.3: Split into train/eval if needed
    # ----------------------------------------------------------------------
    if args.eval_strategy != "no":
        # Create eval split (10% of data)
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
        print(f"\n✓ Split dataset: {len(train_dataset)} train, {len(eval_dataset)} eval")
    else:
        train_dataset = dataset
        eval_dataset = None
        print(f"\n✓ Using {len(train_dataset)} examples for training (no eval)")

    return train_dataset, eval_dataset


# ==============================================================================
# STEP 4: CONFIGURE PEFT (LoRA)
# ==============================================================================
# WHAT: Configure LoRA if requested
# HOW:  Create LoraConfig with appropriate settings
# WHY:  LoRA enables training large models on consumer GPUs
# ==============================================================================

def get_peft_config(args):
    """
    Create LoRA configuration if requested.

    LoRA (Low-Rank Adaptation) adds small trainable matrices to the model:
        W = W_pretrained + (A @ B)
    where A is [d, r] and B is [r, d], with r << d.

    This means we only train ~0.1% of parameters!
    """

    if not args.use_lora:
        return None

    print("\nConfiguring LoRA:")
    print(f"  - Rank (r): {args.lora_r}")
    print(f"  - Alpha: {args.lora_alpha}")
    print(f"  - Scaling factor: {args.lora_alpha / args.lora_r}")

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,  # Dropout for regularization
        bias="none",         # Don't train biases (not needed for most tasks)
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention weights
            "gate_proj", "up_proj", "down_proj",     # MLP weights
        ],
        # EXPLANATION of target_modules:
        # We apply LoRA to attention and MLP projections.
        # - Attention: q, k, v, o (query, key, value, output)
        # - MLP: gate, up, down (gating, up-projection, down-projection)
        # WHY: These are the most important parameters for adaptation.
        # Embedding and LM head are NOT adapted (too many params, not worth it).
    )

    # Calculate trainable parameters
    # Formula: 2 * r * d * num_layers * num_target_modules
    # For Llama-3-8B: 2 * 16 * 4096 * 32 * 7 ≈ 29M parameters
    # Compare to 8B total = 0.36% trainable!

    print(f"  - Target modules: {peft_config.target_modules}")
    print(f"  ✓ LoRA configured")

    return peft_config


# ==============================================================================
# STEP 5: CONFIGURE TRAINING
# ==============================================================================
# WHAT: Create SFTConfig with all training hyperparameters
# HOW:  Set learning rate, batch size, optimization settings, etc.
# WHY:  Controls the entire training process
# ==============================================================================

def get_training_config(args):
    """
    Create SFTConfig (extends TrainingArguments with SFT-specific options).
    """

    print("\nTraining Configuration:")

    config = SFTConfig(
        # ==================================================================
        # Output and logging
        # ==================================================================
        output_dir=args.output_dir,
        run_name=f"sft-{args.model_name.split('/')[-1]}",
        logging_steps=10,
        logging_first_step=True,
        # WHY log_first_step: See if training starts correctly before waiting 10 steps

        # ==================================================================
        # Training schedule
        # ==================================================================
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # EFFECTIVE BATCH SIZE = batch_size * gradient_accumulation_steps * num_gpus
        # Example: 2 * 8 * 4 = 64

        # ==================================================================
        # Optimization
        # ==================================================================
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",  # Cosine decay (standard for fine-tuning)
        warmup_ratio=0.03,            # Warm up for first 3% of steps
        # WHY warmup: Prevents instability at the start of training.
        # Learning rate gradually increases from 0 to max over warmup period.

        weight_decay=0.01,  # L2 regularization (prevents overfitting)
        # EXPLANATION: 0.01 is standard. Higher (0.1) = more regularization.

        max_grad_norm=1.0,  # Gradient clipping (prevents exploding gradients)
        # WHY: Clips gradients to maximum norm of 1.0. Essential for stability.

        # ==================================================================
        # Precision and efficiency
        # ==================================================================
        bf16=torch.cuda.is_bf16_supported(),  # Use bf16 if GPU supports it
        fp16=False,  # Don't use fp16 (bf16 is better on modern GPUs)
        # EXPLANATION:
        # - bf16: Better for training (wider dynamic range)
        # - fp16: Can be faster on older GPUs (V100) but less stable

        gradient_checkpointing=True,  # Save memory by recomputing activations
        # MEMORY SAVED: ~40-50% of activation memory
        # SPEED COST: ~20-30% slower
        # CONCLUSION: Worth it for large models!

        # ==================================================================
        # SFT-specific options
        # ==================================================================
        max_seq_length=args.max_seq_length,  # Maximum sequence length
        packing=args.packing,                 # Pack sequences to reduce padding
        packing_strategy="bfd" if args.packing else None,
        # EXPLANATION of BFD (Best-Fit Decreasing):
        # 1. Sort sequences by length (descending)
        # 2. Pack into bins using best-fit algorithm
        # 3. Minimizes wasted space (padding)
        # SPEEDUP: 2-3x for datasets with variable-length sequences!

        dataset_text_field="text",  # Column name for text data (if not using messages)
        # NOTE: Ignored if dataset has "messages" column

        # ==================================================================
        # Evaluation
        # ==================================================================
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps if args.eval_strategy == "steps" else None,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        # WHY same batch size: Consistent memory usage

        # ==================================================================
        # Checkpointing
        # ==================================================================
        save_strategy=args.save_strategy,
        save_steps=args.save_steps if args.save_strategy == "steps" else None,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True if args.eval_strategy != "no" else False,
        # WHY load_best_model: Prevents overfitting by using the checkpoint with best eval loss

        metric_for_best_model="eval_loss",  # Metric to determine "best" model
        greater_is_better=False,             # Lower loss = better

        # ==================================================================
        # Model loading (passed to AutoModel.from_pretrained)
        # ==================================================================
        model_init_kwargs=None,  # We'll pass this separately to allow model string input

        # ==================================================================
        # Hub integration
        # ==================================================================
        push_to_hub=args.hub_model_id is not None,
        hub_model_id=args.hub_model_id,
        hub_private_repo=True,  # Make repo private by default
    )

    # Print effective batch size
    effective_batch = (
        args.per_device_train_batch_size * args.gradient_accumulation_steps * torch.cuda.device_count()
    )
    print(f"  - Effective batch size: {effective_batch}")
    print(f"    ({args.per_device_train_batch_size} batch × {args.gradient_accumulation_steps} accum × {torch.cuda.device_count()} GPUs)")
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"  - Max sequence length: {args.max_seq_length}")
    print(f"  - Packing: {'✓ BFD' if args.packing else '✗'}")
    print(f"  - Gradient checkpointing: ✓")
    print(f"  - Precision: {'bf16' if config.bf16 else 'fp32'}")

    return config


# ==============================================================================
# STEP 6: INITIALIZE TRAINER
# ==============================================================================
# WHAT: Create SFTTrainer instance
# HOW:  Pass model, config, dataset, and PEFT config
# WHY:  SFTTrainer handles all the complexity of SFT training
# ==============================================================================

def initialize_trainer(model_name, tokenizer, model_kwargs, train_dataset, eval_dataset, training_config, peft_config):
    """
    Initialize SFTTrainer.

    SFTTrainer will:
    1. Load the model (with quantization if configured)
    2. Apply PEFT (LoRA) if configured
    3. Prepare data collator
    4. Set up distributed training
    5. Initialize optimizer and scheduler
    """

    print("\nInitializing SFTTrainer...")

    trainer = SFTTrainer(
        model=model_name,  # Can be string (model will be loaded) or PreTrainedModel instance
        args=training_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,  # New name for tokenizer (supports Processors too)
    )
    # EXPLANATION of what happens inside SFTTrainer.__init__:
    # 1. If model is string, load with AutoModelForCausalLM.from_pretrained(**model_kwargs)
    # 2. If peft_config provided, wrap model with get_peft_model()
    # 3. Create DataCollatorForLanguageModeling (handles padding, masking, etc.)
    # 4. Prepare dataset (tokenize, pack, truncate)
    # 5. Initialize Trainer (parent class from transformers)

    # Note: We pass model_init_kwargs separately via training_config because
    # SFTTrainer expects it there (not as a direct argument)
    if model_kwargs:
        trainer.args.model_init_kwargs = model_kwargs

    print("✓ Trainer initialized")

    # Print model info
    if hasattr(trainer.model, 'print_trainable_parameters'):
        # This method is added by PEFT
        trainer.model.print_trainable_parameters()
    else:
        total_params = sum(p.numel() for p in trainer.model.parameters())
        trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    return trainer


# ==============================================================================
# STEP 7: TRAINING
# ==============================================================================
# WHAT: Run the actual training loop
# HOW:  Call trainer.train()
# WHY:  This is where the magic happens!
# ==============================================================================

def train(trainer):
    """
    Execute training.

    trainer.train() will:
    1. Loop over epochs
    2. For each batch:
       a. Forward pass (compute loss)
       b. Backward pass (compute gradients)
       c. Optimizer step (update weights)
    3. Periodically: evaluate, save checkpoints, log metrics
    4. Return training results
    """

    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)

    # Check if resuming from checkpoint
    # (SFTTrainer automatically detects checkpoints in output_dir)
    train_result = trainer.train()

    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)

    # Print training metrics
    metrics = train_result.metrics
    print("\nFinal Training Metrics:")
    for key, value in metrics.items():
        print(f"  - {key}: {value}")

    return train_result


# ==============================================================================
# STEP 8: SAVE MODEL
# ==============================================================================
# WHAT: Save the fine-tuned model and tokenizer
# HOW:  Use trainer.save_model()
# WHY:  So we can use the model later for inference or further training
# ==============================================================================

def save_model(trainer, tokenizer, args):
    """
    Save the trained model and tokenizer.

    For LoRA models, this saves:
    - adapter_config.json (LoRA configuration)
    - adapter_model.safetensors (LoRA weights, typically 50-200MB)

    For full fine-tuning, this saves:
    - config.json (model configuration)
    - model.safetensors (full model weights, typically 13-28GB for 7B)
    """

    print(f"\nSaving model to {args.output_dir}...")

    # Save model
    trainer.save_model(args.output_dir)

    # Save tokenizer (important for chat templates!)
    tokenizer.save_pretrained(args.output_dir)

    print("✓ Model and tokenizer saved")

    # Print what was saved
    import os
    files = os.listdir(args.output_dir)
    print(f"\nSaved files:")
    for f in sorted(files):
        size = os.path.getsize(os.path.join(args.output_dir, f)) / (1024**2)  # MB
        print(f"  - {f} ({size:.1f} MB)")


# ==============================================================================
# STEP 9: PUSH TO HUB (OPTIONAL)
# ==============================================================================
# WHAT: Upload model to HuggingFace Hub
# HOW:  Use trainer.push_to_hub()
# WHY:  Share your model with the world or your team!
# ==============================================================================

def push_to_hub(trainer, args):
    """
    Push the model to HuggingFace Hub.

    Requires:
    - HuggingFace account
    - HF_TOKEN environment variable or huggingface-cli login
    """

    if args.hub_model_id is None:
        return

    print(f"\nPushing model to HuggingFace Hub: {args.hub_model_id}...")

    trainer.push_to_hub()

    print(f"✓ Model pushed to https://huggingface.co/{args.hub_model_id}")


# ==============================================================================
# STEP 10: INFERENCE TEST (OPTIONAL)
# ==============================================================================
# WHAT: Test the fine-tuned model with a sample prompt
# HOW:  Generate text using the model's generate() method
# WHY:  Verify that the model works and produces sensible outputs
# ==============================================================================

def test_inference(trainer, tokenizer):
    """
    Run a quick inference test to verify the model works.
    """

    print("\n" + "="*70)
    print("INFERENCE TEST")
    print("="*70)

    # Sample prompt (adjust based on your model's chat template)
    messages = [
        {"role": "user", "content": "What is machine learning?"}
    ]

    # Apply chat template
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        # Fallback for models without chat template
        prompt = "What is machine learning?\n\nAnswer:"

    print(f"Prompt:\n{prompt}\n")

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(trainer.model.device)

    # Generate
    print("Generating...")
    outputs = trainer.model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"\nGenerated response:\n{response}")
    print("\n" + "="*70)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """
    Main training pipeline:
    1. Parse arguments
    2. Load model and tokenizer
    3. Load dataset
    4. Configure PEFT (LoRA)
    5. Configure training
    6. Initialize trainer
    7. Train
    8. Save
    9. Push to hub (optional)
    10. Test inference (optional)
    """

    # Step 1: Parse arguments
    args = parse_args()
    print("SFT Training Script")
    print("="*70)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Output: {args.output_dir}")
    print(f"Mode: {'LoRA' if args.use_lora else 'Full Fine-Tuning'}")
    if args.load_in_4bit:
        print(f"Quantization: 4-bit")
    elif args.load_in_8bit:
        print(f"Quantization: 8-bit")
    print("="*70)

    # Step 2: Load model and tokenizer
    model_name, tokenizer, model_kwargs = load_model_and_tokenizer(args)

    # Step 3: Load dataset
    train_dataset, eval_dataset = load_and_validate_dataset(args)

    # Step 4: Configure PEFT
    peft_config = get_peft_config(args)

    # Step 5: Configure training
    training_config = get_training_config(args)

    # Step 6: Initialize trainer
    trainer = initialize_trainer(
        model_name=model_name,
        tokenizer=tokenizer,
        model_kwargs=model_kwargs,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_config=training_config,
        peft_config=peft_config,
    )

    # Step 7: Train
    train(trainer)

    # Step 8: Save
    save_model(trainer, tokenizer, args)

    # Step 9: Push to hub
    push_to_hub(trainer, args)

    # Step 10: Test inference
    test_inference(trainer, tokenizer)

    print("\n✓ All done!")


if __name__ == "__main__":
    main()


# ==============================================================================
# USAGE EXAMPLES
# ==============================================================================

"""
# Example 1: Quick test with small model
python sft_training_ANNOTATED.py \
    --model_name Qwen/Qwen2-0.5B-Instruct \
    --dataset_name trl-lib/Capybara \
    --dataset_split train[:100] \
    --num_train_epochs 1 \
    --output_dir ./test_output

# Example 2: Full fine-tuning of Llama-3-8B (requires A100 80GB)
python sft_training_ANNOTATED.py \
    --model_name meta-llama/Llama-3-8B \
    --dataset_name trl-lib/Capybara \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --output_dir ./llama3-8b-sft

# Example 3: LoRA fine-tuning of Llama-3-8B (fits on RTX 6000 Ada 48GB)
python sft_training_ANNOTATED.py \
    --model_name meta-llama/Llama-3-8B \
    --dataset_name trl-lib/Capybara \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --output_dir ./llama3-8b-lora

# Example 4: 4-bit LoRA on consumer GPU (RTX 4090 24GB)
python sft_training_ANNOTATED.py \
    --model_name meta-llama/Llama-3-8B \
    --dataset_name trl-lib/Capybara \
    --use_lora \
    --load_in_4bit \
    --lora_r 16 \
    --lora_alpha 32 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_seq_length 1024 \
    --output_dir ./llama3-8b-qlora

# Example 5: Push to HuggingFace Hub
python sft_training_ANNOTATED.py \
    --model_name Qwen/Qwen2-7B \
    --dataset_name trl-lib/Capybara \
    --use_lora \
    --hub_model_id username/qwen2-7b-capybara \
    --output_dir ./qwen2-7b-sft

# Example 6: Multi-GPU training (automatically uses all available GPUs)
# No changes needed - just run with torchrun or accelerate!
accelerate launch sft_training_ANNOTATED.py \
    --model_name meta-llama/Llama-3-8B \
    --dataset_name trl-lib/Capybara \
    --use_lora \
    --output_dir ./llama3-8b-lora-multi-gpu
"""
