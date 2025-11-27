"""
Annotated DPO Training Script
==============================

This script demonstrates a complete DPO training pipeline with detailed
annotations explaining every step.

Based on: trl/scripts/dpo.py

Purpose: Train a model using Direct Preference Optimization to align it
         with human preferences without needing a separate reward model.

Prerequisites:
- A base model (preferably SFT'd)
- Preference dataset with (prompt, chosen, rejected) triples
"""

# ═══════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import DPOConfig, DPOTrainer

# ═══════════════════════════════════════════════════════════════════
# STEP 1: LOAD THE BASE MODEL
# ═══════════════════════════════════════════════════════════════════
# This should ideally be a model that's already been SFT'd (supervised
# fine-tuned) on instruction data. Starting from a raw pretrained model
# works but results may be worse.

model_name = "Qwen/Qwen2-0.5B-Instruct"  # ~500M parameters, good for testing

print(f"Loading policy model: {model_name}")
policy_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # Use bf16 for efficiency (if supported)
    attn_implementation="flash_attention_2",  # Faster attention (optional)
    device_map="auto"  # Automatically distribute across GPUs
)

# WHY these settings?
# - torch_dtype=bfloat16: Reduces memory by 50% vs fp32, minimal quality loss
# - flash_attention_2: 2-3x faster attention computation
# - device_map="auto": Handles multi-GPU placement automatically

# ═══════════════════════════════════════════════════════════════════
# STEP 2: CREATE REFERENCE MODEL
# ═══════════════════════════════════════════════════════════════════
# The reference model (π_ref) is a frozen copy of the initial policy.
# It provides the baseline that the policy is compared against in the
# DPO loss function.

print(f"Loading reference model: {model_name}")
reference_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)

# IMPORTANT: The reference model is NEVER updated during training.
# It stays frozen to provide a stable comparison point.

# ALTERNATIVE (for memory efficiency):
# If using PEFT/LoRA, you can set ref_model=None and the trainer will
# use the base model (with adapters disabled) as the reference.
# This saves ~14GB of memory for a 7B model!

# ═══════════════════════════════════════════════════════════════════
# STEP 3: LOAD TOKENIZER
# ═══════════════════════════════════════════════════════════════════

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ensure padding token is set (required for batching)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    # WHY: Some tokenizers (like GPT-2) don't have a pad token by default.
    # We use EOS as padding since we mask it out anyway.

# ═══════════════════════════════════════════════════════════════════
# STEP 4: LOAD PREFERENCE DATASET
# ═══════════════════════════════════════════════════════════════════
# DPO requires a dataset of preference pairs. Each example must have:
# - prompt: The instruction/question
# - chosen: The preferred response
# - rejected: The dispreferred response

dataset_name = "trl-lib/ultrafeedback_binarized"

print(f"Loading dataset: {dataset_name}")
dataset = load_dataset(dataset_name, split="train[:1000]")  # Small subset for demo

# Let's inspect a sample
print("\n" + "="*70)
print("SAMPLE DATA POINT:")
print("="*70)
example = dataset[0]
print(f"Prompt: {example['prompt'][:100]}...")
print(f"\nChosen: {example['chosen'][:100]}...")
print(f"\nRejected: {example['rejected'][:100]}...")
print("="*70 + "\n")

# WHAT MAKES GOOD PREFERENCE DATA?
# 1. Clear distinction between chosen and rejected
# 2. Diverse prompts covering different tasks
# 3. High-quality chosen responses (grammatical, helpful, accurate)
# 4. Realistic rejected responses (not obviously bad)
# 5. Sufficient volume (10K+ pairs recommended)

# ═══════════════════════════════════════════════════════════════════
# STEP 5: CONFIGURE DPO TRAINING
# ═══════════════════════════════════════════════════════════════════
# DPO has many hyperparameters. Let's break down the most important ones.

config = DPOConfig(
    # ──────────────────────────────────────────────────────────────
    # OUTPUT AND LOGGING
    # ──────────────────────────────────────────────────────────────
    output_dir="./qwen2-dpo-output",
    run_name="qwen2-dpo-demo",
    logging_steps=10,  # Log every 10 steps
    save_steps=100,    # Save checkpoint every 100 steps
    report_to="wandb",  # Options: "wandb", "tensorboard", "none"

    # ──────────────────────────────────────────────────────────────
    # DPO-SPECIFIC PARAMETERS
    # ──────────────────────────────────────────────────────────────
    beta=0.1,
    # β (beta): The temperature parameter in the DPO loss.
    # Controls how strongly preferences are enforced.
    # - Lower β (0.05-0.1): More conservative, softer preferences
    # - Higher β (0.3-0.5): More aggressive, sharper preferences
    # RULE OF THUMB: Start with 0.1, increase if model isn't learning

    loss_type="sigmoid",
    # The DPO loss variant to use. Options include:
    # - "sigmoid": Original DPO loss (recommended default)
    # - "ipo": IPO loss (more robust to noise)
    # - "hinge": SLiC hinge loss (margin-based)
    # - "robust": Robust DPO (handles label noise better)
    # See DPO_Loss_Reference.md for full list and explanations

    label_smoothing=0.0,
    # Add regularization assuming labels might be noisy.
    # Typical range: 0.0 (no smoothing) to 0.1 (moderate smoothing)
    # USE WHEN: Your preference data might contain errors

    # ──────────────────────────────────────────────────────────────
    # SEQUENCE LENGTH PARAMETERS
    # ──────────────────────────────────────────────────────────────
    max_length=512,
    # Maximum total sequence length (prompt + completion).
    # Sequences longer than this will be truncated.
    # TRADE-OFF: Longer = more context but more memory

    max_prompt_length=256,
    # Maximum length for prompts alone.
    # STRATEGY: Allocate more tokens to completions than prompts

    max_completion_length=256,
    # Maximum length for completions alone.
    # NOTE: max_prompt_length + max_completion_length can exceed max_length;
    # final concatenated sequence is truncated to max_length

    truncation_mode="keep_end",
    # How to truncate when prompt+completion exceeds max_length:
    # - "keep_end": Keep the end of the sequence (recent context)
    # - "keep_start": Keep the beginning (for summary-style tasks)

    # ──────────────────────────────────────────────────────────────
    # OPTIMIZATION PARAMETERS
    # ──────────────────────────────────────────────────────────────
    learning_rate=5e-7,
    # DPO typically uses VERY low learning rates (1e-7 to 1e-6).
    # WHY: The model is already well-trained (from SFT), we're doing
    # fine-grained alignment, not learning from scratch.
    # START: 5e-7 for 7B models, 1e-6 for smaller models

    per_device_train_batch_size=2,
    # Batch size per GPU. Keep small for memory efficiency.
    # RECOMMENDATION: 1-4 for 7B models, 4-8 for smaller models

    gradient_accumulation_steps=8,
    # Accumulate gradients over N steps before updating.
    # EFFECTIVE BATCH SIZE = per_device_batch_size * num_gpus * gradient_accumulation_steps
    # EXAMPLE: 2 * 1 * 8 = 16 effective batch size

    num_train_epochs=3,
    # How many passes through the dataset.
    # DPO typically needs 1-3 epochs.
    # MORE EPOCHS: Better alignment but risk of overfitting

    warmup_ratio=0.1,
    # Fraction of training to use for learning rate warmup.
    # Helps stabilize training at the start.

    # ──────────────────────────────────────────────────────────────
    # MEMORY OPTIMIZATION
    # ──────────────────────────────────────────────────────────────
    gradient_checkpointing=True,
    # Trade computation for memory by recomputing activations.
    # SAVES: ~40% memory
    # COSTS: ~20% slower training
    # RECOMMENDATION: Always use for large models

    bf16=True,  # Use bfloat16 precision
    # WHY: Reduces memory, minimal quality loss, faster on modern GPUs
    # NOTE: Requires Ampere+ GPUs (A100, A6000, RTX 30xx/40xx)

    # ──────────────────────────────────────────────────────────────
    # ADVANCED: REFERENCE MODEL OPTIMIZATION
    # ──────────────────────────────────────────────────────────────
    precompute_ref_log_probs=False,
    # If True, compute reference log probs ONCE before training starts,
    # then cache them. This allows removing the ref model from memory.
    # SAVES: ~14GB for 7B model
    # USE WHEN: Memory is tight and dataset fits in RAM
    # NOTE: Can't use with data augmentation

    # ──────────────────────────────────────────────────────────────
    # EVALUATION
    # ──────────────────────────────────────────────────────────────
    eval_strategy="steps",  # Evaluate every N steps
    eval_steps=50,          # Evaluate every 50 steps
    # Evaluation computes metrics on a held-out set to monitor
    # overfitting and training progress
)

# ═══════════════════════════════════════════════════════════════════
# STEP 6: INITIALIZE THE TRAINER
# ═══════════════════════════════════════════════════════════════════

print("Initializing DPO trainer...")
trainer = DPOTrainer(
    model=policy_model,        # Model being trained
    ref_model=reference_model,  # Frozen reference model
    args=config,               # Training configuration
    train_dataset=dataset,     # Preference data
    processing_class=tokenizer # Tokenizer for text processing
)

# WHAT HAPPENS DURING INITIALIZATION?
# 1. Dataset preprocessing:
#    - Extract prompts from conversational data
#    - Apply chat template
#    - Tokenize all text
#    - Create attention masks
# 2. Model setup:
#    - Prepare for distributed training (if multi-GPU)
#    - Set up gradient checkpointing
#    - Initialize optimizer and scheduler
# 3. Reference model setup:
#    - Move to appropriate device
#    - Set to eval mode (no dropout, etc.)

# ═══════════════════════════════════════════════════════════════════
# STEP 7: TRAIN THE MODEL
# ═══════════════════════════════════════════════════════════════════

print("\nStarting training...")
print("="*70)

trainer.train()

# WHAT HAPPENS DURING TRAINING?
# For each batch:
# 1. Collate and pad data
# 2. Concatenate chosen and rejected responses
# 3. Forward pass through policy model
# 4. Forward pass through reference model (or use cached log probs)
# 5. Compute DPO loss
# 6. Backpropagate and update policy model
# 7. Log metrics (loss, rewards, accuracy)

# KEY METRICS TO MONITOR:
# - loss: Should decrease over time
# - reward_margin: chosen_reward - rejected_reward (should be positive and grow)
# - accuracy: Fraction where chosen_reward > rejected_reward (target: >0.6)
# - learning_rate: Should follow warmup then decay schedule

print("="*70)
print("Training complete!")

# ═══════════════════════════════════════════════════════════════════
# STEP 8: SAVE THE ALIGNED MODEL
# ═══════════════════════════════════════════════════════════════════

output_path = "./qwen2-dpo-aligned"
print(f"\nSaving aligned model to {output_path}")

trainer.save_model(output_path)
tokenizer.save_pretrained(output_path)

print(f"Model saved! You can now load it with:")
print(f"  model = AutoModelForCausalLM.from_pretrained('{output_path}')")

# ═══════════════════════════════════════════════════════════════════
# STEP 9: OPTIONAL - EVALUATE THE MODEL
# ═══════════════════════════════════════════════════════════════════

if config.eval_strategy != "no":
    print("\nRunning final evaluation...")
    metrics = trainer.evaluate()
    print("\nFinal Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

# INTERPRETING METRICS:
# - eval_loss: Lower is better (target: <0.5)
# - eval_rewards/chosen: Average reward for chosen responses
# - eval_rewards/rejected: Average reward for rejected responses
# - eval_rewards/margin: chosen - rejected (target: >0.5)
# - eval_rewards/accuracy: % where chosen > rejected (target: >0.7)

# ═══════════════════════════════════════════════════════════════════
# STEP 10: TEST THE ALIGNED MODEL
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("TESTING THE ALIGNED MODEL")
print("="*70)

# Load the aligned model
aligned_model = AutoModelForCausalLM.from_pretrained(
    output_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Test prompt
test_prompt = "Explain quantum computing in simple terms."

print(f"\nPrompt: {test_prompt}")
print("\nGenerating response...")

# Tokenize input
inputs = tokenizer(test_prompt, return_tensors="pt").to(aligned_model.device)

# Generate
outputs = aligned_model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    do_sample=True,
    top_p=0.9
)

# Decode and print
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nResponse: {response}")

print("\n" + "="*70)
print("DONE!")
print("="*70)

# ═══════════════════════════════════════════════════════════════════
# TROUBLESHOOTING GUIDE
# ═══════════════════════════════════════════════════════════════════
"""
COMMON ISSUES AND SOLUTIONS:

1. OUT OF MEMORY (OOM)
   Symptoms: CUDA OOM error
   Solutions:
   - Reduce per_device_train_batch_size (try 1)
   - Increase gradient_accumulation_steps
   - Enable gradient_checkpointing=True
   - Use precompute_ref_log_probs=True
   - Reduce max_length
   - Use PEFT/LoRA instead of full fine-tuning

2. LOSS NOT DECREASING
   Symptoms: Loss stays flat or increases
   Solutions:
   - Check if learning_rate is too low (try 1e-6)
   - Ensure beta is not too low (try 0.2)
   - Verify dataset quality (inspect samples)
   - Check if model was loaded correctly
   - Try different loss_type (e.g., "ipo")

3. ACCURACY NOT IMPROVING
   Symptoms: accuracy metric stays around 0.5
   Solutions:
   - Increase beta (try 0.3-0.5)
   - Train for more epochs
   - Check dataset for clear preferences
   - Verify prompt quality
   - Try a better base model (SFT first if not done)

4. MODEL OUTPUTS GIBBERISH
   Symptoms: Generated text is nonsensical
   Solutions:
   - Learning rate might be too high (try 1e-7)
   - Too many epochs (reduce to 1-2)
   - Beta too high (try 0.1)
   - Check if reference model is properly frozen
   - Verify tokenizer is correct for the model

5. TRAINING TOO SLOW
   Symptoms: Steps take >5 seconds each
   Solutions:
   - Enable flash_attention_2
   - Use bf16=True
   - Reduce max_length
   - Increase per_device_train_batch_size (if memory allows)
   - Use precompute_ref_log_probs=True
   - Upgrade to faster GPUs (A100, H100)

6. REWARD MARGIN DECREASING
   Symptoms: chosen_reward - rejected_reward gets smaller
   Solutions:
   - This might indicate overfitting
   - Reduce num_train_epochs
   - Add regularization (label_smoothing=0.05)
   - Increase learning_rate warmup
   - Use early stopping based on eval metrics
"""

# ═══════════════════════════════════════════════════════════════════
# HYPERPARAMETER TUNING GUIDE
# ═══════════════════════════════════════════════════════════════════
"""
PARAMETER SENSITIVITY (from most to least important):

1. learning_rate (CRITICAL)
   - Too high: Model degrades, outputs gibberish
   - Too low: No learning, loss doesn't decrease
   - Recommended range: [1e-7, 5e-6]
   - Start: 5e-7 for 7B, 1e-6 for <1B

2. beta (VERY IMPORTANT)
   - Too high: Overfitting to preferences, less diverse
   - Too low: Insufficient alignment
   - Recommended range: [0.05, 0.5]
   - Start: 0.1, increase if not learning

3. num_train_epochs (IMPORTANT)
   - Too many: Overfitting
   - Too few: Underfitting
   - Recommended: 1-3 epochs
   - Monitor eval metrics to decide

4. batch_size (effective) (MODERATE)
   - Larger: More stable gradients, faster convergence
   - Smaller: Less memory, more noise
   - Recommended: 16-64 effective batch size

5. max_length (MODERATE)
   - Longer: More context, better quality
   - Shorter: Less memory, faster training
   - Recommended: 512-1024 for general tasks

6. label_smoothing (MINOR)
   - Use only if data is noisy
   - Recommended: 0.0-0.1

7. loss_type (SITUATIONAL)
   - Default: "sigmoid" works well
   - Noisy data: Try "ipo" or "robust"
   - Margin-based: Try "hinge"
"""
