#!/usr/bin/env python3
"""
KTO Training Script (Kahneman-Tversky Optimization)

Trains models using binary preference data (desirable/undesirable) without
requiring pairwise comparisons.

USAGE:
    python kto_training_ANNOTATED.py \\
        --model Qwen/Qwen2-0.5B \\
        --dataset trl-lib/ultrafeedback_binarized \\
        --beta 0.1

REQUIREMENTS:
    pip install trl transformers datasets peft torch
"""

import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import KTOTrainer, KTOConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train model with KTO")

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-0.5B",
        help="Base model to align"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="trl-lib/ultrafeedback_binarized",
        help="Dataset with binary preference labels (or pairwise - will auto-convert)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./kto_output",
        help="Directory to save trained model"
    )

    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use LoRA for parameter-efficient training"
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="KTO temperature parameter (higher = less deviation from reference)"
    )

    parser.add_argument(
        "--desirable_weight",
        type=float,
        default=1.0,
        help="Weight for desirable examples (adjust for class imbalance)"
    )

    parser.add_argument(
        "--undesirable_weight",
        type=float,
        default=1.0,
        help="Weight for undesirable examples (adjust for class imbalance)"
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum sequence length (prompt + completion)"
    )

    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=512,
        help="Maximum prompt length"
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-7,
        help="Learning rate (typically lower than SFT)"
    )

    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size per device (needs >=16 effective for stable KL)"
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )

    parser.add_argument(
        "--precompute_ref_log_probs",
        action="store_true",
        help="Precompute reference log probs to save memory"
    )

    parser.add_argument(
        "--loss_type",
        type=str,
        default="kto",
        choices=["kto", "apo_zero_unpaired"],
        help="Loss variant: 'kto' (standard) or 'apo_zero_unpaired'"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Training {args.model} with KTO")
    print(f"Dataset: {args.dataset}")
    print(f"Beta: {args.beta}")
    print(f"Loss type: {args.loss_type}")
    print(f"LoRA: {args.use_lora}")
    print(f"Precompute ref logprobs: {args.precompute_ref_log_probs}")

    # ================================================================
    # 1. LOAD MODEL AND TOKENIZER
    # ================================================================
    print("\nLoading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ================================================================
    # 2. LOAD DATASET
    # ================================================================
    # Dataset can be:
    # - Binary format: {"prompt": str, "completion": str, "label": bool}
    # - Pairwise format: {"prompt": str, "chosen": str, "rejected": str}
    #   (KTOTrainer auto-converts pairwise → binary)
    print(f"\nLoading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split="train")

    # Check format
    if "label" in dataset.column_names:
        print("✓ Dataset has binary labels (desirable/undesirable)")
        print(f"  Columns: {dataset.column_names}")
    elif "chosen" in dataset.column_names and "rejected" in dataset.column_names:
        print("✓ Dataset has pairwise format - will auto-convert to binary")
        print(f"  Columns: {dataset.column_names}")
    else:
        raise ValueError(
            "Dataset must have either (prompt, completion, label) or "
            "(prompt, chosen, rejected) columns"
        )

    # Split into train/eval
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")

    # Inspect example
    example = train_dataset[0]
    print(f"\nExample from dataset:")
    for key, value in example.items():
        if isinstance(value, str):
            print(f"  {key}: {value[:100]}...")
        else:
            print(f"  {key}: {value}")

    # Calculate class balance (if binary format)
    if "label" in train_dataset.column_names:
        num_desirable = sum(train_dataset["label"])
        num_undesirable = len(train_dataset) - num_desirable
        print(f"\nClass balance:")
        print(f"  Desirable: {num_desirable} ({100*num_desirable/len(train_dataset):.1f}%)")
        print(f"  Undesirable: {num_undesirable} ({100*num_undesirable/len(train_dataset):.1f}%)")

        # Suggest weights if imbalanced
        if abs(num_desirable - num_undesirable) > 0.2 * len(train_dataset):
            ratio = max(num_desirable, num_undesirable) / min(num_desirable, num_undesirable)
            print(f"\n⚠ Dataset is imbalanced (ratio {ratio:.2f}:1)")
            if num_desirable > num_undesirable:
                print(f"  Suggestion: --undesirable_weight {ratio:.2f}")
            else:
                print(f"  Suggestion: --desirable_weight {ratio:.2f}")

    # ================================================================
    # 3. CONFIGURE TRAINING
    # ================================================================
    config = KTOConfig(
        output_dir=args.output_dir,

        # KTO-specific parameters
        beta=args.beta,
        desirable_weight=args.desirable_weight,
        undesirable_weight=args.undesirable_weight,
        loss_type=args.loss_type,

        # Sequence lengths
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,

        # Memory optimization
        precompute_ref_log_probs=args.precompute_ref_log_probs,
        gradient_checkpointing=True,

        # Learning
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,

        # Batch sizes
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        # Evaluation
        eval_strategy="steps",
        eval_steps=500,
        generate_during_eval=False,  # Set True to see sample generations

        # Checkpointing
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        load_best_model_at_end=True,

        # Optimization
        bf16=True,  # Mixed precision

        # Logging
        logging_steps=10,
        report_to=["tensorboard"],
    )

    # Print effective batch size
    effective_batch = (
        args.per_device_train_batch_size *
        args.gradient_accumulation_steps
    )
    print(f"\nEffective batch size: {effective_batch}")
    if effective_batch < 16:
        print("⚠ Warning: Effective batch size < 16 may cause unstable KL estimation")
        print("  Consider increasing batch size or gradient_accumulation_steps")

    # ================================================================
    # 4. CONFIGURE LORA (OPTIONAL)
    # ================================================================
    peft_config = None
    if args.use_lora:
        print("\nUsing LoRA for parameter-efficient training")
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )

    # ================================================================
    # 5. INITIALIZE TRAINER
    # ================================================================
    print("\nInitializing KTOTrainer...")
    trainer = KTOTrainer(
        model=model,
        ref_model=None,  # Will be auto-created from model
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Print model info
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(
        p.numel() for p in trainer.model.parameters() if p.requires_grad
    )
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # ================================================================
    # 6. TRAIN
    # ================================================================
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    print("\nNote: First epoch may be slower if precomputing reference log probs")

    trainer.train()

    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)

    # ================================================================
    # 7. SAVE FINAL MODEL
    # ================================================================
    print(f"\nSaving final model to {args.output_dir}")
    trainer.save_model(args.output_dir)

    # ================================================================
    # 8. EVALUATE
    # ================================================================
    print("\nFinal evaluation:")
    metrics = trainer.evaluate()

    print(f"  Loss: {metrics.get('eval_loss', 'N/A'):.4f}")
    print(f"  Rewards/chosen: {metrics.get('eval_rewards/chosen', 'N/A')}")
    print(f"  Rewards/rejected: {metrics.get('eval_rewards/rejected', 'N/A')}")
    print(f"  Rewards/margins: {metrics.get('eval_rewards/margins', 'N/A')}")
    print(f"  KL: {metrics.get('eval_kl', 'N/A')}")

    # ================================================================
    # 9. TEST INFERENCE
    # ================================================================
    print("\n" + "="*70)
    print("TESTING INFERENCE")
    print("="*70)

    # Load final model
    from transformers import pipeline

    pipe = pipeline(
        "text-generation",
        model=args.output_dir,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype="auto",
    )

    # Test prompts
    test_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about coding.",
    ]

    print("\nGenerating responses with aligned model:\n")
    for prompt in test_prompts:
        print(f"Prompt: {prompt}")
        output = pipe(
            prompt,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        response = output[0]["generated_text"][len(prompt):].strip()
        print(f"Response: {response}\n")

    # ================================================================
    # 10. SUMMARY
    # ================================================================
    print("="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Method: KTO ({args.loss_type})")
    print(f"Beta: {args.beta}")
    print(f"Weights: desirable={args.desirable_weight}, undesirable={args.undesirable_weight}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Final loss: {metrics.get('eval_loss', 'N/A'):.4f}")
    print(f"Reward margin: {metrics.get('eval_rewards/margins', 'N/A')}")
    print(f"Output: {args.output_dir}")
    print("\n✓ Training complete! Model aligned with KTO.")


if __name__ == "__main__":
    main()
