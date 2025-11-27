#!/usr/bin/env python3
"""
RLOO Training Script (REINFORCE Leave-One-Out)

Trains models using RL with leave-one-out baseline for variance reduction.

USAGE:
    python rloo_training_ANNOTATED.py \\
        --model Qwen/Qwen2-0.5B \\
        --dataset trl-lib/tldr \\
        --num_generations 4

REQUIREMENTS:
    pip install trl transformers datasets torch
"""

import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import RLOOTrainer, RLOOConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train model with RLOO")

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-0.5B",
        help="Base model to train"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="trl-lib/tldr",
        help="Dataset with 'prompt' column"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./rloo_output",
        help="Directory to save trained model"
    )

    parser.add_argument(
        "--num_generations",
        type=int,
        default=4,
        help="Number of completions per prompt (2-8 recommended)"
    )

    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=512,
        help="Maximum prompt length"
    )

    parser.add_argument(
        "--max_completion_length",
        type=int,
        default=256,
        help="Maximum completion length"
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=0.05,
        help="KL penalty coefficient (set 0 to disable reference model)"
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.2,
        help="PPO clip range"
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
        help="Learning rate"
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
        default=1,
        help="Prompts per device (not completions!)"
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Training {args.model} with RLOO")
    print(f"Dataset: {args.dataset}")
    print(f"Num generations: {args.num_generations}")

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

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ================================================================
    # 2. LOAD DATASET
    # ================================================================
    # Dataset must have "prompt" column
    print(f"\nLoading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split="train")

    if "prompt" not in dataset.column_names:
        raise ValueError(f"Dataset must have 'prompt' column. Found: {dataset.column_names}")

    print(f"✓ Dataset has 'prompt' column")
    print(f"Train samples: {len(dataset)}")

    # Inspect example
    print(f"\nExample prompt: {dataset[0]['prompt'][:100]}...")

    # ================================================================
    # 3. DEFINE REWARD FUNCTION
    # ================================================================
    # Custom reward function example
    def reward_function(prompts, completions, **kwargs):
        """
        Example reward function.

        Args:
            prompts: List of prompts
            completions: List of completions
            **kwargs: Additional dataset columns + trainer_state

        Returns:
            List of rewards (floats)
        """
        rewards = []
        for completion in completions:
            # Example: Reward based on length and unique words
            words = completion.split()
            length_score = min(len(words) / 50, 1.0)  # Normalize
            unique_ratio = len(set(words)) / max(len(words), 1)

            # Combined reward
            reward = length_score + unique_ratio
            rewards.append(reward)

        return rewards

    print("\nUsing custom reward function")
    print("  Rewards: length + word diversity")

    # Alternative: Use pretrained reward model
    # reward_funcs = "weqweasdas/RM-Mistral-7B"

    # ================================================================
    # 4. CONFIGURE TRAINING
    # ================================================================
    # Check batch size divisibility
    effective_batch = (
        args.per_device_train_batch_size *
        args.gradient_accumulation_steps
    )
    if effective_batch % args.num_generations != 0:
        raise ValueError(
            f"Effective batch size ({effective_batch}) must be divisible by "
            f"num_generations ({args.num_generations})"
        )

    print(f"\nEffective batch size: {effective_batch}")
    print(f"Prompts per update: {effective_batch // args.num_generations}")
    print(f"Completions per update: {effective_batch}")

    config = RLOOConfig(
        output_dir=args.output_dir,

        # Generation
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=1.0,

        # RL hyperparameters
        beta=args.beta,
        epsilon=args.epsilon,
        normalize_advantages=True,
        reward_clip_range=(-10, 10),

        # Training
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        # Logging
        logging_steps=10,
        log_completions=True,  # Log sample completions
        report_to=["tensorboard"],

        # Optimization
        gradient_checkpointing=True,
        bf16=True,

        # Checkpointing
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
    )

    # ================================================================
    # 5. INITIALIZE TRAINER
    # ================================================================
    print("\nInitializing RLOOTrainer...")

    trainer = RLOOTrainer(
        model=model,
        reward_funcs=reward_function,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Print model info
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(
        p.numel() for p in trainer.model.parameters() if p.requires_grad
    )
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    if args.beta > 0:
        print(f"\nReference model: Enabled (beta={args.beta})")
    else:
        print(f"\nReference model: Disabled (beta=0) - saves memory!")

    # ================================================================
    # 6. TRAIN
    # ================================================================
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    print(f"\nTraining flow:")
    print(f"  1. For each prompt, generate {args.num_generations} completions")
    print(f"  2. Compute rewards for all completions")
    print(f"  3. Compute leave-one-out baselines")
    print(f"  4. Update policy with clipped loss")

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
    # 8. TEST INFERENCE
    # ================================================================
    print("\n" + "="*70)
    print("TESTING INFERENCE")
    print("="*70)

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
        "Summarize: The quick brown fox jumps over the lazy dog.",
        "Explain: What is machine learning?",
        "Write: A haiku about coding.",
    ]

    print("\nGenerating with RLOO-trained model:\n")
    for i, prompt in enumerate(test_prompts, 1):
        print(f"[{i}] Prompt: {prompt}")
        output = pipe(
            prompt,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        response = output[0]["generated_text"][len(prompt):].strip()
        print(f"    Response: {response}\n")

    # ================================================================
    # 9. SUMMARY
    # ================================================================
    print("="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Method: RLOO (REINFORCE Leave-One-Out)")
    print(f"Num generations: {args.num_generations}")
    print(f"Beta: {args.beta}")
    print(f"Epsilon: {args.epsilon}")
    print(f"Training samples: {len(dataset)}")
    print(f"\nKey advantage:")
    print(f"  ✓ Simpler than PPO (no value function)")
    print(f"  ✓ Lower variance than REINFORCE")
    print(f"  ✓ Flexible reward functions")
    print(f"\nOutput: {args.output_dir}")
    print("\n✓ Training complete! Model optimized with RLOO.")


if __name__ == "__main__":
    main()
