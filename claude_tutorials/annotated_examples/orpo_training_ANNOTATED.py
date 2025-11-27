#!/usr/bin/env python3
"""
ORPO Training Script (Odds Ratio Preference Optimization)

Trains models using preference data WITHOUT a reference model - combines
SFT and alignment in a single stage.

USAGE:
    python orpo_training_ANNOTATED.py \\
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
from trl import ORPOTrainer, ORPOConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train model with ORPO")

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-0.5B",
        help="Base model to train (can be base or SFT model)"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="trl-lib/ultrafeedback_binarized",
        help="Pairwise preference dataset (prompt, chosen, rejected)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./orpo_output",
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
        help="Lambda parameter - weight of odds ratio loss (0.05-0.5)"
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
        default=8e-7,
        help="Learning rate"
    )

    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs (ORPO can train longer than DPO)"
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size per device"
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Training {args.model} with ORPO")
    print(f"Dataset: {args.dataset}")
    print(f"Beta (λ): {args.beta}")
    print(f"LoRA: {args.use_lora}")

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
    # Dataset must have: prompt, chosen, rejected
    print(f"\nLoading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split="train")

    # Verify format
    required_columns = ["prompt", "chosen", "rejected"]
    for col in required_columns:
        if col not in dataset.column_names:
            raise ValueError(f"Dataset must have '{col}' column. Found: {dataset.column_names}")

    print(f"✓ Dataset has required columns: {required_columns}")

    # Split into train/eval
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")

    # Inspect example
    example = train_dataset[0]
    print(f"\nExample from dataset:")
    print(f"  Prompt: {str(example['prompt'])[:100]}...")
    print(f"  Chosen: {str(example['chosen'])[:100]}...")
    print(f"  Rejected: {str(example['rejected'])[:100]}...")

    # ================================================================
    # 3. CONFIGURE TRAINING
    # ================================================================
    config = ORPOConfig(
        output_dir=args.output_dir,

        # ORPO-specific
        beta=args.beta,  # Lambda in the paper

        # Sequence lengths
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,

        # Learning
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,

        # Batch sizes
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        # Memory optimization
        gradient_checkpointing=True,
        bf16=True,

        # Evaluation
        eval_strategy="steps",
        eval_steps=500,
        generate_during_eval=False,  # Set True to see sample generations

        # Checkpointing
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_rewards/margins",  # Maximize preference gap

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
    print("\nInitializing ORPOTrainer...")
    print("Note: ORPO does NOT use a reference model - single model training!")

    trainer = ORPOTrainer(
        model=model,
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
    print(f"\nMemory advantage: ~50% less than DPO/KTO (no reference model)")

    # ================================================================
    # 6. TRAIN
    # ================================================================
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    print("\nTraining dynamics to watch:")
    print("  1. NLL loss should decrease (SFT learning)")
    print("  2. Margins should increase (preference learning)")
    print("  3. Accuracies should climb above 0.6")

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

    # Print key metrics
    print(f"\n{'='*50}")
    print("FINAL METRICS")
    print(f"{'='*50}")
    print(f"  Loss: {metrics.get('eval_loss', 'N/A'):.4f}")
    print(f"  NLL Loss: {metrics.get('eval_nll_loss', 'N/A'):.4f}")
    print(f"  Log Odds Ratio: {metrics.get('eval_log_odds_ratio', 'N/A'):.4f}")
    print(f"  Rewards/chosen: {metrics.get('eval_rewards/chosen', 'N/A'):.2f}")
    print(f"  Rewards/rejected: {metrics.get('eval_rewards/rejected', 'N/A'):.2f}")
    print(f"  Rewards/margins: {metrics.get('eval_rewards/margins', 'N/A'):.2f}")
    print(f"  Rewards/accuracies: {metrics.get('eval_rewards/accuracies', 'N/A'):.2%}")

    # Interpret results
    print(f"\n{'='*50}")
    print("INTERPRETATION")
    print(f"{'='*50}")

    nll = metrics.get('eval_nll_loss', float('inf'))
    if nll < 1.5:
        print("✓ NLL Loss: Excellent generation quality")
    elif nll < 2.5:
        print("✓ NLL Loss: Good generation quality")
    else:
        print("⚠ NLL Loss: May need more SFT training")

    margin = metrics.get('eval_rewards/margins', 0)
    if margin > 30:
        print("✓ Margins: Excellent preference separation")
    elif margin > 15:
        print("✓ Margins: Good preference separation")
    else:
        print("⚠ Margins: Weak preference separation - consider increasing beta")

    accuracy = metrics.get('eval_rewards/accuracies', 0)
    if accuracy > 0.75:
        print("✓ Accuracies: Strong preference learning")
    elif accuracy > 0.6:
        print("✓ Accuracies: Moderate preference learning")
    else:
        print("⚠ Accuracies: Weak preference learning")

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
        "Explain the concept of machine learning in simple terms.",
        "What are the benefits of regular exercise?",
        "Write a professional email declining a job offer politely.",
    ]

    print("\nGenerating responses with ORPO-aligned model:\n")
    for i, prompt in enumerate(test_prompts, 1):
        print(f"[{i}] Prompt: {prompt}")
        output = pipe(
            prompt,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        response = output[0]["generated_text"][len(prompt):].strip()
        print(f"    Response: {response}\n")

    # ================================================================
    # 10. COMPARISON WITH BASE MODEL
    # ================================================================
    print("="*70)
    print("BASE VS. ORPO COMPARISON")
    print("="*70)

    # Load base model for comparison
    print("\nLoading base model for comparison...")
    base_pipe = pipeline(
        "text-generation",
        model=args.model,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype="auto",
    )

    test_prompt = test_prompts[0]
    print(f"\nPrompt: {test_prompt}\n")

    print("Base model response:")
    base_output = base_pipe(
        test_prompt,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
    )
    base_response = base_output[0]["generated_text"][len(test_prompt):].strip()
    print(f"  {base_response}\n")

    print("ORPO model response:")
    orpo_output = pipe(
        test_prompt,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
    )
    orpo_response = orpo_output[0]["generated_text"][len(test_prompt):].strip()
    print(f"  {orpo_response}\n")

    # ================================================================
    # 11. SUMMARY
    # ================================================================
    print("="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Method: ORPO (Odds Ratio Preference Optimization)")
    print(f"Beta (λ): {args.beta}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Epochs: {args.num_train_epochs}")
    print(f"\nFinal metrics:")
    print(f"  Loss: {metrics.get('eval_loss', 'N/A'):.4f}")
    print(f"  NLL: {metrics.get('eval_nll_loss', 'N/A'):.4f}")
    print(f"  Margins: {metrics.get('eval_rewards/margins', 'N/A'):.2f}")
    print(f"  Accuracy: {metrics.get('eval_rewards/accuracies', 'N/A'):.2%}")
    print(f"\nKey advantage:")
    print(f"  ✓ No reference model needed")
    print(f"  ✓ ~50% memory savings vs DPO/KTO")
    print(f"  ✓ Single-stage SFT + alignment")
    print(f"\nOutput: {args.output_dir}")
    print("\n✓ Training complete! Model aligned with ORPO.")


if __name__ == "__main__":
    main()
