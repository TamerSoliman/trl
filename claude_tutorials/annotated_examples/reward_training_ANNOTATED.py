#!/usr/bin/env python3
"""
Reward Model Training Script

Trains a reward model from human preference data for use in RLHF.
The reward model learns to predict which response humans prefer.

USAGE:
    python reward_training_ANNOTATED.py \\
        --model Qwen/Qwen2-0.5B \\
        --dataset trl-lib/ultrafeedback_binarized

REQUIREMENTS:
    pip install trl transformers datasets peft torch
"""

import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from peft import LoraConfig
from trl import RewardTrainer, RewardConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train reward model")

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-0.5B",
        help="Base model for reward model"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="trl-lib/ultrafeedback_binarized",
        help="Preference dataset with chosen/rejected pairs"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./reward_model",
        help="Directory to save trained model"
    )

    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use LoRA for parameter-efficient training"
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate (lower than SFT typical)"
    )

    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of training epochs (usually 1 is enough)"
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size per device (preference pairs)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Training reward model from {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"LoRA: {args.use_lora}")

    # ================================================================
    # 1. LOAD DATASET
    # ================================================================
    # Dataset must have "chosen" and "rejected" columns
    print("\nLoading preference dataset...")
    dataset = load_dataset(args.dataset, split="train")

    # Optional: Create small eval set
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")

    # Inspect first example
    example = train_dataset[0]
    print(f"\nExample chosen: {example['chosen'][:100]}...")
    print(f"Example rejected: {example['rejected'][:100]}...")

    # ================================================================
    # 2. CONFIGURE TRAINING
    # ================================================================
    config = RewardConfig(
        output_dir=args.output_dir,

        # Learning
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,

        # Batch sizes
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=4,  # Effective batch = 4 * batch_size

        # Evaluation
        eval_strategy="steps",
        eval_steps=500,

        # Checkpointing
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        load_best_model_at_end=True,

        # Optimization
        gradient_checkpointing=True,  # Save memory
        bf16=True,                     # Mixed precision

        # Data processing
        max_length=args.max_length,
        remove_unused_columns=False,   # Keep all dataset columns

        # Logging
        logging_steps=10,
        report_to=["tensorboard"],
    )

    # ================================================================
    # 3. CONFIGURE LORA (OPTIONAL)
    # ================================================================
    peft_config = None
    if args.use_lora:
        print("\nUsing LoRA for parameter-efficient training")
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            task_type="SEQ_CLS",  # Sequence classification
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

    # ================================================================
    # 4. DEFINE METRICS
    # ================================================================
    def compute_metrics(eval_pred):
        """
        Compute accuracy: fraction of pairs where chosen > rejected.
        """
        predictions, _ = eval_pred
        # predictions: [batch_size * 2, 1] (chosen + rejected concatenated)

        # Split into chosen and rejected halves
        batch_size = predictions.shape[0] // 2
        chosen_rewards = predictions[:batch_size]
        rejected_rewards = predictions[batch_size:]

        # Accuracy: chosen scores > rejected scores
        accuracy = (chosen_rewards > rejected_rewards).astype(float).mean()

        # Compute average margin
        margin = (chosen_rewards - rejected_rewards).mean()

        return {
            "accuracy": accuracy,
            "margin": margin,
        }

    # ================================================================
    # 5. INITIALIZE TRAINER
    # ================================================================
    print("\nInitializing RewardTrainer...")
    trainer = RewardTrainer(
        model=args.model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        compute_metrics=compute_metrics,
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
    # 8. EVALUATE ON TEST SET
    # ================================================================
    print("\nFinal evaluation:")
    metrics = trainer.evaluate()
    print(f"  Accuracy: {metrics['eval_accuracy']:.4f}")
    print(f"  Margin: {metrics['eval_margin']:.4f}")

    # ================================================================
    # 9. TEST INFERENCE
    # ================================================================
    print("\n" + "="*70)
    print("TESTING INFERENCE")
    print("="*70)

    # Load trained model for inference
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
    model.eval()

    # Test on example
    test_prompt = "What is the capital of France?"
    good_response = "The capital of France is Paris."
    bad_response = "I don't know."

    # Tokenize
    good_inputs = tokenizer(
        test_prompt + good_response,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
    )
    bad_inputs = tokenizer(
        test_prompt + bad_response,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
    )

    # Get rewards
    import torch
    with torch.no_grad():
        good_reward = model(**good_inputs).logits[0, 0].item()
        bad_reward = model(**bad_inputs).logits[0, 0].item()

    print(f"\nTest prompt: {test_prompt}")
    print(f"Good response reward: {good_reward:.4f}")
    print(f"Bad response reward: {bad_reward:.4f}")
    print(f"Preference: {'✓ Correct' if good_reward > bad_reward else '✗ Wrong'}")

    print("\n✓ Training complete! Reward model ready for RLHF.")


if __name__ == "__main__":
    main()
