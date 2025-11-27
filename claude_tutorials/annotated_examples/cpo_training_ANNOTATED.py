#!/usr/bin/env python3
"""
CPO Training Script (Contrastive Preference Optimization)

USAGE:
    python cpo_training_ANNOTATED.py \\
        --model Qwen/Qwen2-0.5B \\
        --dataset trl-lib/ultrafeedback_binarized \\
        --loss_type sigmoid

REQUIREMENTS:
    pip install trl transformers datasets torch
"""

import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl.experimental import CPOTrainer, CPOConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train model with CPO")

    parser.add_argument("--model", type=str, default="Qwen/Qwen2-0.5B")
    parser.add_argument("--dataset", type=str, default="trl-lib/ultrafeedback_binarized")
    parser.add_argument("--output_dir", type=str, default="./cpo_output")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--loss_type", type=str, default="sigmoid",
                       choices=["sigmoid", "hinge", "ipo", "simpo"])
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.0, help="AlphaPO parameter")
    parser.add_argument("--learning_rate", type=float, default=5e-7)

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Training {args.model} with CPO")
    print(f"Loss type: {args.loss_type}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_dataset(args.dataset, split="train")
    dataset = dataset.train_test_split(test_size=0.05, seed=42)

    # Configure
    config = CPOConfig(
        output_dir=args.output_dir,
        loss_type=args.loss_type,
        beta=args.beta,
        label_smoothing=args.label_smoothing,
        alpha=args.alpha,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        max_length=1024,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=10,
    )

    # Optional: LoRA
    peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                             task_type="CAUSAL_LM") if args.use_lora else None

    # Train
    trainer = CPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("\n" + "="*70)
    print(f"TRAINING WITH {args.loss_type.upper()} LOSS")
    print("="*70)
    trainer.train()
    trainer.save_model(args.output_dir)

    print(f"\n✓ Training complete! Model saved to {args.output_dir}")
    print(f"  Loss type used: {args.loss_type}")
    print(f"  Beta: {args.beta}")
    if args.label_smoothing > 0:
        print(f"  Label smoothing: {args.label_smoothing}")
    if args.alpha != 0:
        print(f"  AlphaPO alpha: {args.alpha}")


if __name__ == "__main__":
    main()
