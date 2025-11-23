#!/usr/bin/env python3
"""
PPO Training Script - Production RLHF Implementation

PPO (Proximal Policy Optimization) is the gold standard for RLHF training.
Requires 4 models: policy, reference, value, reward.

USAGE:
    python ppo_training_ANNOTATED.py \\
        --policy_model meta-llama/Llama-3-8B-SFT \\
        --reward_model EleutherAI/pythia-1.4b-reward \\
        --dataset_name trl-lib/ultrafeedback-prompt

REQUIREMENTS:
    pip install trl transformers datasets peft accelerate torch

HARDWARE:
    - Full (4x7B models): 4x A100 80GB
    - With optimizations: 1x A100 40GB (LoRA + quantization)
"""

import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl.experimental.ppo import PPOConfig, PPOTrainer


def parse_args():
    parser = argparse.ArgumentParser()

    # Models
    parser.add_argument("--policy_model", default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--reward_model", default="Qwen/Qwen2-0.5B-Instruct")  # Placeholder
    parser.add_argument("--dataset_name", default="trl-lib/ultrafeedback-prompt")

    # Training
    parser.add_argument("--num_ppo_epochs", type=int, default=4)
    parser.add_argument("--kl_coef", type=float, default=0.05)
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)

    # Efficiency
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)

    return parser.parse_args()


def main():
    args = parse_args()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.policy_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load models
    print("Loading policy model...")
    policy = AutoModelForCausalLM.from_pretrained(
        args.policy_model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print("Loading reward model...")
    reward_model = AutoModelForCausalLM.from_pretrained(
        args.reward_model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print("Creating value model (copy of policy with value head)...")
    value_model = AutoModelForCausalLM.from_pretrained(
        args.policy_model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Configure PEFT
    peft_config = None
    if args.use_lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM"
        )

    # Load dataset
    dataset = load_dataset(args.dataset_name, split="train[:1%]")

    # Configure PPO
    config = PPOConfig(
        num_ppo_epochs=args.num_ppo_epochs,
        kl_coef=args.kl_coef,
        cliprange=args.cliprange,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        response_length=128,
        output_dir="./ppo_output",
    )

    # Initialize trainer
    trainer = PPOTrainer(
        args=config,
        processing_class=tokenizer,
        model=policy,
        ref_model=None,  # Will use PEFT adapter switching
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    # Train
    print("Starting PPO training...")
    trainer.train()

    # Save
    trainer.save_model("./ppo_output/final")
    print("Training complete!")


if __name__ == "__main__":
    main()
