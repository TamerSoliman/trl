#!/usr/bin/env python3
"""GRPO Training - Simplified RL without value functions"""

from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig

# Custom reward function example
def math_reward(completions, answers, **kwargs):
    """Reward correctness for math problems."""
    return [1.0 if extract_answer(c) == a else 0.0 
            for c, a in zip(completions, answers)]

def extract_answer(text):
    """Extract final answer from completion."""
    # Implement your logic
    return text.split("####")[-1].strip() if "####" in text else text

# Load dataset
dataset = load_dataset("gsm8k", split="train")

# Configure
config = GRPOConfig(
    output_dir="./grpo_output",
    num_generations=4,  # 4 completions per prompt
    response_length=256,
    kl_coef=0.05,
    learning_rate=1e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    use_vllm=False,  # Set True if vLLM installed
)

# Initialize trainer
trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=math_reward,  # Or use reward model: "model_id"
    args=config,
    train_dataset=dataset,
)

# Train
trainer.train()
trainer.save_model("./grpo_output/final")
