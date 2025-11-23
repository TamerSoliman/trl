# GRPOTrainer: Group Relative Policy Optimization

**Source:** `trl/trainer/grpo_trainer.py` (2,028 lines)
**Paper:** [DeepSeekMath](https://arxiv.org/abs/2402.03300)
**Purpose:** Simplified RL without value functions - uses group-based advantages

---

## Overview

GRPO simplifies RL training by **comparing responses within a group** instead of using a value function.

**Key Innovation:**
```
PPO: advantage = reward - value_function(state)
GRPO: advantage = reward - mean(group_rewards)
```

**Advantages over PPO:**
- Only 2 models needed (policy + reference) vs PPO's 4
- No value function training
- Simpler implementation
- Competitive performance

**When to use:**
- Math/reasoning tasks (original use case)
- Multiple candidate responses per prompt
- Want simpler RL than PPO
- Have reward function (not preference data)

---

## Core Concept: Group-Based Advantages

### The Idea

Instead of training a value function V(s), GRPO:
1. Generates K completions per prompt
2. Computes reward for each
3. Uses group mean as baseline

```python
# Generate K completions per prompt
completions = [generate() for _ in range(K)]  # K=4 typical

# Score all
rewards = [reward_func(prompt, c) for c in completions]

# Compute advantages (group-relative)
mean_reward = mean(rewards)
advantages = [r - mean_reward for r in rewards]
```

**Code:** `grpo_trainer.py:1200-1250`

**Why this works:**
- Group mean is an unbiased estimate of V(s)
- Multiple samples reduce variance
- No need to train separate value model

### Comparison

| Method | Baseline | Models Needed | Variance |
|--------|----------|---------------|----------|
| **REINFORCE** | None | 1 | Very High |
| **GRPO** | Group mean | 2 | Medium |
| **PPO** | Value function | 4 | Low |

---

## Implementation Details

### Initialization

**Key arguments:**

```python
trainer = GRPOTrainer(
    model="meta-llama/Llama-3-8B-SFT",
    reward_funcs=reward_function,  # Can be model or custom function
    train_dataset=dataset,
    args=GRPOConfig(
        num_generations=4,  # K completions per prompt
        kl_coef=0.05,       # KL penalty
    )
)
```

**Reward functions:**
Can be:
1. **Model ID string:** Loads reward model
2. **PreTrainedModel:** Uses directly
3. **Custom function:** `def reward_func(completions, **kwargs) -> list[float]`
4. **List of above:** Multiple rewards summed

**Code:** `grpo_trainer.py:243-256`

### Training Loop

**Structure:**

```
for batch in dataloader:
    # 1. GENERATION PHASE
    prompts = batch["prompts"]
    for _ in range(num_generations):
        completions = generate(prompts)
        logprobs = compute_logprobs(completions)
        rewards = reward_func(prompts, completions)

    # 2. ADVANTAGE COMPUTATION
    for prompt_group in groups:
        mean_reward = mean(rewards_in_group)
        advantages = rewards - mean_reward

    # 3. POLICY OPTIMIZATION
    for epoch in range(num_epochs):
        policy_loss = -advantages * log_prob_ratio
        kl_loss = KL(policy || reference)
        loss = policy_loss + kl_coef * kl_loss
        optimizer.step()
```

**Code:** `grpo_trainer.py:900-1400`

### Generation

GRPO supports two generation modes:

**1. Standard (PyTorch):**
```python
with torch.no_grad():
    outputs = model.generate(
        prompts,
        max_new_tokens=response_length,
        do_sample=True,
        temperature=1.0,
    )
```

**2. vLLM (faster):**
```python
if args.use_vllm:
    llm = LLM(model=model_id)
    outputs = llm.generate(
        prompts,
        SamplingParams(max_tokens=response_length)
    )
```

**Code:** `grpo_trainer.py:600-800`

**vLLM benefits:**
- 2-5x faster generation
- Better batching
- PagedAttention (less memory)

### Loss Computation

**Policy loss (similar to PPO):**

```python
# Compute log prob ratio
new_logprobs = compute_logprobs(model, completions)
ratio = exp(new_logprobs - old_logprobs)

# GRPO uses unclipped loss (no clipping like PPO)
policy_loss = -mean(advantages * log(ratio))
```

**Code:** `grpo_trainer.py:1300-1350`

**KL penalty:**

```python
ref_logprobs = compute_logprobs(ref_model, completions)
kl = new_logprobs - ref_logprobs
kl_loss = kl_coef * mean(kl)
```

**Total loss:**
```python
loss = policy_loss + kl_loss
```

### Key Differences from PPO

| Aspect | PPO | GRPO |
|--------|-----|------|
| **Value function** | Trained separately | None (uses group mean) |
| **Clipping** | Clipped objective | No clipping |
| **Advantage** | GAE with value | Group-relative |
| **Generations** | 1 per prompt | K per prompt (K=4 typical) |
| **Models** | 4 (policy, ref, value, reward) | 2-3 (policy, ref, reward) |

---

## Configuration

### GRPOConfig

**Key parameters:**

```python
@dataclass
class GRPOConfig:
    # Generation
    num_generations: int = 4           # Completions per prompt
    response_length: int = 128         # Max tokens
    temperature: float = 1.0           # Sampling temperature
    use_vllm: bool = False            # Use vLLM for generation

    # Training
    kl_coef: float = 0.05             # KL penalty
    num_epochs_per_update: int = 1    # Optimization epochs
    learning_rate: float = 1e-6       # Lower than SFT

    # Optimization
    per_device_train_batch_size: int = 1  # Prompts per device
    gradient_accumulation_steps: int = 16  # Effective batch
```

**Typical values:**

| Task | num_generations | kl_coef | learning_rate |
|------|----------------|---------|---------------|
| **Math** | 8-16 | 0.01-0.05 | 5e-7 to 1e-6 |
| **Code** | 4-8 | 0.05-0.1 | 1e-6 to 5e-6 |
| **General** | 4 | 0.05 | 1e-6 |

---

## Memory Requirements

**Example: Llama-3-8B**

Without optimizations:
```
Policy model (trainable): 14 GB
Reference model: 14 GB
Reward model: 14 GB
Generations (4×batch): 10-15 GB
Total: ~50-55 GB
```

With optimizations:
```
Policy (LoRA): 14 GB
Reference (shared base): 0 GB (adapter switching)
Reward (4-bit): 4 GB
Generations: 10 GB
Total: ~28 GB (fits A100 40GB!)
```

---

## Production Example

```python
from trl import GRPOTrainer, GRPOConfig

# Define reward function
def math_reward(completions, answers, **kwargs):
    """Custom reward for math problems."""
    rewards = []
    for completion, correct_answer in zip(completions, answers):
        # Extract answer from completion
        predicted = extract_answer(completion)
        # Binary reward
        reward = 1.0 if predicted == correct_answer else 0.0
        rewards.append(reward)
    return rewards

# Configure
config = GRPOConfig(
    num_generations=8,  # Generate 8 solutions per problem
    kl_coef=0.02,
    learning_rate=5e-7,
    use_vllm=True,  # Fast generation
)

# Train
trainer = GRPOTrainer(
    model="meta-llama/Llama-3-8B-SFT",
    reward_funcs=math_reward,
    args=config,
    train_dataset=math_dataset,
)

trainer.train()
```

---

## Summary

**GRPO = Simplified RL**

**Pros:**
- Simpler than PPO (no value function)
- Fewer models (2-3 vs 4)
- Less memory
- Competitive performance

**Cons:**
- Higher variance than PPO
- Requires multiple generations (slower)
- Less sample efficient

**Best for:**
- Tasks with verifiable rewards (math, code)
- When you want RL without PPO complexity
- Limited compute (vs PPO)
