# RLOOTrainer: REINFORCE Leave-One-Out

**Source:** `trl/trainer/rloo_trainer.py` (1600+ lines)
**Purpose:** Reinforcement learning with leave-one-out baseline for variance reduction

---

## Overview

RLOOTrainer implements **REINFORCE Leave-One-Out**, a policy gradient RL method that uses a clever leave-one-out baseline to reduce variance without requiring a value function. It's simpler than PPO but more sample-efficient than vanilla REINFORCE.

### The Key Innovation

**Problem with vanilla REINFORCE:** High variance in gradient estimates

**Traditional solution:** Learn a value function V(s) as baseline (adds complexity)

**RLOO solution:** Use leave-one-out mean of other samples as baseline (no extra learning!)

---

## Core Concept: Leave-One-Out Baseline

### Mathematical Foundation

For each prompt x, generate G completions: {y₁, y₂, ..., yG}

**For completion yᵢ:**
```
Baseline(yᵢ) = mean of OTHER completions
            = (sum of all rewards - reward(yᵢ)) / (G - 1)
            = (Σⱼ≠ᵢ r(xⱼ, yⱼ)) / (G - 1)

Advantage(yᵢ) = reward(yᵢ) - Baseline(yᵢ)
```

**Code:**
```python
grouped_rewards = rewards.view(-1, num_generations)  # Shape: (num_prompts, G)
grouped_sum = grouped_rewards.sum(dim=1, keepdim=True)  # Sum all rewards per prompt

# Leave-one-out baseline
baselines = (grouped_sum - grouped_rewards) / (num_generations - 1)
baselines = baselines.view(-1)  # Flatten

# Advantages
advantages = rewards - baselines
```

**Code location:** `rloo_trainer.py:1403-1407`

**Intuition:** Each completion is compared to the average of its siblings, not to an absolute baseline.

---

## Why Leave-One-Out?

### Variance Reduction

**Vanilla REINFORCE:**
```
∇J = E[r(y) * ∇log π(y|x)]
Variance = Var[r(y)]  [high!]
```

**REINFORCE with baseline:**
```
∇J = E[(r(y) - b) * ∇log π(y|x)]
Variance = Var[r(y) - b]  [lower if b ≈ E[r]]
```

**RLOO baseline:**
```
b(yᵢ) = (1/(G-1)) * Σⱼ≠ᵢ r(yⱼ)
```

**Why it works:**
- Unbiased: E[b(yᵢ)] ≈ E[r(y)] when G is large
- No extra learning needed (unlike V-function)
- Low overhead (just averaging)

### Comparison to PPO

| Method | Baseline | Variance | Complexity |
|--------|----------|----------|------------|
| **Vanilla REINFORCE** | None | Very High | Low |
| **PPO** | Value function V(s) | Low | High (two networks) |
| **RLOO** | Leave-one-out mean | Medium-Low | Low (single network) |

**Trade-off:** RLOO is simpler than PPO but requires generating multiple completions (G ≥ 2).

---

## RLOO Loss Function

### PPO-Style Clipped Loss

```
ratio = π_θ(y|x) / π_old(y|x) = exp(log_π_θ - log_π_old)

L_RLOO = -min(
    ratio * advantage,
    clip(ratio, 1-ε, 1+ε) * advantage
)
```

**Code:**
```python
log_ratio = logps - old_logps
ratio = torch.exp(log_ratio)

coef_clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)

loss1 = ratio * advantages
loss2 = coef_clipped * advantages

loss = -torch.min(loss1, loss2).mean()
```

**Code location:** `rloo_trainer.py:1503-1508`

**Why clipping?** Prevents large policy updates that could destabilize training.

---

## Reward Functions

### Three Types Supported

**1. Pretrained reward model (string):**
```python
trainer = RLOOTrainer(
    model="Qwen/Qwen2-0.5B",
    reward_funcs="weqweasdas/RM-Mistral-7B",  # Load from HF
    ...
)
```

**2. Loaded reward model (PreTrainedModel):**
```python
reward_model = AutoModelForSequenceClassification.from_pretrained("...")

trainer = RLOOTrainer(
    model="Qwen/Qwen2-0.5B",
    reward_funcs=reward_model,
    ...
)
```

**3. Custom reward function (callable):**
```python
def custom_reward(prompts, completions, **kwargs):
    # Return list of rewards
    return [len(c.split()) for c in completions]  # Reward length

trainer = RLOOTrainer(
    model="Qwen/Qwen2-0.5B",
    reward_funcs=custom_reward,
    ...
)
```

**Multiple reward functions:**
```python
trainer = RLOOTrainer(
    reward_funcs=[reward_model_1, custom_reward, "path/to/rm"],
    reward_weights=[0.5, 0.3, 0.2],  # Weight each function
    ...
)
```

**Code location:** `rloo_trainer.py:940-996`

---

## Training Flow

### Step 1: Generate Completions

For each prompt, generate G completions:

```python
for prompt in batch:
    completions = [
        model.generate(prompt) for _ in range(num_generations)
    ]
```

**Typical G values:** 2-8 (more = lower variance, but slower)

**Code location:** `rloo_trainer.py:998-1200` (generation with vLLM or transformers)

### Step 2: Compute Rewards

```python
rewards = []
for prompt, completion in zip(prompts, completions):
    r = reward_func(prompt, completion)
    rewards.append(r)

# Optional: Add KL penalty
if beta > 0:
    kl = KL[π_θ || π_ref]
    rewards = rewards - beta * kl
```

**Code location:** `rloo_trainer.py:1381-1396`

### Step 3: Compute Advantages

```python
# Reshape to (num_prompts, num_generations)
grouped_rewards = rewards.view(-1, G)

# Leave-one-out baseline
grouped_sum = grouped_rewards.sum(dim=1, keepdim=True)
baselines = (grouped_sum - grouped_rewards) / (G - 1)

# Advantages
advantages = rewards - baselines.view(-1)

# Optional: Normalize
if normalize_advantages:
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-4)
```

**Code location:** `rloo_trainer.py:1398-1411`

### Step 4: Update Policy

```python
# Get new log probs
new_logps = model(prompt + completion)

# Compute ratio
ratio = exp(new_logps - old_logps)

# Clipped loss
loss = -min(ratio * advantages, clip(ratio) * advantages).mean()

# Backprop
loss.backward()
optimizer.step()
```

**Code location:** `rloo_trainer.py:1469-1529`

---

## Configuration

### RLOOConfig

```python
from trl import RLOOTrainer, RLOOConfig

config = RLOOConfig(
    # Generation
    num_generations=4,              # G in formulas (2-8 typical)
    max_prompt_length=512,
    max_completion_length=256,
    temperature=1.0,                # Sampling temperature

    # RL hyperparameters
    beta=0.05,                      # KL penalty coefficient
    epsilon=0.2,                    # PPO clip range
    normalize_advantages=True,      # Normalize advantages?

    # Training
    learning_rate=1e-6,             # Lower than SFT
    per_device_train_batch_size=1,  # Per-device prompts (not completions!)
    gradient_accumulation_steps=8,
    num_train_epochs=1,

    # Reward
    reward_clip_range=(-10, 10),    # Clip extreme rewards

    # vLLM (optional, for fast generation)
    use_vllm=False,                 # Use vLLM for generation?

    output_dir="./rloo_output",
)
```

**Code location:** `trl/trainer/rloo_config.py`

**Important:** Effective batch size must be divisible by `num_generations`.

---

## Production Example

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import RLOOTrainer, RLOOConfig

# Load model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")

# Load dataset (just prompts)
dataset = load_dataset("trl-lib/tldr", split="train")

# Define reward function
def reward_func(prompts, completions, **kwargs):
    # Example: Reward based on length and diversity
    rewards = []
    for c in completions:
        length_score = min(len(c.split()) / 50, 1.0)  # Normalize to [0,1]
        unique_words = len(set(c.split()))
        diversity_score = unique_words / max(len(c.split()), 1)
        rewards.append(length_score + diversity_score)
    return rewards

# Configure
config = RLOOConfig(
    output_dir="./rloo_model",
    num_generations=4,
    max_prompt_length=512,
    max_completion_length=256,
    learning_rate=1e-6,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    beta=0.05,
    epsilon=0.2,
    normalize_advantages=True,
    num_train_epochs=1,
    logging_steps=10,
)

# Train
trainer = RLOOTrainer(
    model=model,
    reward_funcs=reward_func,
    args=config,
    train_dataset=dataset,
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model("./rloo_final")
```

---

## Metrics

```python
{
    "reward": 2.5,                    # Mean reward per prompt
    "reward_std": 0.8,                # Std of rewards (within groups)
    "advantages": 0.1,                # Mean advantage (should be ~0)
    "clip_ratio/region_mean": 0.05,   # Fraction clipped
    "kl": 0.02,                       # KL divergence (if beta > 0)
    "entropy": 3.5,                   # Policy entropy
}
```

**Healthy training:**
- `reward` increasing
- `clip_ratio` < 0.2 (not clipping too much)
- `kl` low and stable
- `entropy` not collapsing

---

## Common Issues

### Issue 1: High variance

**Symptoms:** Loss/rewards oscillating wildly

**Solutions:**
- Increase `num_generations` (4 → 8)
- Enable `normalize_advantages=True`
- Use `reward_clip_range` to clip outliers
- Lower learning rate

### Issue 2: Batch size errors

**Problem:** "Batch size must be divisible by num_generations"

**Solution:** Ensure:
```
per_device_batch_size * num_gpus * gradient_accumulation_steps % num_generations == 0
```

### Issue 3: Slow training

**Problem:** Training is very slow

**Solutions:**
- Use vLLM for generation (`use_vllm=True`)
- Reduce `num_generations`
- Reduce `max_completion_length`
- Use larger batch sizes

---

## Comparison: RLOO vs PPO vs DPO

| Feature | RLOO | PPO | DPO |
|---------|------|-----|-----|
| **Baseline** | Leave-one-out | Value function V | Implicit (reference model) |
| **Requires** | Reward function | Reward function | Preference data |
| **Complexity** | Low | High | Medium |
| **Sample efficiency** | Medium | High | High |
| **Variance** | Medium-Low | Low | N/A |
| **Training speed** | Slow (multiple gens) | Medium | Fast |

**When to use RLOO:**
- Have reward function (not preference pairs)
- Want simpler than PPO
- Don't need maximum sample efficiency

**When NOT to use:**
- Have pairwise preferences (use DPO)
- Need maximum sample efficiency (use PPO)
- Can't afford multiple generations per prompt

---

## Memory Requirements

**7B model:**
- **Model:** ~14 GB
- **Reference model (if beta > 0):** ~14 GB
- **Activations (G completions):** ~G × 2 GB
- **Total:** 14-42 GB depending on G and beta

**Tips:**
- Set `beta=0` to skip reference model (~14 GB savings)
- Use smaller `num_generations`
- Use LoRA for both policy and reward models

---

## Summary

**RLOOTrainer = RL + Leave-One-Out Baseline**

**Key formulas:**
```
Baseline(yᵢ) = (Σⱼ≠ᵢ r(yⱼ)) / (G - 1)
Advantage(yᵢ) = r(yᵢ) - Baseline(yᵢ)
L = -min(ratio * A, clip(ratio) * A)
```

**Advantages:**
- Simpler than PPO (no value function)
- Lower variance than vanilla REINFORCE
- Flexible reward functions

**Trade-offs:**
- Requires multiple generations (slower)
- Less sample-efficient than PPO
- Still needs reward function (can't use preferences directly)

**Use cases:**
- RL from reward functions
- Middle ground between REINFORCE and PPO
- Research/prototyping RL methods

**Output:**
A policy optimized via RL using leave-one-out variance reduction.
