# PPO Concepts Reference: RL Theory to Practice

**Purpose:** Mathematical foundations for Proximal Policy Optimization

---

## The Core PPO Algorithm

### Objective Function

PPO maximizes:

```
L^CLIP(θ) = E_t[min(r_t(θ) Â_t, clip(r_t(θ), 1-ε, 1+ε) Â_t)]

where:
  r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)  (probability ratio)
  Â_t = GAE advantage estimate
  ε = clip range (typically 0.2)
```

**Code mapping:** `ppo_trainer.py:656-661`

**Why clipping works:**

```
If Â > 0 (good action):
  Unclipped: r * Â (encourage increasing π)
  But clip at 1+ε to prevent too large increase

If Â < 0 (bad action):
  Unclipped: r * Â (encourage decreasing π)
  But clip at 1-ε to prevent too large decrease
```

**Intuition:** Don't trust your gradient too much. Make small, conservative updates.

---

## Generalized Advantage Estimation (GAE)

### Formula

```
Â_t^GAE(γ,λ) = Σ_{l=0}^∞ (γλ)^l δ_{t+l}^V

where:
  δ_t^V = r_t + γV(s_{t+1}) - V(s_t)  (TD error)
  γ = discount factor
  λ = GAE parameter
```

**Code:** `ppo_trainer.py:602-613`

```python
lastgaelam = 0
for t in reversed(range(T)):
    nextvalu

es = V[t+1] if t < T-1 else 0
    delta = rewards[t] + gamma * nextvalues - V[t]
    lastgaelam = delta + gamma * lam * lastgaelam
    advantages[t] = lastgaelam
```

### Bias-Variance Tradeoff

```
λ = 0:  Â_t = δ_t = r_t + γV(s_{t+1}) - V(s_t)
        (1-step TD, high bias, low variance)

λ = 1:  Â_t = Σ_{l=0}^∞ γ^l r_{t+l} - V(s_t)
        (Monte Carlo, low bias, high variance)

λ = 0.95: Sweet spot (typically)
```

**Why GAE?** Balances:
- **Bias:** How wrong is the estimate on average
- **Variance:** How much does the estimate fluctuate

**Empirical finding:** λ=0.95 works well across many domains.

---

## The Four Models Explained

### 1. Policy π_θ (Actor)

**Role:** Generates actions (responses).

**Training:** Updated via PPO loss.

**Objective:** Maximize expected reward while staying close to reference.

```
max E[r(s,a) - β·KL(π_θ || π_ref)]
```

### 2. Value Function V_φ (Critic)

**Role:** Predicts expected future reward.

**Training:** Regression to actual returns.

**Loss:**
```
L_V = E[(V_φ(s) - R_target)^2]

where R_target = Â + V_old(s) (the return)
```

**Why needed?** Reduces variance in advantage estimates (actor-critic architecture).

### 3. Reference Policy π_ref

**Role:** Prevents policy from deviating too far from initialization.

**Training:** NOT trained (frozen).

**Usage:**
```python
kl = log π_θ(a|s) - log π_ref(a|s)
penalty = -kl_coef * kl
```

**Without reference:** Policy could exploit reward model by generating adversarial examples that score high but are nonsense.

### 4. Reward Model r_ψ

**Role:** Provides learning signal (scores responses).

**Training:** Trained separately on preference data before PPO.

**Preference training:**
```python
# Data: (prompt, chosen_response, rejected_response)
# Loss:
loss = -log σ(r(prompt, chosen) - r(prompt, rejected))
```

---

## KL Divergence and Regularization

### KL Penalty

```
KL(π_θ || π_ref) = E_{a~π_θ}[log π_θ(a|s) - log π_ref(a|s)]
```

**In PPO:**
```python
logr = ref_logprobs - logprobs  # log(π_ref/π_θ)
kl = -logr  # k1 estimator
reward = base_reward - kl_coef * kl
```

**Effect of kl_coef:**
- **High (0.1):** Conservative, stays close to reference
- **Low (0.01):** Aggressive, explores more
- **Typical:** 0.05

### KL Estimators

**k1 (default):**
```
KL ≈ log(π_θ/π_ref) = -log(π_ref/π_θ)
```
Straightforward, unbiased.

**k3:**
```
KL ≈ (π_ref/π_θ) - 1 - log(π_ref/π_θ)
```
Lower variance, still unbiased.

**Code:** `ppo_trainer.py:589`

From [Approximating KL Divergence](http://joschu.net/blog/kl-approx.html).

---

## Reward Shaping

### Combining Rewards

```python
# Base reward from reward model (only at sequence end)
rewards = zeros_like(response)
rewards[sequence_end] = reward_model_score

# Add KL penalty at each token
rewards += -kl_coef * kl_per_token

# Optionally whiten (normalize)
if whiten:
    rewards = (rewards - mean) / std
```

**Code:** `ppo_trainer.py:586-599`

### Whitening

```python
whitened = (values - mean(values)) / std(values)
```

**Why?** Normalizes rewards across batch, making gradients more stable.

**When to use?**
- **Large batch sizes (>= 32):** Safe to whiten
- **Small batches (< 16):** Skip whitening (statistics are noisy)

---

## Value Function Training

### Clipped Value Loss

```
L_V = E[max((V(s) - R)^2,
             (clip(V(s), V_old - ε_v, V_old + ε_v) - R)^2)]

where R = advantage + V_old (the return)
```

**Code:** `ppo_trainer.py:644-652`

**Why clip?** Same reason as policy clipping - prevent destructive updates.

**Typical ε_v:** 0.2 (same as policy cliprange)

---

## Hyperparameter Guide

### Critical Hyperparameters

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| `kl_coef` | 0.01-0.1 | 0.05 | KL penalty strength |
| `cliprange` | 0.1-0.3 | 0.2 | Policy clip range |
| `num_ppo_epochs` | 1-8 | 4 | Epochs per batch |
| `gamma` | 0.95-1.0 | 1.0 | Discount factor |
| `lam` | 0.9-0.99 | 0.95 | GAE lambda |
| `vf_coef` | 0.05-0.5 | 0.1 | Value loss weight |

### Tuning Strategies

**Problem: KL divergence too high (>0.1)**
```
Solution:
- Increase kl_coef (e.g., 0.05 → 0.1)
- Decrease cliprange (e.g., 0.2 → 0.1)
- Decrease learning rate
```

**Problem: Policy not improving**
```
Solution:
- Decrease kl_coef (e.g., 0.05 → 0.02)
- Increase cliprange (e.g., 0.2 → 0.3)
- Increase learning rate
- Check reward model quality
```

**Problem: Training unstable (loss spikes)**
```
Solution:
- Decrease cliprange
- Increase num_ppo_epochs
- Enable reward whitening
- Decrease learning rate
```

**Problem: Overfitting to reward model**
```
Solution:
- Increase kl_coef
- Decrease num_ppo_epochs
- Use larger, more diverse prompts
```

---

## Memory and Compute

### Memory Breakdown (7B model)

```
Policy (trainable):       14 GB (bf16)
Reference (frozen):       14 GB
Value (trainable):        14 GB
Reward (frozen):          14 GB
Activations:              15-20 GB
Rollout storage:          5-10 GB
Total:                    ~75-85 GB
```

**Optimizations:**

1. **LoRA on policy + value:**
   - Saves: 28 GB (frozen adapters for reference)
   - New total: ~50 GB

2. **Quantize reward model:**
   - Saves: 7-10 GB (8bit) or 10-12 GB (4bit)
   - New total: ~40-47 GB

3. **Quantize value model:**
   - Saves: 7-10 GB
   - New total: ~30-40 GB

4. **DeepSpeed ZeRO-3:**
   - Can train with even less per-GPU memory
   - Tradeoff: Slower generation

**Result:** Can fit on A100 40GB with LoRA + quantization!

### Compute Bottleneck

**Generation is the slowest part:**
```
Typical batch:
- Generation: 10-30 seconds (depends on response_length)
- PPO training: 2-5 seconds (4 epochs)
```

**Speedups:**
- Use vLLM for faster generation
- Reduce response_length
- Use speculative decoding
- Use larger batch sizes

---

## Comparison with Other Methods

| Method | Models | Online | Sample Efficiency | Quality | Memory |
|--------|--------|--------|-------------------|---------|--------|
| **PPO** | 4 | Yes | Low (needs generation) | Highest | Very High |
| **DPO** | 2 | No | High (offline dataset) | High | Medium |
| **RLOO** | 3 | Yes | Medium | High | High |
| **GRPO** | 2 | Yes | Medium-High | High | Medium |

**When to use PPO:**
- Maximum quality needed
- Have reward model (not just preferences)
- Have compute budget
- Want online learning (model improves from own outputs)

**When NOT to use PPO:**
- Limited compute
- Fast iteration needed → Use DPO
- No reward model → Use DPO
- Small dataset → Use DPO

---

## Common Pitfalls

### 1. Forgetting to freeze reference/reward models
```python
# BAD
ref_model.train()  # Will update during training!

# GOOD
ref_model.eval()
for param in ref_model.parameters():
    param.requires_grad = False
```

### 2. Not handling padding correctly
```python
# BAD
advantages = compute_advantages(values, rewards)

# GOOD
advantages = masked_compute_advantages(values, rewards, ~padding_mask)
```

Padding tokens should NOT contribute to loss or advantages.

### 3. Wrong KL direction
```python
# BAD (penalizes ref being different from policy)
kl = log π_ref - log π_θ

# GOOD (penalizes policy being different from ref)
kl = log π_θ - log π_ref
```

### 4. Not enough mini-batches
```python
# BAD (large variance)
num_mini_batches = 1

# GOOD
num_mini_batches = 4  # or more
```

More mini-batches = more stable training.

### 5. Forgetting to detach old values
```python
# BAD (backprop through old values)
ratio = new_logprobs / old_logprobs

# GOOD (old_logprobs are constants)
ratio = exp(new_logprobs - old_logprobs.detach())
```

In TRL, old_logprobs are computed in rollout phase (no_grad), so this is automatic.

---

## Summary

**PPO in 3 sentences:**
1. Generate responses, score them, compute advantages using a value function.
2. Update policy to increase good actions (high advantage) but clip updates to prevent destruction.
3. Update value function to better predict returns.

**Key innovations:**
- **Clipped objective:** Trust region without complicated math
- **GAE:** Bias-variance balance in advantage estimation
- **Actor-critic:** Value function reduces variance

**The math → code pipeline:**
```
Math:     L^CLIP = E[min(r·Â, clip(r, 1-ε, 1+ε)·Â)]
          ↓
Code:     pg_loss = mean(max(-ratio * advantage,
                              -clip(ratio, 1-ε, 1+ε) * advantage))
          ↓
Location: ppo_trainer.py:656-661
```

PPO is the gold standard for RLHF when quality matters more than speed!
