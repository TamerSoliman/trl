# GRPO Concepts: Group-Relative Advantages

## Core Formula

```
Advantage_i = Reward_i - mean(Rewards_in_group)

where group = all K completions for same prompt
```

**Code:** `grpo_trainer.py:1200-1250`

## Why It Works

**Traditional RL baseline:** V(s) from trained value function
**GRPO baseline:** Empirical mean of K samples

**Theorem:** E[mean(R_1, ..., R_K)] = V(s) as K → ∞

**Practical:** K=4-8 gives low enough variance

## Comparison

```
Method          | Baseline        | Variance | Bias
----------------|-----------------|----------|------
REINFORCE       | 0               | High     | None
GRPO (K=4)      | Group mean      | Medium   | Low
PPO             | V_φ(s)          | Low      | Medium
```

## Loss Functions

**Policy loss:**
```
L_policy = -E[A_i * log(π_θ/π_ref)]
```

**KL regularization:**
```
L_KL = β * E[KL(π_θ || π_ref)]
```

**Total:**
```
L = L_policy + L_KL
```

## Hyperparameters

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| num_generations | 2-16 | 4 | More = lower variance, slower |
| kl_coef | 0.01-0.1 | 0.05 | KL penalty strength |
| learning_rate | 1e-7 to 1e-5 | 1e-6 | Step size |

## Memory Formula

```
Memory = Model_size + K * (seq_len * batch_size * hidden_dim * 2)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                      Storing K generations
```

For K=4, this adds ~4x generation memory vs single generation.

**Optimization:** Use vLLM with PagedAttention to reduce this.
