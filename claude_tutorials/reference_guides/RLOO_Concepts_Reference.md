# RLOO Concepts Reference

## Leave-One-Out Baseline

### Definition

For G samples from same prompt:
```
Baseline(sample i) = mean(all others except i)
                   = (sum_all - sample_i) / (G - 1)
```

### Mathematical Derivation

**Goal:** Reduce variance in policy gradient

**Standard REINFORCE:**
```
∇J = E_y~π[r(y) * ∇log π(y|x)]
```

**With baseline b:**
```
∇J = E_y~π[(r(y) - b) * ∇log π(y|x)]
```

**Bias requirement:** E[b * ∇log π] = 0

**RLOO baseline:**
For samples {y₁, ..., yG} from same prompt:
```
b(yᵢ) = (1/(G-1)) * Σⱼ≠ᵢ r(xⱼ, yⱼ)
```

**Unbiased proof:**
```
E[b(yᵢ) * ∇log π(yᵢ|x)] = E[(1/(G-1)) * Σⱼ≠ᵢ r(yⱼ) * ∇log π(yᵢ|x)]
                        = (1/(G-1)) * Σⱼ≠ᵢ E[r(yⱼ)] * E[∇log π(yᵢ|x)]
                        ≈ 0  (when yᵢ independent of yⱼ)
```

### Variance Reduction

**Variance without baseline:**
```
Var[r(y) * ∇log π] = E[r²] * E[(∇log π)²] - E[r]² * E[∇log π]²
```

**Variance with LOO baseline:**
```
Var[(r - b_LOO) * ∇log π] << Var[r * ∇log π]
```

**Empirical reduction:** ~50-80% variance reduction with G=4

---

## Clipped Objective

### PPO-Style Clipping

```
ratio = π_θ(y|x) / π_old(y|x)

L_clip = -E[min(
    ratio * advantage,
    clip(ratio, 1-ε, 1+ε) * advantage
)]
```

### Why Clip?

**Without clipping:**
```
∇L = -advantage * (ratio * ∇log π)
```

If ratio >> 1 and advantage > 0 → huge gradient!

**With clipping:**
```
∇L = {
    -advantage * ∇log π           if ratio ∈ [1-ε, 1+ε]
    0                              if outside range AND sign matches
}
```

**Effect:** Prevents destructive updates

### Asymmetric Clipping (DAPO)

**Standard:** ε_low = ε_high = 0.2

**DAPO variant:**
```
ε_low = 0.2
ε_high = 0.28  (more permissive for improvements)
```

**Rationale:** Allow larger increases when advantage > 0

---

## Hyperparameter Guide

### num_generations

**Range:** 2-16
**Default:** 4

**Effect:**
- **Low (2):** Fast, high variance
- **Medium (4-8):** Balanced
- **High (16+):** Slow, low variance

**Rule of thumb:** Start with 4, increase if variance is high

### epsilon (clip range)

**Range:** 0.1 - 0.3
**Default:** 0.2

**Effect:**
- **Low (0.1):** Conservative updates, stable but slow
- **High (0.3):** Aggressive updates, faster but less stable

### beta (KL coefficient)

**Range:** 0.0 - 0.1
**Default:** 0.05

**Effect:**
- **0.0:** No KL penalty (no reference model needed!)
- **0.01:** Light regularization
- **0.05:** Standard
- **0.1:** Strong regularization

**Memory tip:** Set `beta=0` to save ~14GB (no reference model)

### normalize_advantages

**Default:** True

**Effect:**
- **True:** Advantages have mean=0, std=1
- **False:** Raw advantage values

**Recommendation:** Always True for stability

---

## Training Dynamics

### Phase 1: Exploration (0-20% steps)

**Behavior:**
- Rewards vary widely
- High variance in advantages
- Policy exploring

**Metrics:**
- `reward_std` high
- `clip_ratio` low (few clips)

### Phase 2: Exploitation (20-80% steps)

**Behavior:**
- Policy converging
- Rewards increasing
- Variance decreasing

**Metrics:**
- `reward` increasing steadily
- `reward_std` decreasing
- `clip_ratio` increasing (0.05-0.15)

### Phase 3: Convergence (80-100% steps)

**Behavior:**
- Rewards plateauing
- Policy stable

**Metrics:**
- `reward` flat
- `clip_ratio` high (> 0.15)
- `kl` stable

---

## Batch Size Constraint

### Divisibility Requirement

```
effective_batch_size % num_generations == 0
```

where:
```
effective_batch_size = (
    per_device_batch_size *
    num_gpus *
    gradient_accumulation_steps
)
```

**Why?** Each prompt needs exactly G completions in same batch.

**Example:**
```
per_device_batch_size = 2
num_gpus = 4
gradient_accumulation_steps = 2
num_generations = 4

effective = 2 * 4 * 2 = 16
16 % 4 = 0  ✓
```

---

## Comparison Table

| Aspect | RLOO | PPO | REINFORCE |
|--------|------|-----|-----------|
| **Baseline** | Leave-one-out | Value function | None or constant |
| **Variance** | Medium-Low | Low | Very High |
| **Bias** | Unbiased | Unbiased (if V accurate) | Unbiased |
| **Complexity** | Low | High | Very Low |
| **Extra networks** | 0 | 1 (value) | 0 |
| **Samples needed** | G per prompt | 1 per prompt | Many prompts |

**Trade-off:** RLOO trades samples (G > 1) for simpler architecture (no V)

---

## vLLM Integration

### Why vLLM?

**Problem:** Generating G completions per prompt is slow

**Solution:** vLLM (10-20x faster generation)

**Modes:**
1. **Server:** Separate vLLM server
2. **Colocate:** vLLM shares training GPUs

### Memory Trade-offs

**Without vLLM:**
- Training uses all GPU memory
- Generation uses model in training mode

**With vLLM (colocate):**
- Reserve 30% GPU for vLLM (`vllm_gpu_memory_utilization=0.3`)
- 70% for training
- Total: Can train larger models!

**With vLLM (server):**
- Training GPU: 100% for training
- Separate GPU: 100% for generation
- Best throughput, but needs extra hardware

---

## Summary

**RLOO = REINFORCE + LOO Baseline + PPO Clipping**

**Key equations:**
```
Baseline(i) = (Σⱼ≠ᵢ r(j)) / (G-1)
Advantage(i) = r(i) - Baseline(i)
L = -E[min(ratio*A, clip(ratio)*A)]
```

**Ideal for:**
- Prototyping RL
- When PPO feels too complex
- Research on policy gradient methods

**Not ideal for:**
- Maximum sample efficiency (use PPO)
- Fast training (multiple generations = slow)
- Preference data (use DPO)
