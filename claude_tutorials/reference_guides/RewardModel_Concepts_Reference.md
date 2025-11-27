# Reward Model Concepts Reference

## Bradley-Terry Model

### Foundation

Models pairwise preferences as:

```
P(A > B) = exp(r_A) / (exp(r_A) + exp(r_B))
         = σ(r_A - r_B)
         = 1 / (1 + exp(r_B - r_A))
```

### Derivation

Assume each item has latent "strength" r. Probability of A winning:

```
P(A > B) = P(strength_A > strength_B)
         = P(r_A + ε_A > r_B + ε_B)    (with Gumbel noise ε)
         = σ(r_A - r_B)                (closed form)
```

### Loss Function

Maximum likelihood estimation:

```
L = -Σ log P(y_w > y_l)
  = -Σ log σ(r_w - r_l)
  = Σ log(1 + exp(r_l - r_w))
```

**Code:** `reward_trainer.py:compute_loss()`

## Practical Considerations

### Dataset Size

**Minimum:** 1,000 preference pairs
**Good:** 10,000+ pairs
**Excellent:** 100,000+ pairs

**Why?** Reward models generalize slowly - need diverse examples.

### Prompt Distribution

**Critical:** Reward model must cover downstream task distribution.

If training RM on coding preferences, but using PPO for math → poor performance.

### Preference Quality

**Sources ranked by quality:**
1. Expert human labelers (expensive, best)
2. Crowdsourced humans (cheap, noisy)
3. AI-generated (cheapest, noisiest)

**Rule of thumb:** Better 1,000 high-quality pairs than 100,000 low-quality.

## Reward Scale

### Raw Scores

Reward models output **unbounded** scalars. Typical ranges:

```
Llama-7B reward model:  -5 to +5
Qwen-7B reward model:   -10 to +10
```

**Key point:** Absolute values don't matter, only **relative ordering**.

### Normalization

Sometimes useful to normalize for interpretability:

```python
# Z-score normalization
r_normalized = (r - mean) / std

# Sigmoid to [0, 1]
r_normalized = σ(r)
```

**When to normalize:** If combining multiple reward models or displaying to users.

## Advanced: Ensemble Models

Train multiple reward models, average scores:

```python
r_ensemble = mean([r_1, r_2, r_3])
```

**Benefits:**
- Robustness to individual model failures
- Better generalization
- Reduces reward hacking in PPO

**Cost:** 3x inference time.

## Metrics

### Accuracy

```
accuracy = (r_chosen > r_rejected).mean()
```

**Interpretation:**
- 50% = random
- 70% = decent
- 80% = good
- 90%+ = possibly overfitting (check generalization!)

### Ranking Correlation

For test sets with multiple responses per prompt:

```python
# Spearman rank correlation
from scipy.stats import spearmanr

true_ranking = [1, 2, 3, 4, 5]  # Human ranking
model_scores = [0.5, 0.3, 0.8, 0.2, 0.9]

correlation = spearmanr(true_ranking, model_scores)
```

**Better metric** than binary accuracy for multi-response scenarios.

## Comparison with Other Approaches

| Method | Data Needed | Output | Complexity |
|--------|-------------|--------|------------|
| **Reward Model** | Pairwise prefs | Scalar score | Medium |
| **DPO** | Pairwise prefs | Implicit reward | Low |
| **RLHF (PPO)** | Pairwise prefs → RM → PPO | Optimized policy | High |

**When to use reward models:**
- Need explicit scores (for filtering, ranking)
- Doing PPO/GRPO (need reward function)
- Want flexibility (can swap models easily)

**When NOT to use:**
- Just want aligned model → Use DPO (skip RM training)
- Limited data (<1k pairs) → DPO more sample efficient
