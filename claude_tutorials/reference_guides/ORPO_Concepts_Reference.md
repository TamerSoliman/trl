# ORPO Concepts Reference

## Odds vs. Probability

### Definitions

**Probability:**
```
P(event) = # favorable / # total
Range: [0, 1]
```

**Odds:**
```
Odds(event) = P(event) / (1 - P(event))
            = P(event) / P(not event)
Range: [0, ∞)
```

### Conversion

**Probability → Odds:**
```
Odds = P / (1-P)
```

**Odds → Probability:**
```
P = Odds / (1 + Odds)
```

### Examples

| Probability | Odds | Interpretation |
|-------------|------|----------------|
| 0.5 | 1:1 | Even odds |
| 0.75 | 3:1 | 3x more likely to happen |
| 0.9 | 9:1 | 9x more likely to happen |
| 0.1 | 1:9 | 9x more likely NOT to happen |

### Why Odds?

**Symmetry:** Odds are symmetric around 1

```
P=0.9 → Odds=9   (9x more likely to happen)
P=0.1 → Odds=1/9 (9x more likely NOT to happen)
```

In log space: log(9) = -log(1/9) = 2.197

**Probability ratios** are NOT symmetric:
```
P_1=0.9, P_2=0.1 → Ratio = 9
P_1=0.1, P_2=0.9 → Ratio = 1/9 (different magnitude!)
```

---

## Odds Ratio

### Definition

**Odds Ratio:** Ratio of two odds

```
OR = Odds_A / Odds_B
   = [P_A/(1-P_A)] / [P_B/(1-P_B)]
```

### Properties

**Symmetric in log space:**
```
log OR(A, B) = -log OR(B, A)
```

**Range:**
- OR = 1: A and B equally likely
- OR > 1: A more likely than B
- OR < 1: B more likely than A

### Application to ORPO

```
OR(chosen, rejected | x) = [P(y_w|x)/(1-P(y_w|x))] / [P(y_l|x)/(1-P(y_l|x))]
```

**Goal:** Maximize OR → Make chosen much more likely than rejected

---

## ORPO Loss Derivation

### Starting Point

**Goal:** Learn a model that prefers chosen over rejected outputs

**Approach:** Maximize odds ratio OR(chosen, rejected)

### Step 1: Express in Log Space

```
log OR = log[P(y_w|x)/(1-P(y_w|x))] - log[P(y_l|x)/(1-P(y_l|x))]
       = [log P(y_w|x) - log(1-P(y_w|x))] - [log P(y_l|x) - log(1-P(y_l|x))]
```

### Step 2: Convert Probabilities to Log Probabilities

For a sequence y = [y_1, ..., y_T]:

```
P(y|x) = ∏ᵢ P(yᵢ|x, y<i)

log P(y|x) = Σᵢ log P(yᵢ|x, y<i)  [sum of token log probs]
```

But for average (not sum):
```
avg_log_p = (1/T) * Σᵢ log P(yᵢ|x, y<i)
```

### Step 3: Log of Complement

```
log(1 - P(y|x)) = ?
```

**Challenge:** No closed form for sequences!

**ORPO approximation:**
```
log(1 - exp(avg_log_p)) ≈ log1p(-exp(avg_log_p))
```

**torch.log1p(x) = log(1 + x)**, numerically stable for small x

### Step 4: Final Log Odds

```
log_odds = [log_p_chosen - log1p(-exp(log_p_chosen))] -
           [log_p_rejected - log1p(-exp(log_p_rejected))]
```

**Code:**
```python
log_odds = (
    (policy_chosen_logps - policy_rejected_logps) -
    (torch.log1p(-torch.exp(policy_chosen_logps)) -
     torch.log1p(-torch.exp(policy_rejected_logps)))
)
```

### Step 5: Preference Loss

**Want:** Maximize log_odds (higher = better preference)

**Loss term:**
```
L_preference = -log sigmoid(log_odds)
```

**Why sigmoid?** Ensures loss is bounded and smooth.

**In practice:** Use `logsigmoid` for numerical stability
```python
L_preference = -F.logsigmoid(log_odds)
```

### Step 6: Weighted Preference Loss

```
L_preference_weighted = -β * E[log σ(log_odds)]
```

where β controls the weight of preference vs. SFT.

### Step 7: Combined with NLL

**Full ORPO loss:**
```
L_ORPO = L_NLL(y_chosen) - β * E[log σ(log_odds)]
         ^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^
            SFT term         Preference term
```

**Note the minus:** We SUBTRACT preference loss because we want to MINIMIZE total loss, but MAXIMIZE log odds.

**Code location:** `orpo_trainer.py:655-686`, `orpo_trainer.py:835`

---

## Why NLL is Necessary

### Without NLL

**Imagine:** L = -β * log σ(log_odds)

**Problem:** Model can minimize loss by making BOTH chosen and rejected probabilities small!

**Example:**
```
log_p_chosen = -100
log_p_rejected = -200

log_odds = [-100 - log(1-exp(-100))] - [-200 - log(1-exp(-200))]
         ≈ -100 - (-200) = 100  [very positive!]

Loss = -β * log σ(100) ≈ 0  [very low!]
```

But the model outputs gibberish for both chosen and rejected!

### With NLL

**Full loss:** L = L_NLL(chosen) - β * log σ(log_odds)

**Now:**
- If model makes chosen probability small → NLL increases
- Model forced to keep chosen probability high
- Preference term still encourages gap with rejected

**Result:** Model learns to generate chosen outputs well AND distinguish them from rejected.

---

## Comparison with DPO Loss

### DPO Loss

```
L_DPO = -log σ(β * [r_chosen - r_rejected])

where r_y = log π_θ(y|x) - log π_ref(y|x)
```

**Key components:**
1. Implicit SFT via reference model constraint
2. Explicit preference via sigmoid loss

### ORPO Loss

```
L_ORPO = L_NLL(chosen) - β * log σ(log_odds)

where log_odds = [log P(y_w) - log(1-P(y_w))] - [log P(y_l) - log(1-P(y_l))]
```

**Key components:**
1. Explicit SFT via NLL
2. Explicit preference via odds ratio

### Conceptual Difference

**DPO:**
- Compares current policy to **reference**
- Reference provides implicit SFT signal
- Preference is relative to fixed baseline

**ORPO:**
- Compares chosen to **rejected**
- NLL provides explicit SFT signal
- Preference is within-batch comparison

### Mathematical Relationship

**If we expand DPO:**
```
L_DPO = -log σ(β * [(log π - log π_ref)_chosen - (log π - log π_ref)_rejected])
      = -log σ(β * [log π_chosen - log π_rejected - (log π_ref_chosen - log π_ref_rejected)])
```

**If π_ref is uniform or cancelled:** This becomes similar to comparing log π_chosen vs. log π_rejected.

**ORPO goes further:** Uses odds instead of just log probabilities, which accounts for the complement (1-P).

---

## Hyperparameter: Beta (λ)

### Role of Beta

**In loss:**
```
L = L_NLL - β * preference_term
```

**Effect:**
- **β = 0:** Pure SFT (no preference learning)
- **β small (0.05):** Mostly SFT, light preference shaping
- **β medium (0.1):** Balanced SFT + preference
- **β large (0.5+):** Heavy preference, may hurt generation quality

### Optimal Range

**Typical:** 0.1 - 0.2

**Tuning strategy:**
1. Start with β=0.1
2. Monitor metrics:
   - NLL decreasing → SFT working
   - Margins increasing → Preference working
3. If margins not improving: increase β
4. If NLL not improving: decrease β

### Comparison to Other Methods

| Method | Parameter | Default |
|--------|-----------|---------|
| ORPO | β (lambda) | 0.1 |
| DPO | β (beta) | 0.1-0.5 |
| KTO | β (beta) | 0.1 |

**Note:** ORPO's β is typically **lower** than DPO's because ORPO has an explicit NLL term.

---

## Training Dynamics

### Phase 1: Early Training (0-20% steps)

**Behavior:**
- NLL drops rapidly (learning to generate chosen)
- Margins start low (model hasn't learned preferences yet)
- Accuracies near 50% (random)

**Healthy signs:**
- NLL < 3.0 after first epoch
- Margins > 0 (even if small)

### Phase 2: Mid Training (20-60% steps)

**Behavior:**
- NLL continues decreasing (slower)
- Margins increase steadily
- Accuracies climb (60-75%)

**Healthy signs:**
- Margins > 10
- Accuracies > 0.6
- Log odds ratio > 1.0

### Phase 3: Late Training (60-100% steps)

**Behavior:**
- NLL plateaus
- Margins plateau or increase slowly
- Accuracies plateau (75-85%)

**Healthy signs:**
- NLL stable (not increasing)
- Margins > 20
- Accuracies > 0.7

### Unhealthy Patterns

**Pattern 1: Margins not increasing**
- Cause: β too low or data quality issues
- Fix: Increase β, check data

**Pattern 2: NLL increasing late**
- Cause: β too high, forgetting to generate
- Fix: Decrease β, add more SFT data

**Pattern 3: Both probabilities dropping**
- Cause: Model collapse
- Fix: Lower β, increase learning rate, check data quality

---

## Metrics Interpretation

### NLL Loss

**Range:** 0.5 - 3.0 (typical)

**Interpretation:**
- < 1.0: Excellent generation quality
- 1.0 - 2.0: Good
- 2.0 - 3.0: Acceptable
- > 3.0: Poor (model struggling to generate)

### Log Odds Ratio

**Range:** -∞ to +∞

**Interpretation:**
- < 0: Rejected preferred (BAD!)
- 0 - 0.5: Weak preference
- 0.5 - 2.0: Moderate preference (good)
- 2.0+: Strong preference

**Formula:**
```
odds_ratio = exp(log_odds_ratio)
```

**Example:** log_odds = 2.0 → OR = e^2 ≈ 7.4 (chosen 7.4x more likely in odds terms)

### Reward Margins

**Range:** 0 to 100+ (higher is better)

**Interpretation:**
- < 5: Very weak separation
- 5 - 20: Weak separation
- 20 - 50: Good separation
- 50+: Excellent separation

**Definition:** avg(log_p_chosen) - avg(log_p_rejected)

### Accuracies

**Range:** 0.0 - 1.0

**Interpretation:**
- 0.5: Random (no learning)
- 0.6 - 0.7: Learning preferences
- 0.7 - 0.85: Good alignment
- 0.85+: Excellent (may be overfitting if >0.95)

---

## Memory Analysis

### Single Model vs. Two Models

**DPO/KTO (two models):**
```
Memory = Model_policy + Model_ref + Activations
       ≈ 14GB + 14GB + 2GB = 30GB  (for 7B model, bf16)
```

**ORPO (one model):**
```
Memory = Model_policy + Activations
       ≈ 14GB + 2GB = 16GB
```

**Savings:** ~47%

### Why Activations Don't Double?

**Concatenated forward:** ORPO runs chosen + rejected in **one** forward pass
```python
batch = concat([chosen_batch, rejected_batch])  # 2x bigger batch
outputs = model(batch)  # Single forward pass
```

**Memory:** Activations scale with batch size, but it's still more memory-efficient than two separate forward passes through two models.

---

## When to Use ORPO

### ✅ Use ORPO When:

1. **Starting from base model**
   - You want SFT + alignment in one stage
   - Don't have separately SFT'd model

2. **Memory constrained**
   - Single GPU with limited VRAM
   - Can't fit two 7B models

3. **Pairwise preference data**
   - Have clear chosen/rejected pairs
   - Data quality is high

4. **Fast iteration**
   - Want single-stage training
   - Simpler pipeline management

### ❌ Don't Use ORPO When:

1. **Already have SFT model**
   - Use DPO instead (more direct)
   - ORPO's NLL term wastes compute

2. **Binary-only feedback**
   - Use KTO (designed for binary labels)
   - ORPO needs pairs

3. **Need explicit reference**
   - For evaluation or KL constraints
   - ORPO has no reference model

4. **Maximum alignment quality**
   - PPO/DPO often achieve better alignment
   - ORPO trades some quality for simplicity

---

## Summary Table

| Concept | Formula | Interpretation |
|---------|---------|----------------|
| **Odds** | P/(1-P) | How many times more likely than not |
| **Odds Ratio** | Odds_A/Odds_B | Relative likelihood in odds terms |
| **Log Odds** | log[P/(1-P)] | Log-space odds (symmetric) |
| **ORPO Loss** | NLL - β*log σ(log_odds) | SFT + Preference |
| **Beta** | Weight on preference | Balances SFT vs. alignment |
| **NLL Term** | Standard cross-entropy | Prevents collapse |
| **Preference Term** | log σ(log_odds) | Widens gap between chosen/rejected |

**Key Innovation:** No reference model needed!

**Trade-off:** Combined objective requires careful tuning of β.
