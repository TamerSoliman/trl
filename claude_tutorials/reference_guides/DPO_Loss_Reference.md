# DPO Loss: Mathematical Foundation to Code Implementation

**Purpose**: This guide provides a complete mapping from the mathematical formulation of Direct Preference Optimization (DPO) to its exact implementation in the TRL library.

---

## Table of Contents

1. [Mathematical Foundations](#mathematical-foundations)
2. [From Theory to Implementation](#from-theory-to-implementation)
3. [Term-by-Term Code Mapping](#term-by-term-code-mapping)
4. [Loss Variants Explained](#loss-variants-explained)
5. [Practical Examples](#practical-examples)

---

## Mathematical Foundations

### The Preference Learning Problem

**Goal**: Given a dataset of preferences `D = {(x, y_w, y_l)}` where:
- `x` = prompt/instruction
- `y_w` = winning/chosen response (preferred by humans)
- `y_l` = losing/rejected response (not preferred)

We want to train a model `π_θ` that assigns higher probability to winning responses.

---

### Bradley-Terry Model

The probability that response `y_w` is preferred over `y_l` given prompt `x`:

```
P(y_w ≻ y_l | x) = σ(r(x, y_w) - r(x, y_l))
```

Where:
- `σ(z) = 1 / (1 + e^(-z))` is the sigmoid function
- `r(x, y)` is a reward function scoring response `y` for prompt `x`

**Maximum Likelihood Objective**:

```
max_r  𝔼_{(x, y_w, y_l) ~ D} [log σ(r(x, y_w) - r(x, y_l))]
```

This is the standard approach in RLHF: train a reward model `r`, then use RL to optimize `π_θ` against `r`.

---

### DPO's Key Insight: Reward Reparameterization

Instead of learning `r` explicitly, DPO reparameterizes it using the policy:

```
r(x, y) = β log(π_θ(y|x) / π_ref(y|x)) + β log Z(x)
```

Where:
- `π_θ` = policy model (being optimized)
- `π_ref` = reference model (frozen, usually the initial SFT model)
- `β` = temperature parameter
- `Z(x)` = partition function (depends only on `x`, not `y`)

**Why this form?** It's derived from the optimal policy for constrained RL with KL penalty.

---

### Substituting into Bradley-Terry

Compute the reward difference:

```
r(x, y_w) - r(x, y_l) = β log(π_θ(y_w|x) / π_ref(y_w|x)) - β log(π_θ(y_l|x) / π_ref(y_l|x))
                       = β [log π_θ(y_w|x) - log π_ref(y_w|x) - log π_θ(y_l|x) + log π_ref(y_l|x)]
```

Note: `Z(x)` cancels out since it doesn't depend on `y`.

Substitute into Bradley-Terry:

```
P(y_w ≻ y_l | x) = σ(β [log π_θ(y_w|x) - log π_ref(y_w|x) - log π_θ(y_l|x) + log π_ref(y_l|x)])
```

---

### DPO Loss Function

The negative log likelihood of the preference data:

```
L_DPO(π_θ; π_ref) = -𝔼_{(x, y_w, y_l) ~ D} [log σ(β · logits)]
```

Where:

```
logits = log π_θ(y_w|x) - log π_θ(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x)
```

**Expanding** `log σ(z) = -log(1 + e^(-z)) = -softplus(-z)`:

```
L_DPO = 𝔼[log(1 + exp(-β · logits))]
```

**Alternative form** using `log σ(z) + log σ(-z) = 0`:

```
L_DPO = 𝔼[softplus(-β · logits)]
```

Where `softplus(x) = log(1 + e^x)`.

---

## From Theory to Implementation

### Step 1: Compute Log Probabilities

For a sequence `y = [y_1, y_2, ..., y_T]` given prompt `x`:

```
log π(y|x) = Σ_{t=1}^T log π(y_t | y_{<t}, x)
```

**In code** (`concatenated_forward()` lines 1479-1725):

```python
# Get logits from model
outputs = model(input_ids, attention_mask)
logits = outputs.logits[:, :-1, :]  # Shift for next token prediction

# Get log probabilities for actual tokens
labels = input_ids[:, 1:]  # Shift labels
all_logps = selective_log_softmax(logits, labels)
# all_logps[i, t] = log P(labels[i, t] | context)

# Sum over sequence (only completion tokens, not prompt)
log_prob = (all_logps * loss_mask).sum(dim=-1)
```

**What is `selective_log_softmax`?**

It computes `log softmax(logits)[labels]`, i.e., the log probability of the actual token that appeared:

```python
def selective_log_softmax(logits, labels):
    # logits: [batch, seq_len, vocab_size]
    # labels: [batch, seq_len]
    log_probs = F.log_softmax(logits, dim=-1)  # [batch, seq_len, vocab_size]
    selected = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1))
    return selected.squeeze(-1)  # [batch, seq_len]
```

---

### Step 2: Compute Log Ratios

For chosen and rejected responses:

```
chosen_logratio = log π_θ(y_w|x) - log π_ref(y_w|x)
rejected_logratio = log π_θ(y_l|x) - log π_ref(y_l|x)
```

**In code** (`dpo_loss()` line 1077-1078):

```python
chosen_logratios = chosen_logps - ref_chosen_logps
rejected_logratios = rejected_logps - ref_rejected_logps
```

---

### Step 3: Compute Logits

```
logits = log π_θ(y_w|x) - log π_θ(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x)
```

Rearranging:

```
logits = [log π_θ(y_w|x) - log π_θ(y_l|x)] - [log π_ref(y_w|x) - log π_ref(y_l|x)]
       = policy_logit_diff - ref_logit_diff
```

**In code** (`dpo_loss()` lines 1092-1100):

```python
logratios = chosen_logps - rejected_logps  # Policy difference
ref_logratios = ref_chosen_logps - ref_rejected_logps  # Reference difference
logits = logratios - ref_logratios
```

---

### Step 4: Compute Loss (Sigmoid Variant)

```
L = -log σ(β · logits) = log(1 + exp(-β · logits)) = softplus(-β · logits)
```

**In code** (`dpo_loss()` lines 1114-1118):

```python
losses = -F.logsigmoid(self.beta * logits)
```

`F.logsigmoid(x)` computes `log(σ(x))` numerically stably.

So `-F.logsigmoid(β · logits)` = `-log σ(β · logits)` = our DPO loss!

---

## Term-by-Term Code Mapping

| Math Expression | Code Variable | File:Line | Notes |
|----------------|---------------|-----------|-------|
| `x` (prompt) | `batch["prompt_input_ids"]` | dpo_trainer.py:151 | Tokenized prompt |
| `y_w` (chosen) | `batch["chosen_input_ids"]` | dpo_trainer.py:153 | Tokenized chosen response |
| `y_l` (rejected) | `batch["rejected_input_ids"]` | dpo_trainer.py:155 | Tokenized rejected response |
| `log π_θ(y_w\|x)` | `chosen_logps` | dpo_trainer.py:1632 | From policy forward pass |
| `log π_θ(y_l\|x)` | `rejected_logps` | dpo_trainer.py:1633 | From policy forward pass |
| `log π_ref(y_w\|x)` | `ref_chosen_logps` | dpo_trainer.py:1632 | From reference forward pass |
| `log π_ref(y_l\|x)` | `ref_rejected_logps` | dpo_trainer.py:1633 | From reference forward pass |
| `β` | `self.beta` | dpo_config.py:30 | Temperature parameter |
| `log(π_θ / π_ref)` for chosen | `chosen_logratios` | dpo_trainer.py:1077 | = `chosen_logps - ref_chosen_logps` |
| `log(π_θ / π_ref)` for rejected | `rejected_logratios` | dpo_trainer.py:1078 | = `rejected_logps - ref_rejected_logps` |
| Logits | `logits` | dpo_trainer.py:1100 | = `logratios - ref_logratios` |
| `σ(β · logits)` | `F.logsigmoid(self.beta * logits)` | dpo_trainer.py:1116 | In log space |
| `L_DPO` | `losses` | dpo_trainer.py:1115-1118 | = `-F.logsigmoid(...)` |
| Implicit reward (chosen) | `chosen_rewards` | dpo_trainer.py:1242 | = `β * chosen_logratios` |
| Implicit reward (rejected) | `rejected_rewards` | dpo_trainer.py:1243 | = `β * rejected_logratios` |

---

## Loss Variants Explained

DPO supports 15+ loss variants. Here are the most important ones:

### 1. Sigmoid Loss (Default, Original DPO)

**Math**:
```
L_sigmoid = -log σ(β · logits)
```

**Code** (line 1114-1118):
```python
losses = -F.logsigmoid(self.beta * logits)
```

**Intuition**:
- Maximize `σ(β · logits)`, which means increase the preference gap
- When `logits > 0`: chosen is preferred (good) → low loss
- When `logits < 0`: rejected is preferred (bad) → high loss

**With Label Smoothing** (ε):
```
L = -log σ(β · logits) · (1-ε) - log σ(-β · logits) · ε
```

```python
losses = (
    -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
    - F.logsigmoid(-self.beta * logits) * self.label_smoothing
)
```

Label smoothing adds regularization assuming labels might be noisy.

---

### 2. IPO Loss (Identity Preference Optimization)

**Paper**: [Azar et al., 2023](https://arxiv.org/abs/2310.12036)

**Math**:
```
L_IPO = (logits - 1/(2β))²
```

**Code** (line 1139-1141):
```python
losses = (logits - 1 / (2 * self.beta)) ** 2
```

**Intuition**:
- MSE loss with target `1/(2β)`
- More robust to outliers than sigmoid
- Less sensitive to β choice

**When to use**: When you have noisy preference data.

---

### 3. Hinge Loss (SLiC)

**Paper**: [Zhao et al., 2023](https://arxiv.org/abs/2305.10425)

**Math**:
```
L_hinge = max(0, 1 - β · logits)
```

**Code** (line 1136-1137):
```python
losses = torch.relu(1 - self.beta * logits)
```

**Intuition**:
- Margin-based loss (like SVM)
- Only penalize when margin < 1
- No penalty when `β · logits ≥ 1`

**When to use**: When you want margin-based learning.

---

### 4. APO-Zero Loss (Alignment via Preference Optimization)

**Paper**: [Zeng et al., 2024](https://arxiv.org/abs/2408.06266)

**Math**:
```
L_apo_zero = (1 - σ(β · chosen_logratio)) + σ(β · rejected_logratio)
```

**Code** (line 1195-1200):
```python
losses_chosen = 1 - F.sigmoid(self.beta * chosen_logratios)  # Increase chosen
losses_rejected = F.sigmoid(self.beta * rejected_logratios)  # Decrease rejected
losses = losses_chosen + losses_rejected
```

**Intuition**:
- Directly push up chosen likelihood
- Directly push down rejected likelihood
- Symmetric treatment of chosen/rejected

**When to use**: When you believe chosen outputs are better than your model's default.

---

### 5. Robust DPO

**Paper**: [Wu et al., 2024](https://arxiv.org/abs/2403.00409)

**Math**:
```
L_robust = [-log σ(β · logits) + log σ(-β · logits)] / (1 - 2ε)
```

**Code** (line 1120-1124):
```python
losses = (
    -F.logsigmoid(self.beta * logits)
    + F.logsigmoid(-self.beta * logits)  # Note: + instead of -
) / (1 - 2 * self.label_smoothing)
```

**Intuition**:
- Unbiased estimate robust to preference noise
- Suitable when human annotations are unreliable

---

## Practical Examples

### Example 1: Computing DPO Loss by Hand

**Setup**:
- Prompt: "What is 2+2?"
- Chosen: "2+2 equals 4."
- Rejected: "I don't know."
- β = 0.5

**Step 1**: Get log probabilities (from model forward pass):

```python
chosen_logps = torch.tensor([-2.3])     # log π_θ(chosen|prompt)
rejected_logps = torch.tensor([-4.2])   # log π_θ(rejected|prompt)
ref_chosen_logps = torch.tensor([-2.5]) # log π_ref(chosen|prompt)
ref_rejected_logps = torch.tensor([-3.8]) # log π_ref(rejected|prompt)
```

**Step 2**: Compute log ratios:

```python
chosen_logratios = chosen_logps - ref_chosen_logps
                 = -2.3 - (-2.5)
                 = 0.2

rejected_logratios = rejected_logps - ref_rejected_logps
                   = -4.2 - (-3.8)
                   = -0.4
```

**Interpretation**:
- Policy assigns higher probability to chosen than reference (0.2 > 0) ✓
- Policy assigns lower probability to rejected than reference (-0.4 < 0) ✓

**Step 3**: Compute logits:

```python
logratios = chosen_logps - rejected_logps
          = -2.3 - (-4.2)
          = 1.9

ref_logratios = ref_chosen_logps - ref_rejected_logps
              = -2.5 - (-3.8)
              = 1.3

logits = logratios - ref_logratios
       = 1.9 - 1.3
       = 0.6
```

**Interpretation**: Policy separates chosen/rejected more than reference (0.6 > 0) ✓

**Step 4**: Compute loss:

```python
beta_logits = 0.5 * 0.6 = 0.3

sigma = 1 / (1 + exp(-0.3)) = 0.574

loss = -log(0.574) = 0.554
```

**Result**: Loss is 0.554. Since this is relatively low (< 1), the policy is doing well!

---

### Example 2: Understanding the Gradient

**Question**: How does the loss gradient encourage the policy to prefer chosen over rejected?

**Math**:

```
∂L/∂logits = ∂/∂logits [-log σ(β · logits)]
           = -σ'(β · logits) · β / σ(β · logits)
           = -σ(β · logits) · (1 - σ(β · logits)) · β / σ(β · logits)
           = -β · (1 - σ(β · logits))
           = β · (σ(β · logits) - 1)
```

Since `0 < σ < 1`, we have `σ - 1 < 0`, so `∂L/∂logits < 0`.

**Interpretation**: The loss wants to **increase** `logits`.

Now, `logits = log π_θ(chosen) - log π_θ(rejected) - [constant from ref]`.

So increasing `logits` means:
- **Increase** `log π_θ(chosen)` → increase probability of chosen ✓
- **Decrease** `log π_θ(rejected)` → decrease probability of rejected ✓

---

### Example 3: Effect of β

**Scenario**: Same setup as Example 1, but try β = 0.1 vs β = 1.0.

**β = 0.1** (low temperature):
```
beta_logits = 0.1 * 0.6 = 0.06
sigma = 1 / (1 + exp(-0.06)) = 0.515
loss = -log(0.515) = 0.663
```

**β = 1.0** (high temperature):
```
beta_logits = 1.0 * 0.6 = 0.6
sigma = 1 / (1 + exp(-0.6)) = 0.646
loss = -log(0.646) = 0.437
```

**Observation**:
- Higher β → Lower loss (for same `logits`)
- Higher β → Sharper preference distinction
- Lower β → Softer preference, more conservative updates

**Rule of thumb**:
- β ∈ [0.1, 0.3]: Conservative, good for noisy data
- β ∈ [0.3, 0.5]: Standard, works for most cases
- β > 0.5: Aggressive, use with caution

---

## Summary

### Key Equations

1. **DPO Loss**:
   ```
   L_DPO = -log σ(β · [log π_θ(y_w|x) - log π_θ(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x)])
   ```

2. **Implicit Reward**:
   ```
   r(x, y) = β log(π_θ(y|x) / π_ref(y|x))
   ```

3. **Preference Probability**:
   ```
   P(y_w ≻ y_l | x) = σ(r(x, y_w) - r(x, y_l))
   ```

### Code Implementation Path

```
tokenize_row() → DataCollator → concatenated_forward() → dpo_loss() → compute_loss()
```

### Critical Insights

1. **No reward model needed**: Rewards are implicit in log probability ratios
2. **Single forward pass**: Concatenation trick processes chosen + rejected together
3. **β controls strength**: Higher β = stronger preference signal
4. **Reference model prevents collapse**: KL penalty keeps policy close to reference
5. **Many loss variants**: 15+ variants for different scenarios

---

## Next Steps

- See `RLHF_DPO_Guide.md` for full data flow and lifecycle
- See `annotated_trainers/DPOTrainer_ANNOTATED.md` for line-by-line code walkthrough
- See `annotated_examples/dpo_example_ANNOTATED.md` for complete training script
