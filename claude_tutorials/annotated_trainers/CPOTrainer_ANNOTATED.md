# CPOTrainer: Contrastive Preference Optimization

**Source:** `trl/experimental/cpo/cpo_trainer.py` (900+ lines)
**Purpose:** Preference learning without reference model, with NLL term and multiple loss types

---

## Overview

CPOTrainer implements **Contrastive Preference Optimization**, similar to ORPO but with additional flexibility:
- Multiple loss types (sigmoid, hinge, IPO, SimPO)
- Optional AlphaPO reward transformation
- Label smoothing for noisy preferences

Like ORPO, CPO eliminates the reference model and combines SFT with preference learning.

---

## Core Loss

### Standard CPO (Sigmoid)

```
L_CPO = L_NLL(chosen) + L_preference

where:
L_preference = -log σ(β * (log_p_chosen - log_p_rejected))
```

**With label smoothing:**
```
L = -(1-ε)*log σ(β*Δ) - ε*log σ(-β*Δ)
```

**Code location:** `cpo_trainer.py:707-710`

### Variants

**1. Sigmoid (default/CPO):**
```python
losses = -F.logsigmoid(beta * (chosen_logps - rejected_logps))
```

**2. Hinge:**
```python
losses = torch.relu(1 - beta * (chosen_logps - rejected_logps))
```

**3. IPO (Identity Preference Optimization):**
```python
logits = chosen_logps - rejected_logps
losses = (logits - 1/(2*beta))**2
```

**4. SimPO:**
```python
logits = (chosen_logps - rejected_logps) - gamma/beta
losses = -F.logsigmoid(beta * logits)
```

**Code location:** `cpo_trainer.py:697-719`

---

## AlphaPO Transformation

### Optional Reward Transformation

**Standard:** Use log probabilities directly

**AlphaPO (α ≠ 0):** Transform probabilities before computing loss

```python
r(y) = (1 - p(y)^(-α)) / α

where p(y) = exp(log_p(y))
```

**Effect:**
- **α > 0:** Emphasizes low-probability outputs more
- **α < 0:** Emphasizes high-probability outputs more
- **α = 0:** Standard log probabilities

**Code location:** `cpo_trainer.py:679-691`

---

## Configuration

```python
from trl.experimental import CPOTrainer, CPOConfig

config = CPOConfig(
    # Loss variant
    loss_type="sigmoid",  # "sigmoid", "hinge", "ipo", "simpo"
    beta=0.1,
    label_smoothing=0.0,  # 0-0.1 typical

    # AlphaPO
    alpha=0.0,  # Set non-zero to enable transformation

    # SimPO (only if loss_type="simpo")
    simpo_gamma=0.5,

    # Standard training
    learning_rate=5e-7,
    max_length=1024,
    ...
)
```

---

## Comparison with ORPO

| Feature | CPO | ORPO |
|---------|-----|------|
| **Reference model** | No | No |
| **NLL term** | Yes | Yes |
| **Loss types** | 4 options | 1 (odds ratio) |
| **Reward transform** | Optional (AlphaPO) | No |
| **Label smoothing** | Yes | No |

**Use CPO over ORPO when:**
- Want flexibility in loss function
- Have noisy preference labels (use label smoothing)
- Want to experiment with different objectives

---

## Summary

**CPOTrainer = ORPO + More Loss Options**

**Key formula (sigmoid):**
```
L = L_NLL(chosen) - log σ(β*(log_chosen - log_rejected))
```

**Advantages:**
- No reference model (50% memory savings)
- Multiple loss types for experimentation
- Label smoothing for robustness

**Use cases:**
- Similar to ORPO but want more control
- Noisy preference data
- Research on preference optimization objectives
