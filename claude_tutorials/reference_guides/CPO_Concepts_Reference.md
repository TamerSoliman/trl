# CPO Concepts Reference

## Loss Variants

### Sigmoid (Default CPO)

```
L = -log σ(β * Δ)
where Δ = log_p_chosen - log_p_rejected
```

**Gradient:** -β * σ(-β*Δ) * (1 - σ(-β*Δ))

**Properties:**
- Smooth, differentiable everywhere
- Asymptotically approaches 0 as Δ → ∞
- Similar to DPO but no reference model

### Hinge

```
L = max(0, 1 - β * Δ)
```

**Gradient:**
- -β if Δ < 1/β
- 0 otherwise

**Properties:**
- Non-smooth at Δ = 1/β
- Margin-based (SVM-like)
- Stops learning once margin satisfied

### IPO (Identity Preference Optimization)

```
L = (Δ - 1/(2*β))²
```

**Gradient:** 2(Δ - 1/(2*β))

**Properties:**
- Quadratic loss
- Targets specific margin 1/(2*β)
- More stable than sigmoid for some tasks

### SimPO

```
L = -log σ(β * (Δ - γ/β))
```

**Properties:**
- Sigmoid with length penalty γ
- Encourages longer responses if γ > 0
- From "SimPO: Simple Preference Optimization" paper

---

## AlphaPO Transformation

### Motivation

**Problem:** Log probabilities can be very negative for long sequences

**Solution:** Transform probabilities before computing loss

### Formula

```
r_α(y) = (1 - p(y)^(-α)) / α

where p(y) = exp(log_p(y))
```

### Special Cases

**α → 0:**
```
lim_{α→0} r_α(y) = log p(y)  [standard]
```

**α = 1:**
```
r_1(y) = (1 - 1/p(y)) = (p(y) - 1)/p(y)
```

**α = -1:**
```
r_{-1}(y) = (1 - p(y))/-1 = p(y) - 1
```

### Effect on Low vs High Probabilities

| α | Low p (e.g., 0.01) | High p (e.g., 0.9) |
|---|-------------------|-------------------|
| **-1** | -0.99 (large penalty) | -0.1 (small penalty) |
| **0** | log(0.01) = -4.6 | log(0.9) = -0.11 |
| **1** | 0.99 | 0.11 |

**Interpretation:**
- **α > 0:** Amplifies low probabilities
- **α < 0:** Amplifies high probabilities

---

## Label Smoothing

### Concept

**Standard loss:** Assume labels are 100% correct

**With smoothing (ε):** Account for label noise

```
L = (1-ε) * L(correct) + ε * L(incorrect)
  = (1-ε) * (-log σ(β*Δ)) + ε * (-log σ(-β*Δ))
```

### Effect

**ε = 0:** Trust labels completely
**ε = 0.1:** 10% chance labels are wrong
**ε = 0.5:** Maximum uncertainty (uniform)

**When to use:**
- Crowdsourced labels (noisy)
- AI-generated preferences
- Ambiguous comparisons

---

## Hyperparameter Guide

### beta

**Range:** 0.05 - 0.5
**Default:** 0.1

Same as DPO - controls preference strength

### loss_type

**Default:** "sigmoid"

**When to use each:**
- **sigmoid:** General purpose, smooth
- **hinge:** Want margin-based learning
- **ipo:** More stable, quadratic
- **simpo:** Want length control

### alpha (AlphaPO)

**Default:** 0.0 (disabled)
**Range:** -2.0 to 2.0

**Experimental** - start with 0.0

### label_smoothing

**Default:** 0.0
**Range:** 0.0 - 0.2

**Use 0.05-0.1 for noisy labels**

---

## Comparison Table

| Loss Type | Smooth? | Margin? | Complexity |
|-----------|---------|---------|------------|
| **Sigmoid** | ✓ | ✗ | Low |
| **Hinge** | ✗ | ✓ | Low |
| **IPO** | ✓ | ✓ | Medium |
| **SimPO** | ✓ | ✓ | Medium |

---

## Summary

**CPO = Flexible DPO without Reference Model**

**Key innovations:**
1. Multiple loss functions
2. AlphaPO transformation
3. Label smoothing

**Best for:** Experimentation and research on preference optimization
