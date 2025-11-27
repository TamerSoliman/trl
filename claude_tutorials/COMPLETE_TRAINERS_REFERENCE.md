# Complete TRL Trainers Reference

Comprehensive guide to all TRL alignment trainers with code mappings and examples.

---

## 1. RewardTrainer

**Purpose:** Train reward models from human preference data  
**File:** `trl/trainer/reward_trainer.py` (606 lines)  
**Use case:** First step in RLHF pipeline (before PPO)

### Core Concept
Learns to predict human preferences using Bradley-Terry model:
```
P(y_w > y_l) = σ(r(x, y_w) - r(x, y_l))
Loss = -log σ(r_w - r_l)
```

### Training Script
```python
from trl import RewardTrainer, RewardConfig

config = RewardConfig(
    output_dir="./reward_model",
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    max_length=512,
)

trainer = RewardTrainer(
    model="Qwen/Qwen2-0.5B",
    args=config,
    train_dataset=preference_dataset,  # Must have chosen/rejected columns
)

trainer.train()
```

**Key Feature:** Automatically adds scalar head to model for reward prediction.

---

## 2. KTOTrainer (Kahneman-Tversky Optimization)

**Purpose:** Alignment without paired preferences  
**File:** `trl/trainer/kto_trainer.py` (1,707 lines)  
**Paper:** [KTO](https://arxiv.org/abs/2402.01306)

### Core Concept
Uses **unpaired** binary feedback (thumbs up/down) instead of preferences:
```
L_KTO = E[v(r_desirable)] + E[v(-r_undesirable)]

where v(z) = {  1 - exp(z/β)  if z ≥ 0
             {  exp(z/β) - 1   if z < 0
```

### When to Use
- Have binary labels (good/bad) but not paired comparisons
- Simpler data collection than DPO
- Similar performance to DPO

### Training Script
```python
from trl import KTOTrainer, KTOConfig

config = KTOConfig(
    beta=0.1,  # KTO beta parameter
    desirable_weight=1.0,
    undesirable_weight=1.0,
)

trainer = KTOTrainer(
    model="model_id",
    args=config,
    train_dataset=dataset,  # Needs 'completion' and 'label' (True/False)
)

trainer.train()
```

---

## 3. ORPOTrainer (Odds Ratio Preference Optimization)

**Purpose:** SFT + preference learning in one step  
**File:** `trl/trainer/orpo_trainer.py` (1,058 lines)  
**Paper:** [ORPO](https://arxiv.org/abs/2403.07691)

### Core Concept
Combines SFT loss with odds ratio loss - **no reference model needed**:
```
L_ORPO = L_SFT + λ * L_OR

where L_OR = -log σ(log(odds_chosen/odds_rejected))
      odds = P(y|x) / (1 - P(y|x))
```

### Advantages
- Single-stage training (vs SFT then DPO)
- No reference model (saves memory)
- Competitive with DPO

### Training Script
```python
from trl import ORPOTrainer, ORPOConfig

config = ORPOConfig(
    beta=0.1,  # OR loss weight
    learning_rate=5e-6,
)

trainer = ORPOTrainer(
    model="model_id",
    args=config,
    train_dataset=preference_dataset,
)

trainer.train()
```

---

## 4. RLOOTrainer (REINFORCE Leave-One-Out)

**Purpose:** RL with lower variance than REINFORCE  
**File:** `trl/trainer/rloo_trainer.py` (1,190 lines)  
**Paper:** [RLOO](https://arxiv.org/abs/2402.14740)

### Core Concept
Uses leave-one-out baseline to reduce variance:
```
Advantage_i = Reward_i - mean(Rewards_{j≠i})
```

Similar to GRPO but:
- GRPO: Uses all samples in mean
- RLOO: Excludes current sample from baseline

### When to Use
- Want RL without value function (like GRPO)
- Need lower variance than REINFORCE
- Between GRPO and PPO in complexity

### Training Script
```python
from trl import RLOOTrainer, RLOOConfig

config = RLOOConfig(
    num_samples=4,  # Samples per prompt
    kl_coef=0.05,
)

trainer = RLOOTrainer(
    model="model_id",
    reward_model="reward_model_id",
    args=config,
    train_dataset=dataset,
)

trainer.train()
```

---

## 5. CPOTrainer (Contrastive Preference Optimization)

**Purpose:** Preference learning with contrastive loss  
**File:** `trl/trainer/cpo_trainer.py` (1,198 lines)  
**Paper:** [CPO](https://arxiv.org/abs/2401.08417)

### Core Concept
SimCLR-style contrastive learning for preferences:
```
L_CPO = -log(exp(sim(h_w, h_ref)) / Σ_i exp(sim(h_i, h_ref)))

where h = hidden states, sim = cosine similarity
```

### Advantages
- Can leverage unlabeled data
- More robust to noisy preferences
- Better generalization

### Training Script
```python
from trl import CPOTrainer, CPOConfig

config = CPOConfig(
    beta=0.1,
    loss_type="simpo",  # or "hinge"
)

trainer = CPOTrainer(
    model="model_id",
    args=config,
    train_dataset=preference_dataset,
)

trainer.train()
```

---

## 6. BCOTrainer (Binary Classifier Optimization)

**Purpose:** Use offline RL data for preference learning  
**File:** `trl/trainer/bco_trainer.py` (897 lines)

### Core Concept
Treats preference learning as binary classification:
```
D*(a|s) = π*(a|s) / (π*(a|s) + π_ref(a|s))  # Optimal classifier

L_BCO = -E[log D(a_good|s)] - E[log(1 - D(a_bad|s))]
```

### When to Use
- Have offline RL trajectories
- Want to use binary classifier tricks (data augmentation, etc.)
- Alternative to DPO

### Training Script
```python
from trl import BCOTrainer, BCOConfig

trainer = BCOTrainer(
    model="model_id",
    args=BCOConfig(),
    train_dataset=dataset,
)

trainer.train()
```

---

## 7. OnlineDPOTrainer

**Purpose:** DPO with online generation (best of both)  
**File:** `trl/trainer/online_dpo_trainer.py` (1,679 lines)

### Core Concept
Combines DPO's simplicity with online learning:
1. Generate responses with current policy
2. Score with reward model
3. Create on-the-fly preference pairs
4. Train with DPO loss

### Advantages
- Explores with current policy (not fixed dataset)
- Simpler than PPO (no value function)
- Better than offline DPO (adapts to policy)

### Training Script
```python
from trl import OnlineDPOTrainer, OnlineDPOConfig

config = OnlineDPOConfig(
    beta=0.1,
    num_generations=2,  # Generate pairs online
)

trainer = OnlineDPOTrainer(
    model="model_id",
    reward_model="reward_model_id",
    args=config,
    train_dataset=prompt_dataset,  # Just prompts, not preferences
)

trainer.train()
```

---

## 8. NashMDTrainer (Nash Mirror Descent)

**Purpose:** Game-theoretic approach to alignment  
**File:** `trl/trainer/nash_md_trainer.py` (1,031 lines)  
**Paper:** [Nash Learning](https://arxiv.org/abs/2312.00886)

### Core Concept
Finds Nash equilibrium between:
- Player 1: Policy trying to maximize reward
- Player 2: Adversary trying to find weaknesses

```
π* = argmax_π min_ψ E[r(s,a) - ψ(s,a)]
```

### When to Use
- Want robustness to reward hacking
- Adversarial training mindset
- Research/advanced use case

### Training Script
```python
from trl import NashMDTrainer, NashMDConfig

trainer = NashMDTrainer(
    model="model_id",
    reward_model="reward_model_id",
    args=NashMDConfig(),
    train_dataset=dataset,
)

trainer.train()
```

---

## 9. GKDTrainer (Generalized Knowledge Distillation)

**Purpose:** Distill capabilities from larger model  
**File:** `trl/trainer/gkd_trainer.py` (575 lines)  
**Paper:** [GKD](https://arxiv.org/abs/2306.13649)

### Core Concept
On-policy distillation with KL divergence:
```
L_GKD = KL(student || teacher on student-generated data)
```

Not alignment per se, but useful for:
- Compressing aligned models
- Transferring capabilities

### Training Script
```python
from trl import GKDTrainer, GKDConfig

config = GKDConfig(
    temperature=2.0,  # Soften distributions
)

trainer = GKDTrainer(
    model="small_model",
    teacher_model="large_aligned_model",
    args=config,
    train_dataset=dataset,
)

trainer.train()
```

---

## 10. XPOTrainer (eXponential Preference Optimization)

**Purpose:** Alternative to DPO with exponential weighting  
**File:** `trl/trainer/xpo_trainer.py` (1,293 lines)

### Core Concept
Exponentially weights preference violations:
```
L_XPO = E[exp(β * (r_rejected - r_chosen))]
```

More aggressive than DPO's log-sigmoid.

### Training Script
```python
from trl import XPOTrainer, XPOConfig

trainer = XPOTrainer(
    model="model_id",
    args=XPOConfig(beta=0.1),
    train_dataset=preference_dataset,
)

trainer.train()
```

---

## 11. PRMTrainer (Process Reward Model)

**Purpose:** Train step-by-step reward models  
**File:** `trl/trainer/prm_trainer.py` (445 lines)  
**Use case:** Math/reasoning where each step can be evaluated

### Core Concept
Unlike outcome reward models, PRM scores each intermediate step:
```
Input: "Step 1: ... [+] Step 2: ... [-] Step 3: ... [+]"
Output: [score_1, score_2, score_3]
```

### When to Use
- Multi-step reasoning (math, code)
- Have step-level annotations
- Want to identify where model goes wrong

### Training Script
```python
from trl import PRMTrainer, PRMConfig

trainer = PRMTrainer(
    model="model_id",
    args=PRMConfig(),
    train_dataset=step_annotated_dataset,
    # Dataset must have step-by-step labels
)

trainer.train()
```

---

## Comparison Table

| Trainer | Type | Models | Online | Complexity | Best For |
|---------|------|--------|--------|------------|----------|
| **SFT** | Supervised | 1 | No | Low | Foundation |
| **RewardTrainer** | Supervised | 1 | No | Low | Train RM |
| **DPO** | Preference | 2 | No | Low | Fast alignment |
| **KTO** | Preference | 2 | No | Low | Binary feedback |
| **ORPO** | Preference | 1 | No | Low | One-stage |
| **CPO** | Preference | 2 | No | Medium | Robust |
| **BCO** | Preference | 2 | No | Medium | Offline RL |
| **XPO** | Preference | 2 | No | Low | DPO variant |
| **PPO** | RL | 4 | Yes | High | Max quality |
| **GRPO** | RL | 2-3 | Yes | Medium | Simplified RL |
| **RLOO** | RL | 2-3 | Yes | Medium | Variance reduction |
| **OnlineDPO** | RL+Pref | 2-3 | Yes | Medium | Hybrid |
| **NashMD** | Game Theory | 3 | Yes | High | Robustness |
| **GKD** | Distillation | 2 | Yes | Medium | Compression |
| **PRM** | Supervised | 1 | No | Low | Step rewards |

---

## Decision Tree

```
Start here → Do you have...

├─ Raw text only? → SFT
│
├─ Preference pairs (A > B)?
│  ├─ Want simplest/fastest? → DPO or ORPO
│  ├─ No reference model? → ORPO
│  ├─ Want most robust? → CPO
│  └─ Research/experimental? → XPO, BCO
│
├─ Binary feedback (good/bad)?
│  └─ → KTO
│
├─ Reward model?
│  ├─ Want max quality? → PPO
│  ├─ Want simpler? → GRPO or RLOO
│  ├─ Want hybrid? → OnlineDPO
│  └─ Want robust? → NashMD
│
├─ Large teacher model?
│  └─ → GKD
│
└─ Step-by-step data?
   └─ → PRM
```

---

## Memory Requirements (7B Model)

| Trainer | Models Loaded | Approx Memory | A100 40GB | RTX 4090 24GB |
|---------|---------------|---------------|-----------|---------------|
| SFT | 1 | 14 GB | ✓ Full | ✓ LoRA |
| DPO/KTO/ORPO | 1-2 | 14-28 GB | ✓ Full | ✓ LoRA |
| PPO | 4 | 56 GB | ✓ LoRA+quant | ✗ |
| GRPO/RLOO | 2-3 | 28-42 GB | ✓ LoRA | ✓ LoRA+4bit |
| OnlineDPO | 2-3 | 28-42 GB | ✓ LoRA | ✓ LoRA+4bit |

*With LoRA: Trainable params ~0.1-1% of full model  
*With 4-bit: Model size reduced by ~75%

---

## Common Patterns

### All Trainers Support
```python
trainer = SomeTrainer(
    model="model_id",  # or PreTrainedModel instance
    args=SomeConfig(
        output_dir="./output",
        learning_rate=1e-6,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        bf16=True,
        gradient_checkpointing=True,
    ),
    train_dataset=dataset,
    peft_config=LoraConfig(...),  # Optional LoRA
)
```

### Saving and Loading
```python
# Save
trainer.save_model("./final_model")

# Load for inference
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("./final_model")
```

### Distributed Training
```bash
# Automatically uses all GPUs
accelerate launch script.py

# Or with torchrun
torchrun --nproc_per_node=4 script.py
```

---

## Tips

1. **Start with SFT** - Always fine-tune base model first
2. **DPO for speed** - Fastest preference alignment
3. **PPO for quality** - When you need the best
4. **GRPO for math** - Great for verifiable tasks
5. **Try ORPO** - Single-stage is convenient
6. **Use LoRA** - Makes everything fit in memory
7. **Monitor KL** - Should stay < 0.1 for stability
8. **Validate early** - Check outputs every 100 steps

---

## Next Steps

1. See `annotated_trainers/` for line-by-line code walkthroughs
2. See `reference_guides/` for mathematical foundations  
3. See `annotated_examples/` for production scripts
4. Check `00_TRAINER_CATALOG.md` for detailed comparison

For questions or issues, refer to [TRL documentation](https://huggingface.co/docs/trl).
