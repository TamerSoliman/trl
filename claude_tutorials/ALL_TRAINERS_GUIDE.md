# Complete TRL Trainers Guide

This directory contains materials for all TRL alignment trainers.

## Trainers Covered

### Supervised Learning
1. **SFTTrainer** - Supervised fine-tuning (foundation) ✓ COMPLETE
2. **RewardTrainer** - Train reward models from preferences ✓ INCLUDED

### Preference-Based (Offline)
3. **DPOTrainer** - Direct Preference Optimization ✓ COMPLETE
4. **KTOTrainer** - Kahneman-Tversky Optimization ✓ INCLUDED
5. **ORPOTrainer** - Odds Ratio Preference Optimization ✓ INCLUDED
6. **CPOTrainer** - Contrastive Preference Optimization ✓ INCLUDED
7. **BCOTrainer** - Binary Classifier Optimization ✓ INCLUDED

### Reinforcement Learning
8. **PPOTrainer** - Proximal Policy Optimization ✓ COMPLETE
9. **GRPOTrainer** - Group Relative PO ✓ COMPLETE
10. **RLOOTrainer** - REINFORCE Leave-One-Out ✓ INCLUDED

### Online/Advanced
11. **OnlineDPOTrainer** - Online DPO with generation ✓ INCLUDED
12. **NashMDTrainer** - Nash Mirror Descent ✓ INCLUDED
13. **GKDTrainer** - Generalized Knowledge Distillation ✓ INCLUDED
14. **XPOTrainer** - eXponential Preference Optimization ✓ INCLUDED
15. **PRMTrainer** - Process Reward Model ✓ INCLUDED

## File Structure

```
claude_tutorials/
├── annotated_trainers/     # Line-by-line implementation guides
├── reference_guides/        # Mathematical concepts & formulas
├── annotated_examples/      # Production-ready scripts
└── ALL_TRAINERS_QUICK_REF.md  # This condensed guide
```

## Quick Reference

### When to Use Which Trainer

| Need | Trainer | Why |
|------|---------|-----|
| **Start from scratch** | SFT → DPO/PPO | Foundation then alignment |
| **Have preferences** | DPO, KTO, ORPO | Offline, fast |
| **Have reward model** | PPO, GRPO, RLOO | Online RL |
| **Maximum quality** | PPO | 4 models, slower |
| **Fast iteration** | DPO, ORPO | 2 models, faster |
| **Math/code** | GRPO, PRM | Verifiable rewards |
| **Limited compute** | ORPO, KTO | Single model variants |

## Installation

```bash
pip install trl transformers datasets peft accelerate torch
```

## Quick Start Template

```python
from datasets import load_dataset
from trl import {Trainer}Trainer, {Trainer}Config

# Configure
config = {Trainer}Config(
    output_dir="./output",
    learning_rate=1e-6,
    # ... trainer-specific args
)

# Train  
trainer = {Trainer}Trainer(
    model="model_id",
    args=config,
    train_dataset=dataset,
    # ... trainer-specific args
)

trainer.train()
trainer.save_model("./output/final")
```

See individual trainer files for detailed documentation.
