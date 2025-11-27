# TRL Trainer Implementation Deep Dive

**Created**: 2025-11-18
**Last Updated**: 2025-11-27
**Purpose**: Comprehensive analysis and annotation of TRL (Transformer Reinforcement Learning) library trainers for alignment techniques

---

## 📚 What's in This Directory?

This directory contains heavily annotated implementations, mathematical references, and practical guides for understanding and using the TRL library's alignment trainers. **9 trainers fully documented** with complete 3-file sets each, covering **95%+ of real-world use cases**.

### ✅ Fully Documented Trainers (9)
- **SFTTrainer** - Supervised fine-tuning (foundation)
- **DPOTrainer** - Direct preference optimization
- **PPOTrainer** - Proximal policy optimization (classic RLHF)
- **GRPOTrainer** - Group relative policy optimization
- **RewardTrainer** - Reward model training
- **KTOTrainer** - Kahneman-Tversky optimization (binary preferences)
- **ORPOTrainer** - Odds ratio preference optimization (no reference model)
- **RLOOTrainer** - REINFORCE leave-one-out (RL with variance reduction)
- **CPOTrainer** - Contrastive preference optimization (flexible loss variants)

📊 **Total**: 29 files, ~13,400 lines of documentation

📋 **Status**: See [STATUS.md](STATUS.md) for detailed completion tracking

---

## 🚀 Quick Start - Choose Your Path

### Path 1: Complete Beginner (Never trained an LLM)
**Time: ~2 hours**

1. **Start**: [00_TRAINER_CATALOG.md](00_TRAINER_CATALOG.md) (10 min)
   - Understand what each trainer does
   - See comparison table

2. **Learn SFT**: [SFTTrainer_ANNOTATED.md](annotated_trainers/SFTTrainer_ANNOTATED.md) (30 min)
   - Foundation of all alignment
   - How supervised fine-tuning works

3. **Try it**: [sft_training_ANNOTATED.py](annotated_examples/sft_training_ANNOTATED.py) (Run it!)
   - Complete working example
   - Start with small model

### Path 2: Have Preference Data (Most Common)
**Time: ~3 hours**

1. **Understand**: [RLHF_DPO_Guide.md](reference_guides/RLHF_DPO_Guide.md) (20 min)
2. **Math**: [DPO_Loss_Reference.md](reference_guides/DPO_Loss_Reference.md) (30 min)
3. **Code**: [DPOTrainer_ANNOTATED.md](annotated_trainers/DPOTrainer_ANNOTATED.md) (45 min)
4. **Run**: [dpo_training_ANNOTATED.py](annotated_examples/dpo_training_ANNOTATED.py)

### Path 3: Want RL with Rewards
**Time: ~2-3 hours**

Pick based on complexity:
- **Classic RL**: [PPOTrainer_ANNOTATED.md](annotated_trainers/PPOTrainer_ANNOTATED.md)
- **Simpler RL**: [RLOOTrainer_ANNOTATED.md](annotated_trainers/RLOOTrainer_ANNOTATED.md)
- **No Reference**: [GRPOTrainer_ANNOTATED.md](annotated_trainers/GRPOTrainer_ANNOTATED.md)

### Path 4: Just Want to Run Something Now
**Time: 10 minutes**

```bash
# Install
pip install trl transformers datasets

# Run SFT (simplest)
python annotated_examples/sft_training_ANNOTATED.py

# Or DPO (if you have preference data)
python annotated_examples/dpo_training_ANNOTATED.py
```

All scripts have sensible defaults and work out-of-the-box!

---

## 🎯 Which Trainer Should I Use?

```
Do you have instruction data?
├─ YES → Start with SFTTrainer
└─ NO → Get some first!

After SFT, do you want to align with preferences?
├─ Have pairs (chosen vs rejected)?
│  ├─ Want simplest → DPOTrainer ⭐ Most Popular
│  ├─ Want to save memory → ORPOTrainer
│  └─ Want flexible losses → CPOTrainer
│
├─ Have binary feedback (👍/👎)?
│  └─ Use KTOTrainer
│
└─ Have reward function/model?
   ├─ Want best results → PPOTrainer
   ├─ Want simpler → RLOOTrainer
   └─ Want memory efficient → GRPOTrainer
```

**Still unsure?** See [00_TRAINER_CATALOG.md](00_TRAINER_CATALOG.md) for detailed comparison.

---

## 📁 What's in Each File Type?

### `annotated_trainers/*.md` - Code Walkthroughs
**What**: Line-by-line explanation of trainer implementations
**When**: You want to understand HOW it works internally
**Example**: See how DPO computes loss with 15+ variants

### `reference_guides/*.md` - Math & Concepts
**What**: Mathematical derivations and theoretical foundations
**When**: You want to understand WHY it works
**Example**: Complete DPO derivation from Bradley-Terry model

### `annotated_examples/*.py` - Ready-to-Run Scripts
**What**: Production training code with extensive comments
**When**: You want to RUN training now
**Example**: Full DPO training with hyperparameter tuning

---

## 📊 Trainer Comparison

| Trainer | Data | Ref Model | Complexity | Memory | When to Use |
|---------|------|-----------|------------|--------|-------------|
| **SFT** | Instructions | ❌ | ⭐ | ⭐ | Always start here |
| **DPO** | Pairs | ✅ | ⭐⭐ | ⭐⭐⭐ | Most popular |
| **ORPO** | Pairs | ❌ | ⭐⭐ | ⭐⭐ | Save memory |
| **KTO** | Binary | ✅ | ⭐⭐ | ⭐⭐⭐ | Thumbs up/down data |
| **CPO** | Pairs | ❌ | ⭐⭐ | ⭐⭐ | Experiment with losses |
| **PPO** | Reward | ✅+Value | ⭐⭐⭐ | ⭐⭐⭐⭐ | Best quality |
| **RLOO** | Reward | Optional | ⭐⭐ | ⭐⭐ | Simpler than PPO |
| **GRPO** | Reward | ❌ | ⭐⭐ | ⭐ | Memory efficient RL |

⭐ = Low, ⭐⭐⭐⭐ = High

---

## 🔧 Quick Examples

### Train with DPO (Most Common)
```python
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model (already SFT'd)
model = AutoModelForCausalLM.from_pretrained("my-sft-model")
tokenizer = AutoTokenizer.from_pretrained("my-sft-model")

# Load preference data
dataset = load_dataset("trl-lib/ultrafeedback_binarized")

# Configure
config = DPOConfig(
    beta=0.1,
    learning_rate=5e-7,
    per_device_train_batch_size=4,
)

# Train
trainer = DPOTrainer(
    model=model,
    args=config,
    train_dataset=dataset["train"],
    processing_class=tokenizer,
)
trainer.train()
```

### Memory-Optimized Training (24GB GPU)
```python
config = DPOConfig(
    per_device_train_batch_size=1,       # Minimum
    gradient_accumulation_steps=16,      # Effective batch=16
    gradient_checkpointing=True,         # Save ~40% memory
    precompute_ref_log_probs=True,       # No ref model in memory
    bf16=True,                           # Mixed precision
)

# Add LoRA for even more savings
from peft import LoraConfig
peft_config = LoraConfig(r=16, lora_alpha=32)
trainer = DPOTrainer(..., peft_config=peft_config)
```

**See**: Each `*_training_ANNOTATED.py` script for full examples

---

## 📈 Monitor Your Training

### Key Metrics to Watch

**DPO/ORPO/KTO/CPO:**
- ✅ `loss` decreasing
- ✅ `rewards/margins` > 0 and increasing
- ✅ `rewards/accuracies` > 0.7
- ⚠️ If margins negative → increase beta

**PPO/RLOO/GRPO:**
- ✅ `rewards/mean` increasing
- ✅ `objective/kl` < 0.5 (not exploding)
- ✅ `entropy` not collapsing
- ⚠️ If KL high → increase beta or reduce LR

---

## 🐛 Common Issues & Solutions

### Out of Memory
```python
# Try these in order:
1. batch_size = 1
2. gradient_checkpointing = True
3. precompute_ref_log_probs = True  # DPO/KTO
4. Use LoRA/PEFT
5. max_length = 512
```

### Loss Not Decreasing
```python
# Try:
1. Increase learning_rate to 1e-6
2. Increase beta to 0.2-0.3
3. Check data quality
4. Try different loss_type
```

### Model Outputs Gibberish
```python
# Probably:
1. Learning rate too high → reduce to 1e-7
2. Too many epochs → reduce to 1
3. Beta too high → try 0.1
```

**Full troubleshooting**: See each training script

---

## 📚 Additional Resources

### Official Docs
- **TRL Docs**: https://huggingface.co/docs/trl
- **GitHub**: https://github.com/huggingface/trl

### Papers
- **DPO**: [Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)
- **PPO**: [Schulman et al., 2017](https://arxiv.org/abs/1707.06347)
- **KTO**: [Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)
- **ORPO**: [Hong et al., 2024](https://arxiv.org/abs/2403.07691)
- **RLOO**: [Ahmadian et al., 2024](https://arxiv.org/abs/2402.14740)

---

## 🙏 Acknowledgments

- **TRL Team** at Hugging Face
- **Original paper authors**
- **Open source community**

---

**Last Updated**: 2025-11-27
**Status**: 9/15 trainers (95%+ use cases covered)
**License**: Apache 2.0 (same as TRL)

**Ready to start?** Pick a path above and dive in! 🚀
