# TRL Trainer Catalog: Complete Overview of All Alignment Techniques

This document provides a comprehensive catalog of all trainer classes available in the TRL (Transformer Reinforcement Learning) library, organized by their purpose and relationship.

## Quick Reference Table

| Trainer | Type | Paper | Primary Use Case | Key Innovation |
|---------|------|-------|------------------|----------------|
| **SFTTrainer** | Supervised | - | Supervised fine-tuning on instruction data | Foundation for all alignment |
| **DPOTrainer** | Preference | [DPO Paper](https://arxiv.org/abs/2305.18290) | Direct preference optimization without RL | Reward-model-free alignment |
| **PPOTrainer** | RL | [PPO Paper](https://arxiv.org/abs/1707.06347) | Online RL with reward model feedback | Classic RLHF approach |
| **RewardTrainer** | Reward Model | Bradley-Terry Model | Train reward models from preferences | ORM (Outcome Reward Model) |
| **PRMTrainer** | Reward Model | - | Train process reward models | PRM (Process Reward Model) |
| **KTOTrainer** | Preference | [KTO Paper](https://arxiv.org/abs/2402.01306) | Kahneman-Tversky Optimization | Unpaired preference data |
| **ORPOTrainer** | Preference | [ORPO Paper](https://arxiv.org/abs/2403.07691) | Odds Ratio Preference Optimization | SFT + preference in one stage |
| **CPOTrainer** | Preference | [CPO Paper](https://arxiv.org/abs/2401.08417) | Contrastive Preference Optimization | Contrastive learning approach |
| **BCOTrainer** | Preference | [BCO Paper](https://arxiv.org/abs/2404.04656) | Binary Classifier Optimization | Self-play approach |
| **RLOOTrainer** | RL | [RLOO Paper](https://arxiv.org/abs/2402.14740) | RL with leave-one-out baseline | Variance-reduced REINFORCE |
| **GRPOTrainer** | RL | - | Group Relative Policy Optimization | Group-based advantages |
| **OnlineDPOTrainer** | Online Preference | - | DPO with online data generation | Dynamic preference generation |
| **NashMDTrainer** | Game Theory | [Nash-MD Paper](https://arxiv.org/abs/2309.10049) | Nash equilibrium via mirror descent | Game-theoretic alignment |
| **GKDTrainer** | Distillation | [GKD Paper](https://arxiv.org/abs/2306.13649) | Generalized knowledge distillation | Distillation-based alignment |
| **XPOTrainer** | Preference | - | Cross preference optimization | Cross-entropy based |

---

## Detailed Breakdown by Category

### 1. Supervised Fine-Tuning (Foundation)

#### **SFTTrainer**
- **File**: `trl/trainer/sft_trainer.py`
- **Config**: `SFTConfig`
- **Purpose**: Supervised fine-tuning on instruction/conversational data
- **Key Features**:
  - Supports both language modeling and prompt-completion formats
  - Packing support for efficient training
  - Padding-free training with Flash Attention
  - Vision-language model support
  - Completion-only loss (mask prompt tokens)
  - Assistant-only loss for conversational data
  - DFT loss variant for improved generalization
- **Data Format**:
  - Language modeling: `{"text": "..."}` or `{"messages": [...]}`
  - Prompt-completion: `{"prompt": "...", "completion": "..."}`
- **When to Use**: Always the first step before any alignment technique

---

### 2. Preference-Based Alignment (Offline)

These trainers learn from static datasets of preference pairs.

#### **DPOTrainer** (Direct Preference Optimization)
- **File**: `trl/trainer/dpo_trainer.py`
- **Config**: `DPOConfig`
- **Paper**: [Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)
- **Core Idea**: Optimize policy directly using preference data, without training a separate reward model
- **Loss Function** (Sigmoid variant):
  ```
  L_DPO = -log σ(β * (log π_θ(y_w|x) - log π_θ(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x)))
  ```
  Where:
  - `y_w` = chosen/winning response
  - `y_l` = rejected/losing response
  - `π_θ` = policy model
  - `π_ref` = reference model
  - `β` = temperature parameter (typically 0.1-0.5)
  - `σ` = sigmoid function

- **Key Features**:
  - 15+ loss variants (sigmoid, IPO, hinge, BCO, SPP0, APO, etc.)
  - Reference model can be:
    - Separate model (traditional)
    - PEFT adapter with base as reference
    - Precomputed log probabilities
  - Label smoothing for robustness
  - F-divergence variants (KL, JS, Alpha-divergence)
  - Liger kernel support for memory efficiency

- **Data Format**:
  ```python
  {
    "prompt": "What is the capital of France?",
    "chosen": "The capital of France is Paris.",
    "rejected": "I don't know."
  }
  ```

- **When to Use**:
  - You have preference pairs (chosen/rejected)
  - Want simpler training than PPO
  - Don't want to train a separate reward model

---

#### **KTOTrainer** (Kahneman-Tversky Optimization)
- **File**: `trl/trainer/kto_trainer.py`
- **Config**: `KTOConfig`
- **Paper**: [Ethayarajh et al., 2024](https://arxiv.org/abs/2402.01306)
- **Core Idea**: Use prospect theory from behavioral economics; handle unpaired preferences
- **Key Innovation**: Works with binary feedback (good/bad) without explicit pairs
- **Data Format**:
  ```python
  {
    "prompt": "...",
    "completion": "...",
    "label": True  # or False
  }
  ```
- **When to Use**:
  - You have thumbs up/down data
  - Don't have explicit chosen/rejected pairs
  - Want to leverage prospect theory insights

---

#### **ORPOTrainer** (Odds Ratio Preference Optimization)
- **File**: `trl/trainer/orpo_trainer.py`
- **Config**: `ORPOConfig`
- **Paper**: [Hong et al., 2024](https://arxiv.org/abs/2403.07691)
- **Core Idea**: Combine SFT and preference learning in a single stage
- **Loss Function**:
  ```
  L_ORPO = L_SFT + λ * L_OR
  ```
  Where `L_OR` uses odds ratios instead of log probabilities

- **Key Innovation**: No need for separate SFT stage or reference model
- **When to Use**:
  - Want to save compute (single-stage training)
  - Have preference data from the start
  - Want simpler pipeline

---

#### **CPOTrainer** (Contrastive Preference Optimization)
- **File**: `trl/trainer/cpo_trainer.py`
- **Config**: `CPOConfig`
- **Paper**: [Xu et al., 2024](https://arxiv.org/abs/2401.08417)
- **Core Idea**: Use contrastive learning principles for preference optimization
- **When to Use**: When you want contrastive learning benefits in alignment

---

#### **BCOTrainer** (Binary Classifier Optimization)
- **File**: `trl/trainer/bco_trainer.py`
- **Config**: `BCOConfig`
- **Paper**: [Azar et al., 2024](https://arxiv.org/abs/2404.04656)
- **Core Idea**: Self-play approach where model generates its own negatives
- **When to Use**: When you want self-play dynamics in alignment

---

### 3. Online Preference Learning

#### **OnlineDPOTrainer**
- **File**: `trl/trainer/online_dpo_trainer.py`
- **Config**: `OnlineDPOConfig`
- **Core Idea**: Generate preference pairs on-the-fly during training
- **Key Features**:
  - Uses a judge model to score completions
  - Dynamically generates chosen/rejected pairs
  - Combines benefits of online RL with simplicity of DPO
- **Workflow**:
  1. Generate multiple completions for each prompt
  2. Score them with a judge model
  3. Create preference pairs from scores
  4. Update policy with DPO
- **When to Use**:
  - Want benefits of online data
  - Have access to a judge model (LLM-as-judge or reward model)
  - Want more sample efficient than offline DPO

---

### 4. Reinforcement Learning Approaches

#### **PPOTrainer** (Proximal Policy Optimization)
- **File**: `trl/trainer/ppo_trainer.py` (experimental)
- **Config**: `PPOConfig`
- **Paper**: [Schulman et al., 2017](https://arxiv.org/abs/1707.06347)
- **Core Idea**: Classic RLHF with reward model feedback
- **Components**:
  - Policy model (π_θ): The model being trained
  - Reference model (π_ref): KL penalty reference
  - Value model (V): Estimates value of states
  - Reward model (R): Scores completions

- **Loss Function**:
  ```
  L_PPO = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)] - β * KL[π_θ || π_ref]
  ```
  Where:
  - `r_t` = π_θ / π_old (probability ratio)
  - `A_t` = advantage estimate
  - `ε` = clip parameter (typically 0.2)
  - `β` = KL penalty coefficient

- **Training Loop**:
  1. **Generation**: Sample completions from current policy
  2. **Scoring**: Get rewards from reward model
  3. **Advantage**: Compute advantages using value model
  4. **Optimization**: Update policy with clipped objective

- **When to Use**:
  - Need online learning
  - Have a good reward model
  - Want maximum flexibility
  - Can afford computational cost (4 models)

---

#### **RLOOTrainer** (REINFORCE Leave-One-Out)
- **File**: `trl/trainer/rloo_trainer.py`
- **Config**: `RLOOConfig`
- **Paper**: [Ahmadian et al., 2024](https://arxiv.org/abs/2402.14740)
- **Core Idea**: Variance-reduced REINFORCE using leave-one-out baseline
- **Key Innovation**:
  - Simpler than PPO (no value model needed)
  - Lower variance than vanilla REINFORCE
  - Baseline computed from other samples in batch

- **When to Use**:
  - Want online RL benefits
  - Don't want complexity of PPO
  - Have limited compute vs full PPO

---

#### **GRPOTrainer** (Group Relative Policy Optimization)
- **File**: `trl/trainer/grpo_trainer.py`
- **Config**: `GRPOConfig`
- **Core Idea**: Compute advantages relative to groups
- **Key Innovation**: Group-based advantage estimation for better credit assignment
- **When to Use**: When group-relative rewards make sense for your task

---

### 5. Reward Model Training

#### **RewardTrainer** (Outcome Reward Model)
- **File**: `trl/trainer/reward_trainer.py`
- **Config**: `RewardConfig`
- **Purpose**: Train reward models from human preference data
- **Core Idea**: Bradley-Terry model for pairwise preferences
- **Loss Function**:
  ```
  L_RM = -log σ(r_θ(x, y_w) - r_θ(x, y_l))
  ```
  Where:
  - `r_θ` = reward model
  - `y_w` = chosen response
  - `y_l` = rejected response

- **Architecture**: Adds a scalar head to a language model
- **Data Format**: Same as DPO (prompt, chosen, rejected)
- **When to Use**:
  - Need reward model for PPO or RLOO
  - Want to train outcome-based rewards
  - Building RLHF pipeline

---

#### **PRMTrainer** (Process Reward Model)
- **File**: `trl/trainer/prm_trainer.py`
- **Config**: `PRMConfig`
- **Purpose**: Train reward models that score intermediate steps
- **Key Difference from ORM**: Provides rewards for each reasoning step, not just final output
- **Use Case**: Reasoning tasks where you want to reward correct intermediate steps
- **Data Format**: Requires step-level annotations
- **When to Use**:
  - Training on reasoning tasks (math, code)
  - Have step-level feedback
  - Want to guide reasoning process

---

### 6. Advanced/Experimental Techniques

#### **NashMDTrainer** (Nash Mirror Descent)
- **File**: `trl/trainer/nash_md_trainer.py`
- **Config**: `NashMDConfig`
- **Paper**: [Munos et al., 2023](https://arxiv.org/abs/2309.10049)
- **Core Idea**: Find Nash equilibrium between policy and adversarial player
- **When to Use**: Interested in game-theoretic formulation of alignment

---

#### **GKDTrainer** (Generalized Knowledge Distillation)
- **File**: `trl/trainer/gkd_trainer.py`
- **Config**: `GKDConfig`
- **Paper**: [Agarwal et al., 2023](https://arxiv.org/abs/2306.13649)
- **Core Idea**: Distill knowledge from teacher model using on-policy data
- **When to Use**:
  - Distilling a larger aligned model
  - Want on-policy distillation benefits

---

#### **XPOTrainer** (Cross Preference Optimization)
- **File**: `trl/trainer/xpo_trainer.py`
- **Config**: `XPOConfig`
- **Core Idea**: Cross-entropy based preference optimization
- **When to Use**: Experimental cross-entropy variant

---

## Training Pipeline Recommendations

### Basic Alignment Pipeline
```
1. SFTTrainer: Get base instruction-following capability
2. DPOTrainer: Align to preferences
```

### Full RLHF Pipeline
```
1. SFTTrainer: Initial instruction tuning
2. RewardTrainer: Train reward model from preferences
3. PPOTrainer: Optimize policy with RL
```

### Modern Simplified Pipeline
```
1. SFTTrainer: Initial training
2. OnlineDPOTrainer: Online alignment with LLM-as-judge
```

### Single-Stage Pipeline
```
ORPOTrainer: Combined SFT + preference learning
```

---

## File Locations Quick Reference

### Main Trainers (Production-Ready)
- `trl/trainer/sft_trainer.py` - Supervised Fine-Tuning
- `trl/trainer/dpo_trainer.py` - Direct Preference Optimization
- `trl/trainer/kto_trainer.py` - Kahneman-Tversky Optimization
- `trl/trainer/orpo_trainer.py` - Odds Ratio Preference Optimization
- `trl/trainer/reward_trainer.py` - Outcome Reward Model Training
- `trl/trainer/prm_trainer.py` - Process Reward Model Training
- `trl/trainer/rloo_trainer.py` - REINFORCE Leave-One-Out
- `trl/trainer/grpo_trainer.py` - Group Relative Policy Optimization
- `trl/trainer/online_dpo_trainer.py` - Online DPO

### Experimental Trainers
- `trl/experimental/ppo/ppo_trainer.py` - PPO (being refactored)
- `trl/experimental/cpo/cpo_trainer.py` - CPO
- `trl/experimental/bco/bco_trainer.py` - BCO
- `trl/experimental/nash_md/nash_md_trainer.py` - Nash-MD
- `trl/experimental/gkd/gkd_trainer.py` - GKD
- `trl/experimental/xpo/xpo_trainer.py` - XPO

### Example Scripts
- `trl/scripts/sft.py` - SFT example
- `trl/scripts/dpo.py` - DPO example
- `examples/scripts/ppo/ppo.py` - PPO example
- `examples/scripts/kto.py` - KTO example
- `examples/scripts/orpo.py` - ORPO example
- `examples/scripts/reward_modeling.py` - Reward model example
- `examples/scripts/online_dpo.py` - Online DPO example

---

## Common Configuration Parameters

### Shared Across Most Trainers
- `model_name_or_path`: Model to train
- `learning_rate`: Learning rate (varies by method)
- `per_device_train_batch_size`: Batch size per device
- `gradient_accumulation_steps`: Gradient accumulation
- `num_train_epochs` or `max_steps`: Training duration
- `output_dir`: Where to save model

### DPO/Preference-Specific
- `beta`: Temperature parameter (0.1-0.5)
- `loss_type`: Which loss variant to use
- `max_prompt_length`: Max tokens for prompt
- `max_completion_length`: Max tokens for completion
- `label_smoothing`: Regularization (0.0-0.1)

### PPO-Specific
- `reward_model_path`: Path to reward model
- `sft_model_path`: Path to SFT model (for initialization)
- `kl_penalty`: "kl" or "abs" or "mse"
- `init_kl_coef`: Initial KL coefficient
- `vf_coef`: Value function loss coefficient

### SFT-Specific
- `packing`: Pack multiple sequences into one
- `max_seq_length`: Maximum sequence length
- `dataset_text_field`: Field containing text data
- `completion_only_loss`: Only compute loss on completion

---

## Next Steps

For detailed implementation analysis:
1. See annotated trainer files in `annotated_trainers/`
2. See annotated example scripts in `annotated_examples/`
3. See concept guides in `reference_guides/`

Key guides:
- `DPO_Loss_Reference.md` - Deep dive into DPO loss calculation
- `RLHF_DPO_Guide.md` - Complete data flow and lifecycle
