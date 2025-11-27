# PPOTrainer: Complete Implementation Deep Dive

**Source:** `trl/experimental/ppo/ppo_trainer.py` (836 lines)
**Purpose:** Proximal Policy Optimization - Classic RLHF with actor-critic architecture

**Paper:** [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (Schulman et al., 2017)

---

## Table of Contents

1. [Overview](#overview)
2. [The Four Models](#the-four-models)
3. [Initialization](#initialization)
4. [Training Loop](#training-loop)
5. [PPO Loss Functions](#ppo-loss-functions)
6. [Configuration](#configuration)

---

## Overview

### What is PPO?

PPO is a **reinforcement learning** algorithm for fine-tuning language models using human feedback.

**The RLHF Pipeline:**
```
1. Supervised Fine-Tuning (SFT)
   Pretrained model → Instruction-following model

2. Reward Model Training
   Collect human preferences → Train reward model

3. PPO Fine-Tuning ← WE ARE HERE
   Use reward model to optimize policy with RL
```

**Why PPO?**
- More stable than vanilla policy gradient (REINFORCE)
- Prevents catastrophic policy updates via clipping
- Sample-efficient (reuses data for multiple epochs)
- Widely used in production (OpenAI, Anthropic, DeepMind)

### PPO vs DPO

| Aspect | PPO | DPO |
|--------|-----|-----|
| **Models** | 4 (policy, ref, value, reward) | 2 (policy, ref) |
| **Training** | Online RL (generate + train) | Offline (fixed dataset) |
| **Complexity** | High | Low |
| **Memory** | 4x model size | 2x model size |
| **Speed** | Slow (generation bottleneck) | Fast |
| **Quality** | Highest (when tuned well) | Very good |
| **Use case** | Production RLHF at scale | Research, fast iteration |

**When to use PPO:**
- You have a reward model (not just preference data)
- You need online learning (model improves from its own outputs)
- You have compute budget (4 models + generation)
- You're optimizing for maximum quality

---

## The Four Models

PPO requires **four neural networks**:

### 1. Policy Model (π_θ)

**What:** The model being trained. Generates responses to prompts.

**Role:** The "actor" in actor-critic RL.

**Updates:** YES (trained with PPO loss)

**Code:** `self.policy_model`

**Example:**
```python
policy_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B-SFT")
# Started from SFT model, will be further optimized
```

### 2. Reference Model (π_ref)

**What:** Frozen copy of the initial policy (usually the SFT model).

**Role:** Prevents policy from drifting too far from original behavior.

**Updates:** NO (frozen)

**Code:** `self.ref_model`

**Why needed:** Without it, the policy could exploit the reward model by generating nonsense that happens to score high.

**KL penalty:**
```python
kl = log(π_θ(y|x)) - log(π_ref(y|x))
reward = r(x, y) - β * kl
```

The β * kl term penalizes deviation from the reference policy.

### 3. Value Model (V_ϕ)

**What:** Predicts the expected future reward from any state.

**Role:** The "critic" in actor-critic RL. Used to compute advantages.

**Updates:** YES (trained with value function loss)

**Code:** `self.value_model`

**Architecture:** Usually same as policy but with a value head instead of LM head:
```python
# Policy: hidden_states → lm_head → logits over vocabulary
# Value:  hidden_states → value_head → scalar value

class ValueModel(PreTrainedModel):
    def __init__(self, config):
        self.backbone = LlamaModel(config)  # Shared with policy
        self.score = nn.Linear(config.hidden_size, 1)  # Value head
```

**Why needed:** Reduces variance in advantage estimation (compared to Monte Carlo).

### 4. Reward Model (r_ψ)

**What:** Trained to predict human preferences. Scores (prompt, response) pairs.

**Role:** Provides the learning signal (the "reward").

**Updates:** NO (frozen during PPO)

**Code:** `self.reward_model`

**Training:** Trained separately on human preference data before PPO:
```python
# Preference data: prompt + chosen response + rejected response
# Loss: -log σ(r(prompt, chosen) - r(prompt, rejected))
```

---

## Initialization

### PPOTrainer.__init__

**Location:** `ppo_trainer.py:182-389`

#### Arguments

```python
trainer = PPOTrainer(
    args=PPOConfig(...),
    processing_class=tokenizer,  # Tokenizer
    model=policy_model,          # Policy (will be trained)
    ref_model=ref_model,         # Reference (frozen)
    reward_model=reward_model,   # Reward (frozen)
    value_model=value_model,     # Value (will be trained)
    train_dataset=dataset,
    peft_config=lora_config,     # Optional LoRA
)
```

#### Key setup steps

**1. Validate reference model (lines 198-202):**
```python
if ref_model is model:
    raise ValueError(
        "`model` and `ref_model` cannot be the same object. "
        "You must make a copy, or pass `None` if using PEFT."
    )
```

**Why:** If they're the same object, training the policy would also update the reference!

**With PEFT:** Can pass `ref_model=None`. The trainer will use adapter switching:
```python
# Forward with policy adapter
with model.set_adapter("policy"):
    policy_logits = model(...)

# Forward with base model (= reference)
with model.disable_adapter():
    ref_logits = model(...)
```

**2. PEFT wrapping (lines 234-248):**
```python
if peft_config is not None:
    # Merge any existing adapter first
    if isinstance(self.policy_model, PeftModel):
        self.policy_model = self.policy_model.merge_and_unload()

    # Apply new PEFT config
    self.policy_model = get_peft_model(self.policy_model, peft_config)

    # Cast to bf16 if using 4-bit
    if args.bf16 and getattr(self.policy_model, "is_loaded_in_4bit", False):
        peft_module_casting_to_bf16(self.policy_model)
```

**3. Reference model creation (lines 252-257):**
```python
if ref_model:
    self.ref_model = ref_model  # Use provided
elif self.is_peft_model:
    self.ref_model = None  # Will use adapter switching
else:
    self.ref_model = create_reference_model(self.policy_model)  # Create copy
```

**4. Calculate batch sizes (lines 271-300):**

PPO has complex batching hierarchy:

```
total_episodes (e.g., 100,000)
    ↓
num_total_batches = episodes / batch_size (e.g., 1000 batches)
    ↓
batch_size = per_device_batch * grad_accum * num_gpus (e.g., 128)
    ↓
mini_batch_size = batch_size / num_mini_batches (e.g., 32)
    ↓
per_device_train_batch_size (e.g., 4)
```

**Example calculation:**
```python
# Config
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
num_gpus = 2
num_mini_batches = 4

# Computed
local_batch_size = 4 * 4 = 16
batch_size = 16 * 2 = 32
mini_batch_size = 32 / 4 = 8
```

**Why mini-batches?** PPO reuses the same batch for multiple epochs and splits it into mini-batches for stable training.

**5. Disable dropout (lines 305-307):**
```python
for module in [self.policy_model, self.ref_model, self.value_model, self.reward_model]:
    if module is not None:
        disable_dropout_in_model(module)
```

**Why:** Dropout adds noise that interferes with RL training. We want deterministic forward passes.

**6. Wrap policy and value (line 308):**
```python
self.model = PolicyAndValueWrapper(self.policy_model, self.value_model)
```

**Why:** Allow single `accelerator.prepare()` call for both models.

**PolicyAndValueWrapper (lines 116-127):**
```python
class PolicyAndValueWrapper(nn.Module):
    def __init__(self, policy, value_model):
        super().__init__()
        self.policy = policy
        self.value_model = value_model
        self.critic_backbone = getattr(value_model, value_model.base_model_prefix)

    def forward(self, **kwargs):
        output = self.critic_backbone(**kwargs)  # Shared backbone
        logits = self.value_model.score(output.hidden_states[-1])  # Value head
        return self.policy(**kwargs), logits  # (policy_output, value_predictions)
```

---

## Training Loop

### Main Training Loop

**Location:** `ppo_trainer.py:425-763`

PPO training follows this structure:

```
for update in range(num_total_batches):
    # 1. ROLLOUT PHASE (generation)
    Sample batch of prompts
    Generate responses with current policy
    Compute log probabilities (policy and reference)
    Compute values
    Get rewards from reward model

    # 2. ADVANTAGE COMPUTATION
    Compute KL penalty
    Combine rewards: r_total = r_reward - β * KL
    Compute advantages using GAE
    Whiten advantages

    # 3. OPTIMIZATION PHASE (multiple PPO epochs)
    for ppo_epoch in range(num_ppo_epochs):
        Shuffle data
        for mini_batch in mini_batches:
            # Compute PPO losses
            policy_loss = PPO_clip_loss(advantages)
            value_loss = value_clip_loss(returns)
            total_loss = policy_loss + vf_coef * value_loss

            # Backprop
            optimizer.step()
```

Let's break down each phase:

### Phase 1: Rollout (Generation)

**Lines 492-568**

**Step 1.1: Sample prompts (line 491):**
```python
data = next(iter_dataloader)
queries = data["input_ids"].to(device)  # [batch_size, prompt_length]
context_length = queries.shape[1]
```

**Step 1.2: Generate responses (lines 502-511):**
```python
with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
    query_responses, logitss = batch_generation(
        unwrapped_model.policy,
        queries,
        args.local_rollout_forward_batch_size,
        processing_class.pad_token_id,
        generation_config,
    )
```

**What `batch_generation` does:**
1. Autoregressively generates tokens
2. Collects logits at each step
3. Returns full sequences + logits

**Why unwrap?** Accelerate wraps models for distributed training. For generation, we need the unwrapped model.

**Step 1.3: Compute policy log probabilities (lines 513-520):**
```python
for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
    query = queries[i : i + args.local_rollout_forward_batch_size]
    query_response = query_responses[i : i + args.local_rollout_forward_batch_size]
    response = query_response[:, context_length:]  # Extract response part
    logits = logitss[i : i + args.local_rollout_forward_batch_size]
    logprob = selective_log_softmax(logits, response)  # log π_θ(response|prompt)
```

**What is `selective_log_softmax`?**
```python
def selective_log_softmax(logits, labels):
    """
    Efficiently compute log probabilities for selected tokens.

    Instead of:
        log_probs_all = F.log_softmax(logits, dim=-1)  # Full vocabulary
        log_probs = log_probs_all.gather(labels)       # Select

    Do:
        log_probs = logits.gather(labels) - logits.logsumexp(dim=-1)
    """
    return logits.gather(dim=-1, labels.unsqueeze(-1)).squeeze(-1) - logits.logsumexp(dim=-1)
```

**Why selective?** Memory efficient - only computes log probs for generated tokens, not entire vocabulary.

**Step 1.4: Compute reference log probabilities (lines 522-531):**
```python
if ref_policy is None:
    # Use PEFT adapter switching
    with self.null_ref_context():
        ref_output = forward(model.policy, query_response, pad_token_id)
else:
    # Use separate reference model
    ref_output = forward(ref_policy, query_response, pad_token_id)

ref_logits = ref_output.logits[:, context_length - 1 : -1]
ref_logits /= args.temperature + 1e-7
ref_logprob = selective_log_softmax(ref_logits, response)  # log π_ref(response|prompt)
```

**Step 1.5: Truncate at stop token (lines 533-538):**
```python
postprocessed_response = response
if self.stop_token_id is not None:
    postprocessed_response = truncate_response(
        self.stop_token_id, pad_token_id, response
    )
```

**Why?** Model might generate past EOS. We only want to score up to the first EOS.

**Step 1.6: Compute rewards (lines 540-550):**
```python
# Value function: V(s)
full_value, _, _ = get_reward(
    unwrapped_value_model, query_response, pad_token_id, context_length
)
value = full_value[:, context_length - 1 : -1].squeeze(-1)

# Reward model: r(prompt, response)
_, score, _ = get_reward(
    reward_model, postprocessed_query_response, pad_token_id, context_length
)
```

**What `get_reward` does:**
```python
def get_reward(model, sequences, pad_token_id, context_length):
    """
    Get rewards from a reward model or value model.

    Returns:
        (full_values, last_value, mask)
    """
    output = model(sequences)
    # Reward model has a score head that outputs a scalar per token
    values = output.logits.squeeze(-1)  # [batch, seq_len]
    return values, values[:, -1], (sequences != pad_token_id)
```

**Step 1.7: Create padding mask (lines 577-584):**
```python
sequence_lengths = first_true_indices(postprocessed_response == pad_token_id) - 1
response_idxs = torch.arange(responses.shape[1], device=device).repeat(responses.shape[0], 1)
padding_mask = response_idxs > sequence_lengths.unsqueeze(1)

# Mask logprobs
logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)

# Mask values (with padding_mask_p1 to include the value at sequence end)
sequence_lengths_p1 = sequence_lengths + 1
padding_mask_p1 = response_idxs > sequence_lengths_p1.unsqueeze(1)
values = torch.masked_fill(values, padding_mask_p1, 0)
```

**Why `padding_mask_p1`?** Values are used to bootstrap future rewards. We need V(s_{t+1}), so we keep one extra value.

### Phase 2: Advantage Computation

**Lines 586-614**

**Step 2.1: Compute KL divergence (lines 587-590):**
```python
logr = ref_logprobs - logprobs  # log(π_ref / π_θ)
kl = -logr if args.kl_estimator == "k1" else (logr.exp() - 1) - logr  # k3
non_score_reward = -args.kl_coef * kl
```

**KL estimators:**
- **k1:** `-logr = log(π_θ / π_ref)` (straightforward, unbiased)
- **k3:** `exp(logr) - 1 - logr` (lower variance, also unbiased)

From [Approximating KL Divergence](http://joschu.net/blog/kl-approx.html).

**Step 2.2: Combine rewards (lines 591-594):**
```python
rewards = non_score_reward.clone()
actual_start = torch.arange(rewards.size(0), device=device)
actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)
rewards[actual_start, actual_end] += scores  # Add reward at sequence end
```

**Reward structure:**
```
Response:  [tok1, tok2, tok3, EOS, PAD, PAD]
KL reward: [-0.1, -0.05, -0.08, 0,   0,    0  ]
RM reward: [  0,    0,     0,   +5,  0,    0  ]  ← Only at EOS!
Total:     [-0.1, -0.05, -0.08, +5,  0,    0  ]
```

**Why reward at EOS only?** The reward model scores the complete response, not individual tokens.

**Step 2.3: Whiten rewards (lines 597-599):**
```python
if args.whiten_rewards:
    rewards = masked_whiten(rewards, mask=~padding_mask_p1, shift_mean=False)
    rewards = torch.masked_fill(rewards, padding_mask_p1, 0)
```

**Whitening:**
```python
def masked_whiten(values, mask, shift_mean=True):
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) / sqrt(var + eps)
    if not shift_mean:
        whitened += mean  # Keep mean, only standardize variance
    return whitened
```

**Why whiten?** Normalizes rewards across the batch, making training more stable.

**Step 2.4: Compute advantages with GAE (lines 601-613):**

**GAE (Generalized Advantage Estimation):**
```python
lastgaelam = 0
advantages_reversed = []
gen_length = responses.shape[1]

for t in reversed(range(gen_length)):
    nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
    delta = rewards[:, t] + gamma * nextvalues - values[:, t]  # TD error
    lastgaelam = delta + gamma * lam * lastgaelam  # GAE recursion
    advantages_reversed.append(lastgaelam)

advantages = torch.stack(advantages_reversed[::-1], axis=1)
returns = advantages + values  # Returns for value function training
advantages = masked_whiten(advantages, ~padding_mask)
```

**GAE formula:**
```
A^GAE_t = Σ_{l=0}^∞ (γλ)^l δ_{t+l}

where:
  δ_t = r_t + γ V(s_{t+1}) - V(s_t)  (TD error)
  γ = discount factor (default: 1.0)
  λ = GAE lambda (default: 0.95)
```

**Why GAE?**
- **λ=0:** A_t = δ_t (high bias, low variance) - only 1-step lookahead
- **λ=1:** A_t = Σ_l γ^l r_{t+l} - V(s_t) (low bias, high variance) - Monte Carlo
- **λ=0.95:** Balance between bias and variance

**Intuition:** GAE is an exponentially-weighted average of n-step TD errors.

### Phase 3: PPO Optimization

**Lines 616-695**

PPO uses **multiple epochs** on the same batch of data:

```python
for ppo_epoch_idx in range(args.num_ppo_epochs):  # Default: 4
    b_inds = np.random.permutation(args.local_batch_size)  # Shuffle
    for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
        mini_batch_inds = b_inds[mini_batch_start : mini_batch_end]
        for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
            # ... PPO loss computation and optimization
```

**Three levels of batching:**
1. **Outer loop:** Multiple epochs (4 by default)
2. **Middle loop:** Mini-batches (split batch into chunks)
3. **Inner loop:** Micro-batches (gradient accumulation)

**Step 3.1: Forward pass on mini-batch (lines 635-643):**
```python
output, vpred_temp = forward(model, mb_query_responses, pad_token_id)
logits = output.logits[:, context_length - 1 : -1]
logits /= args.temperature + 1e-7
new_logprobs = selective_log_softmax(logits, mb_responses)
new_logprobs = torch.masked_fill(new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB)

vpred = vpred_temp[:, context_length - 1 : -1].squeeze(-1)
vpred = torch.masked_fill(vpred, padding_mask_p1[micro_batch_inds], 0)
```

We recompute logprobs and values for the same responses (now with updated policy/value models).

**Step 3.2: Value function loss (lines 644-655):**

**Clipped value loss (from PPO paper):**
```python
vpredclipped = torch.clamp(
    vpred,
    mb_values - args.cliprange_value,  # Lower bound
    mb_values + args.cliprange_value,  # Upper bound
)
vf_losses1 = (vpred - mb_return) ** 2
vf_losses2 = (vpredclipped - mb_return) ** 2
vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), ~padding_mask_p1)
```

**Why clip?** Prevents the value function from changing too rapidly.

**Formula:**
```
L^VF = 0.5 * max[(V_θ(s) - V_target)^2,
                  (clip(V_θ(s), V_old - ε, V_old + ε) - V_target)^2]
```

**Step 3.3: Policy gradient loss (lines 656-661):**

**PPO clipped objective:**
```python
logprobs_diff = new_logprobs - mb_logprobs
ratio = torch.exp(logprobs_diff)  # π_new / π_old

pg_losses = -mb_advantage * ratio
pg_losses2 = -mb_advantage * torch.clamp(ratio, 1 - cliprange, 1 + cliprange)
pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), ~padding_mask)
```

**Formula:**
```
L^CLIP = min[A * ratio, A * clip(ratio, 1-ε, 1+ε)]

where:
  ratio = π_θ(a|s) / π_old(a|s)
  A = advantage
  ε = cliprange (default: 0.2)
```

**Why clip ratio?** This is the **key insight of PPO**:

If A > 0 (good action):
- Unclipped: increase π_θ (make action more likely)
- Clipped at 1+ε: prevent too large increase

If A < 0 (bad action):
- Unclipped: decrease π_θ (make action less likely)
- Clipped at 1-ε: prevent too large decrease

**Visual:**
```
ratio > 1+ε: Policy increased probability too much → clip at 1+ε
1-ε < ratio < 1+ε: Policy change is reasonable → use unclipped
ratio < 1-ε: Policy decreased probability too much → clip at 1-ε
```

**Step 3.4: Total loss and optimization (lines 662-665):**
```python
loss = pg_loss + args.vf_coef * vf_loss
accelerator.backward(loss)
optimizer.step()
optimizer.zero_grad()
```

**Why `vf_coef`?** Balance policy and value losses. Default 0.1 means value loss is weighted 10x less than policy loss.

**Step 3.5: Metrics (lines 666-683):**
```python
with torch.no_grad():
    pg_clipfrac = masked_mean((pg_losses2 > pg_losses).float(), ~padding_mask)  # Fraction clipped
    prob_dist = F.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
    approxkl = 0.5 * (logprobs_diff ** 2).mean()  # KL approximation

    # Store for logging
    approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
    pg_clipfrac_stats[...] = pg_clipfrac
    pg_loss_stats[...] = pg_loss
    vf_loss_stats[...] = vf_loss
    entropy_stats[...] = entropy.mean()
    ratio_stats[...] = ratio.mean()
```

**Key metrics:**
- **approxkl:** How much policy changed (should be small, < 0.01)
- **pg_clipfrac:** Fraction of policy updates that were clipped (indicates if cliprange is active)
- **entropy:** Policy diversity (should not collapse to zero)
- **ratio:** Policy change magnitude (should stay near 1.0)

### Phase 4: Logging and Cleanup

**Lines 696-757**

```python
with torch.no_grad():
    mean_kl = kl.sum(1).mean()
    mean_entropy = (-logprobs).sum(1).mean()
    mean_non_score_reward = non_score_reward.sum(1).mean()
    rlhf_reward = mean_non_score_reward + scores.mean()

    metrics = {
        "objective/kl": mean_kl,
        "objective/entropy": mean_entropy,
        "objective/rlhf_reward": rlhf_reward,
        "objective/scores": scores.mean(),
        "policy/approxkl_avg": approxkl_stats.mean(),
        "policy/clipfrac_avg": pg_clipfrac_stats.mean(),
        "loss/policy_avg": pg_loss_stats.mean(),
        "loss/value_avg": vf_loss_stats.mean(),
        "val/ratio": ratio_stats.mean(),
        # ... more metrics
    }
    self.log(metrics)

self.lr_scheduler.step()

# Memory cleanup (critical for long training!)
del kl, mean_kl, scores, rewards, advantages, returns, ...
empty_cache()
gc.collect()
```

**Why aggressive cleanup?** PPO holds many large tensors (responses, logprobs, values, advantages). Without cleanup, OOM is likely.

---

## PPO Loss Functions

### Policy Loss (Clipped Objective)

**Mathematical formulation:**

```
L^CLIP(θ) = E_t[min(r_t(θ) Â_t, clip(r_t(θ), 1-ε, 1+ε) Â_t)]

where:
  r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)  (probability ratio)
  Â_t = advantage estimate
  ε = cliprange (typically 0.2)
```

**Code:** `ppo_trainer.py:656-661`

```python
ratio = exp(new_logprobs - old_logprobs)
pg_losses = -advantage * ratio
pg_losses2 = -advantage * clip(ratio, 1 - ε, 1 + ε)
pg_loss = mean(max(pg_losses, pg_losses2))
```

**Why negative?** We want to maximize the objective, but optimizers minimize. So we minimize the negative.

### Value Loss (Clipped MSE)

**Mathematical formulation:**

```
L^VF(ϕ) = E_t[max((V_ϕ(s_t) - V_target)^2,
                   (clip(V_ϕ(s_t), V_old - ε_v, V_old + ε_v) - V_target)^2)]

where:
  V_target = Â_t + V_old(s_t)  (the "return")
  ε_v = cliprange_value (typically 0.2)
```

**Code:** `ppo_trainer.py:644-652`

```python
vpredclipped = clip(vpred, values - ε_v, values + ε_v)
vf_losses1 = (vpred - returns) ** 2
vf_losses2 = (vpredclipped - returns) ** 2
vf_loss = 0.5 * mean(max(vf_losses1, vf_losses2))
```

### Total Loss

```
L = L^CLIP + c_1 * L^VF

where c_1 = vf_coef (typically 0.1)
```

**Code:** `ppo_trainer.py:662`

```python
loss = pg_loss + vf_coef * vf_loss
```

(Note: Entropy bonus is optionally added in some implementations, but not in TRL's PPO)

---

## Configuration

### PPOConfig

**Location:** `ppo_config.py:23-136`

**Key hyperparameters:**

```python
@dataclass
class PPOConfig(OnPolicyConfig):
    # Reward shaping
    kl_coef: float = 0.05          # KL penalty coefficient
    kl_estimator: str = "k1"       # KL estimator ("k1" or "k3")
    gamma: float = 1.0             # Discount factor
    lam: float = 0.95              # GAE lambda

    # PPO clipping
    cliprange: float = 0.2         # Policy clip range
    cliprange_value: float = 0.2   # Value clip range

    # Training
    num_ppo_epochs: int = 4        # PPO epochs per batch
    vf_coef: float = 0.1           # Value loss coefficient
    whiten_rewards: bool = False   # Whiten rewards?

    # Models
    reward_model_path: str = "..."
    model_adapter_name: str | None = None  # For multi-adapter PEFT
    ref_adapter_name: str | None = None
```

**Typical values:**

| Hyperparameter | Conservative | Balanced | Aggressive |
|----------------|--------------|----------|------------|
| `kl_coef` | 0.1 | 0.05 | 0.01 |
| `cliprange` | 0.1 | 0.2 | 0.3 |
| `num_ppo_epochs` | 2 | 4 | 8 |
| `whiten_rewards` | False | False | True |

**Tuning guidance:**

- **High KL divergence?** Increase `kl_coef` (more penalty on deviation)
- **Policy not improving?** Decrease `kl_coef` or increase `cliprange` (allow bigger updates)
- **Unstable training?** Decrease `cliprange`, increase `num_ppo_epochs`
- **Overfitting to reward model?** Increase `kl_coef`, enable `whiten_rewards`

---

## Summary

**PPO Training Flow:**
1. **Generate** responses with current policy
2. **Score** with reward model
3. **Compute** advantages using value function + GAE
4. **Optimize** policy and value function with clipped objectives
5. **Repeat** for multiple epochs on same batch

**Key innovations:**
- **Clipped objective:** Prevents destructive policy updates
- **Value function:** Reduces variance in advantage estimation
- **GAE:** Balances bias-variance tradeoff in advantage computation
- **Multiple epochs:** Reuses data for sample efficiency

**Memory requirements (7B model):**
- Policy model: 14 GB (bf16)
- Reference model: 14 GB
- Value model: 14 GB (shared backbone with policy)
- Reward model: 14 GB
- Activations + rollout: 20-30 GB
- **Total: ~70-80 GB**

**With optimizations:**
- Use PEFT (LoRA): Saves 28 GB (ref becomes adapter switching)
- Quantize reward/value: Saves 14-21 GB
- **Optimized total: ~30-40 GB (fits on A100 40GB)**

PPO is complex but powerful. When tuned well, it produces the highest-quality RLHF models!
