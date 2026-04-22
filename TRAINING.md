# TRILIX-LFT Training Guide

## Quick Start

```bash
# Small config (2048d, 24L) with MoE
python train_small_moe.py

# Monitor training
tensorboard --logdir=./trilix_small_moe_logs
```

## Training Configurations

### Nano Config (Quick Testing)
```python
config = TRILIXConfig.nano()
# 1024d, 16L, rank=64, codebook_k=64
# VRAM: ~2 GB
# Use for: debugging, AGI testing
```

### Small Config (Recommended)
```python
config = TRILIXConfig.small()
config.use_moe = True
config.num_experts = 4
config.moe_top_k = 2
# 2048d, 24L, rank=100, codebook_k=128
# VRAM: ~4 GB
# Use for: serious training
```

### Medium Config (Advanced)
```python
config = TRILIXConfig.medium()
# 4096d, 32L, rank=200, codebook_k=128
# VRAM: ~6-7 GB with gradient checkpointing
# Use for: maximum quality on RTX 3090
```

## Phase-Based Training (AGI)

### Phase 0: Stabilization (Steps 0-300)

**What happens**: Atoms are frozen, only combo_indices and scales learn.

**Why**: Destructive interference occurs when atoms are trained before indices stabilize. Random combo_indices → conflicting gradients → dead atoms.

```python
if step < 300:
    layer.atoms_U.requires_grad = False
    layer.atoms_V.requires_grad = False
    layer.agi_phase = 0
    layer.agi_weight = 0.0
```

**Expected behavior**:
- Loss starts high (~300-500 for random init)
- Scales adjust to factorized structure
- combo_indices begin stabilizing
- **No atom gradients** (atoms frozen)

**Monitoring**:
```python
# Check that atoms are frozen
for name, p in model.named_parameters():
    if 'atoms' in name:
        assert not p.requires_grad, "Atoms should be frozen!"
```

### Phase 1: AGI Warmup (Steps 300-500)

**What happens**: Atoms unfrozen, AGI starts at 0.01 and ramps to 0.1

**Why**: Gradual activation prevents shock to the system. AGI provides direct gradient path while STE path stabilizes.

```python
elif step < 500:
    layer.atoms_U.requires_grad = True
    layer.atoms_V.requires_grad = True
    layer.agi_phase = 1
    progress = (step - 300) / 200
    layer.agi_weight = 0.01 * progress  # 0.01 → 0.1
```

**Expected behavior**:
- AGI loss appears and decreases
- Atoms begin receiving gradients
- Loss may temporarily spike (atoms adjusting)

**Monitoring**:
```python
# Check AGI status
agi_loss = outputs["aux_losses"].get("agi_total", 0)
print(f"AGI Loss: {agi_loss:.4f}")

# Check atom gradients
alive_atoms = sum(
    1 for name, p in model.named_parameters()
    if 'atoms' in name and p.grad is not None 
    and p.grad.abs().mean() > 1e-10
)
print(f"Alive atoms: {alive_atoms}")
```

### Phase 2: Full AGI (Steps 500+)

**What happens**: AGI at full strength (weight = 0.1), atoms learn with coherent signals

**Why**: By now, combo_indices are stable and atoms receive consistent gradients.

```python
else:
    layer.agi_phase = 1
    layer.agi_weight = min(0.1, 0.01 + (step - 500) * 0.0001)
```

**Expected behavior**:
- Atoms actively learning
- AGI loss decreasing
- Main loss steadily dropping

**Success indicators**:
- AGI alignment loss < 1.0
- >80% atoms have non-zero gradients
- CE loss dropping consistently

## MoE Training Recommendations

### Gemini's Recommendations for Small-MoE

#### 1. Top-2 Gating Normalization

**Critical**: Ensure gates sum to 1 via softmax:

```python
# In MoECodebook.forward()
gates = gates / gates.sum(dim=-1, keepdim=True)  # Normalize!
```

**Add jitter for stability** (optional):

```python
# During early training (steps < 1000)
if step < 1000 and self.training:
    jitter = torch.randn_like(router_logits) * 0.01
    router_logits = router_logits + jitter
```

#### 2. AGI in MoE Context

**Critical**: AGI must be calculated for ALL experts, not just active ones!

```python
# AGI for all experts (even inactive)
agi_loss_all = 0
for i, expert in enumerate(self.experts):
    soft_codebook = expert.get_codewords_soft()
    hard_codebook = expert.get_codewords()
    agi_loss_all += F.mse_loss(soft_codebook, hard_codebook.detach())

# Only active experts contribute to main loss
active_loss = sum(gates[i] * expert_output[i] for i in active_ids)
```

**Why**: Inactive experts must continue learning or they'll stay random.

#### 3. Load Balancing

MoE requires load balancing loss to prevent expert collapse:

```python
router_prob = F.softmax(router_logits, dim=-1)
aux_loss = num_experts * (
    router_prob.mean(dim=[0,1]) * 
    (router_prob > 0).float().mean(dim=[0,1])
).sum()

# Add to total loss
total_loss = ce_loss + 0.01 * aux_loss
```

### Expert Usage Monitoring

Track expert utilization:

```python
expert_counts = torch.zeros(num_experts)
for batch_idx in range(batch_size):
    for seq_idx in range(seq_len):
        for k in range(top_k):
            expert_id = expert_ids[batch_idx, seq_idx, k].item()
            expert_counts[expert_id] += 1

# Check balance
min_usage = expert_counts.min()
max_usage = expert_counts.max()
imbalance = max_usage / (min_usage + 1e-10)

if imbalance > 5.0:
    print("⚠️  Expert imbalance detected! Increase load balancing weight.")
```

## Hyperparameter Tuning

### Learning Rates

```python
optimizer = torch.optim.AdamW([
    {"params": scale_params, "lr": 3e-3, "weight_decay": 0.0},    # Fast
    {"params": moe_params, "lr": 1e-4, "weight_decay": 0.1},     # MoE routing
    {"params": binary_params, "lr": 3e-5, "weight_decay": 0.0}, # Slow (STE)
    {"params": other_params, "lr": 3e-4, "weight_decay": 0.1},  # Normal
], betas=(0.9, 0.95))
```

**Tuning guidelines**:
- Scale LR (3e-3): If scales explode (>10), reduce to 1e-3
- Binary LR (3e-5): If atoms dead, try 1e-4
- MoE LR (1e-4): If routing unstable, reduce to 3e-5

### Gradient Clipping

```python
# Different clips for different parameter types
torch.nn.utils.clip_grad_norm_(scale_params, max_norm=0.5)
torch.nn.utils.clip_grad_norm_(moe_params, max_norm=1.0)
torch.nn.utils.clip_grad_norm_(binary_params, max_norm=1.0)
torch.nn.utils.clip_grad_norm_(other_params, max_norm=1.0)
```

**Tuning guidelines**:
- Scale clip (0.5): Prevents runaway scales
- Others (1.0): Standard
- If gradients vanish: increase to 2.0
- If gradients explode: decrease to 0.1

### Batch Size and Accumulation

**RTX 3090 (24 GB)**:
```python
# Small config
batch_size = 4
seq_len = 256
grad_accumulation = 16  # Effective batch = 64

# Or for more stability:
batch_size = 2
seq_len = 512
grad_accumulation = 32  # Effective batch = 64
```

**Memory-saving tips**:
- Use gradient checkpointing (medium config)
- Reduce seq_len to 128 if OOM
- Enable mixed precision training (FP16/BF16)

## Debugging Guide

### Issue: Atoms Still Dead (100%)

**Diagnosis**: AGI not working or atoms not unfrozen

**Solution**:
```python
# Check phase
for layer in model.modules():
    if hasattr(layer, 'agi_phase'):
        print(f"Phase: {layer.agi_phase}, Weight: {layer.agi_weight}")
        print(f"atoms_U.requires_grad: {layer.atoms_U.requires_grad}")

# Verify AGI forward
assert hasattr(layer, '_cached_agi_loss')
print(f"AGI loss: {layer._cached_agi_loss}")
```

### Issue: MoE Expert Collapse (1 expert dominates)

**Diagnosis**: Load balancing loss too weak

**Solution**:
```python
# Increase load balancing weight
self.moe_aux_weight = 0.1  # was 0.01

# Or add expert dropout
def forward(self, x):
    if self.training and random.random() < 0.1:
        # Randomly disable one expert during training
        pass
```

### Issue: Scales Explode (>10)

**Diagnosis**: Scale LR too high or no gradient clipping

**Solution**:
```python
# Reduce scale LR
scale_lr = 1e-3  # was 3e-3

# Increase clipping
torch.nn.utils.clip_grad_norm_(scale_params, max_norm=0.1)

# Check scale values
if scale_max > 9.0:
    print("⚠️  Scales near boundary - adjusting...")
    for p in scale_params:
        p.data.clamp_(-8.0, 8.0)
```

### Issue: Loss Not Decreasing

**Diagnosis**: Multiple possible causes

**Checklist**:
1. [ ] AGI activated? (step > 300)
2. [ ] Atoms have gradients? (check with hooks)
3. [ ] Scales reasonable? (max < 10)
4. [ ] MoE balanced? (imbalance < 5x)
5. [ ] Learning rate schedule? (warmup implemented)

## Advanced Techniques

### Knowledge Distillation

Coming in v2.0: Distill from larger teacher model (4-bit):

```python
# Teacher: Llama-3.2-1B-Instruct in 4-bit (~0.8 GB)
teacher_logits = teacher_model(input_ids)
student_logits = model(input_ids)

# KL divergence loss
distill_loss = F.kl_div(
    F.log_softmax(student_logits / temperature, dim=-1),
    F.softmax(teacher_logits / temperature, dim=-1),
    reduction='batchmean'
)

total_loss = ce_loss + 0.5 * distill_loss
```

### Gradient Checkpointing

For Medium config (4096d):

```python
from torch.utils.checkpoint import checkpoint

class TRILIXLayer(nn.Module):
    def forward(self, x):
        # Checkpoint attention
        attn_out = checkpoint(self.self_attn, x)
        x = x + attn_out
        
        # Checkpoint FFN
        ffn_out = checkpoint(self.mlp, x)
        x = x + ffn_out
        
        return x
```

### Curriculum Learning

Start with easier tasks:

```python
# Phase 1 (steps 0-1000): Short sequences
seq_len = 128

# Phase 2 (steps 1000-3000): Medium sequences
seq_len = 256

# Phase 3 (steps 3000+): Full sequences
seq_len = 512
```

## Performance Benchmarks

### Expected Training Speed (RTX 3090)

| Config | Tok/s | Time/1k steps | VRAM |
|--------|-------|---------------|------|
| Nano | ~2500 | ~3 min | 2 GB |
| Small | ~1200 | ~7 min | 4 GB |
| Small+MoE | ~1000 | ~8 min | 5 GB |
| Medium | ~600 | ~15 min | 7 GB |

### Expected Loss Curves

**Nano config (1024d, 16L)**:
- Step 0: ~62 (ln(vocab_size))
- Step 300: ~15-20 (AGI activates)
- Step 1000: ~8-12
- Step 5000: ~5-7

**Small config (2048d, 24L)**:
- Step 0: ~65
- Step 300: ~18-25
- Step 1000: ~10-15
- Step 5000: ~6-8

**Small+MoE config**:
- Step 0: ~70 (higher capacity)
- Step 300: ~20-30
- Step 1000: ~12-18
- Step 5000: ~5-7 (better than non-MoE!)

## Troubleshooting

### Memory Issues

**OOM during forward**:
```python
# Reduce batch size
batch_size = 2  # was 4

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Clear cache
torch.cuda.empty_cache()
```

**OOM during backward**:
```python
# Accumulate gradients over more steps
grad_accumulation = 32  # was 16

# Reduce sequence length
seq_len = 128  # was 256
```

### NaN Loss

**Check**:
1. Scale values (should be < 10)
2. XOR temperature (should be > 0.01)
3. Learning rate (try 10x smaller)
4. Gradient norms (check if exploding)

**Fix**:
```python
# Detect NaN
if torch.isnan(loss):
    print("NaN detected! Debugging...")
    for name, p in model.named_parameters():
        if p.grad is not None and torch.isnan(p.grad).any():
            print(f"NaN gradient in {name}")
            # Zero out NaN
            p.grad.data = torch.where(
                torch.isnan(p.grad), 
                torch.zeros_like(p.grad), 
                p.grad
            )
```

## References

- **LittleBit**: [NeurIPS 2025]
- **VQ-VAE**: [van den Oord et al., 2017]
- **STE**: [Bengio et al., 2013]
- **MoE**: [Shazeer et al., 2017]
- **Switch Transformer**: [Fedus et al., 2021]

---

**Last Updated**: 2025-04-22
**Version**: v1.0.0
