# TRILIX-LFT Status Report for Claude Opus
## Executive Summary

We have implemented a **Triple-Level Indexed eXtreme Latent Factorized Transformer (TRILIX-LFT)** with target 0.0048 BPW (bits per weight). The architecture is based on your design with three compression levels:

1. **Level 1**: Latent Factorization (W ≈ U·V^T)
2. **Level 2**: Vector Quantization Codebook (indices → codewords)
3. **Level 3**: XOR Atoms (codewords = XOR of learnable atoms)

## Current Status: ✅ Architecture Works, ❌ Dead Atoms Problem

### What's Working
- Model initializes and runs on RTX 3090 (11.7 GB VRAM, 46% GPU util)
- Forward/backward pass completes without NaN
- Loss ~10.4 (random baseline, ln(32000) ≈ 10.37)
- Scales (row_scale, col_scale, latent_scale) are learning (LR 10x)
- All auxiliary losses (commitment, etc.) compute correctly

### Critical Issue: 100% Dead Atoms
**Observation**: All 3584 atoms across all layers show zero gradients.
**Consequence**: Level 3 (XOR atoms) is not learning at all.

## Technical Architecture Details

### Three-Level Compression Chain

```
Input x
    ↓
Level 1: Latent Factorization
    W_eff = diag(h) · U · diag(l) · V^T · diag(g)
    Where h, g, l are BF16 scales (learnable)
    U ∈ {±1}^{d_out × r}, V ∈ {±1}^{d_in × r} (binary)
    ↓
Level 2: VQ Codebook
    U[i] = C_U[idx_U[i]]  (lookup by index)
    C_U ∈ {±1}^{k × r} (k codewords, each r-dimensional)
    idx_U[i] = argmax(softmax(idx_U_logits[i]))  (STE)
    ↓
Level 3: XOR Atoms
    C_U[j] = sign(Σ_{b=1}^{3} α[j,b] · Atom_U[β[j,b]])
    Where:
      - α[j,b] ∈ R (continuous weights, learnable)
      - β[j,b] ∈ {0..31} (atom indices, STE)
      - Atom_U ∈ {±1}^{32 × r} (32 binary atoms)
```

### Gradient Flow Chain (The Problem)

**Full path from loss to atoms:**

```
Loss → Softmax(idx_U_logits) → STE (one-hot) 
    → Gather (codebook[indices])
    → einsum with atoms (via combo_indices)
    → combo_indices are STE from combo_indices_logits
    → einsum: selected = combo_indices_hard @ atoms_binary
    → tanh(sign(selected)/temperature)
    → STEBinary.apply → Loss
```

**Chain of STE operations:**
1. `idx_U_logits` → `softmax` → `argmax` → `STE` (one-hot)
2. `combo_indices_logits` → `softmax` → `argmax` → `STE` (one-hot) per codeword
3. `atoms_U` → `sign` → `STE` (binary)
4. `tanh(combined/temp)` → `STE` (binary output)

**The problem**: Gradient attenuation through 4 levels of STE:
- Each STE potentially kills 90% of gradient
- 4 levels = 0.1^4 = 0.0001 (0.01%) of original gradient
- Effectively zero gradient reaches atoms

## Code Structure

### File: `trilix/layers.py` (Core implementation)

```python
class TRILIXLinear(nn.Module):
    def __init__(self, in_f, out_f, rank=64, codebook_size=64, num_atoms=16):
        # Level 1: Scales
        self.row_scale = nn.Parameter(torch.ones(out_f))
        self.col_scale = nn.Parameter(torch.ones(in_f))
        self.latent_scale = nn.Parameter(torch.ones(rank))
        
        # Level 2: Codebook indices
        self.idx_U_logits = nn.Parameter(torch.randn(out_f, codebook_size) * 0.01)
        self.idx_V_logits = nn.Parameter(torch.randn(in_f, codebook_size) * 0.01)
        
        # Level 3: XOR Atoms
        self.atoms_U = nn.Parameter(torch.randn(num_atoms, rank) * 0.01)
        self.atoms_V = nn.Parameter(torch.randn(num_atoms, rank) * 0.01)
        self.combo_weights_U = nn.Parameter(torch.rand(codebook_size, 3) * 0.5 + 0.25)
        self.combo_indices_U_logits = nn.Parameter(
            torch.randn(codebook_size, 3, num_atoms) * 0.01
        )
    
    def forward(self, x):
        # Decode codebook from atoms
        codebook_U = self._decode_codebook(...)
        
        # Quantize U, V via codebook lookup
        U = self._quantize_U(codebook_U)  # Uses idx_U_logits with STE
        V = self._quantize_V(codebook_V)
        
        # Forward with scales
        return x @ (U * latent_scale @ V.T) * row_scale
    
    def _decode_codebook_entry(self, combo_weights, combo_indices_hard, atoms_binary):
        # combo_indices_hard: [k, 3, num_atoms] one-hot from STE
        # atoms_binary: [num_atoms, r] from STE(sign(atoms_U))
        selected = torch.einsum("kba,ar->kbr", combo_indices_hard, atoms_binary)
        weighted = selected * combo_weights.unsqueeze(-1)
        combined = weighted.sum(dim=1)
        # Soft XOR with temperature annealing
        if self.xor_temp_steps < 2000 and self.xor_temperature > 0.05:
            combined = torch.tanh(combined / self.xor_temperature)
        codebook = STEBinary.apply(combined)
        return codebook
```

### STE Implementation

```python
class STEBinary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.sign(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # Straight-through

class STEIndex(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_soft, x_hard):
        ctx.save_for_backward(x_soft)
        return x_hard
    
    @staticmethod
    def backward(ctx, grad_output):
        x_soft, = ctx.saved_tensors
        return grad_output, None  # Gradient to soft version
```

## The Dead Atoms Problem

### Current State
- **3584/3584 atoms dead** (100% of all atoms across all layers)
- Atoms_U.grad and Atoms_V.grad are None or all zeros
- combo_weights.grad has values (learning)
- combo_indices_logits.grad has values (learning via STE)
- Scales are learning well

### Root Cause Analysis

The gradient chain is too long and broken:

```
Loss
    ↓ [valid grad]
idx_U_logits.softmax
    ↓ [STE: grad passes]
idx_U_hard (one-hot)
    ↓ [gather: grad passes]
U_soft = codebook[idx_U]
    ↓ [valid grad]
U_hard = codebook_U[idx_U_hard] via STE
    ↓ [valid grad through codebook lookup]
combo_weights_U and combo_indices_logits
    ↓ [STE: grad PASSES to combo_indices_logits but NOT to atoms]
combo_indices_hard
    ↓ [einsum with atoms: SHOULD propagate grad]
selected_atoms = combo_indices_hard @ atoms_binary
    ↓ [PROBLEM: atoms_binary = sign(atoms_U), STE]
atoms_U
```

**The Break**: When we do `einsum("kba,ar->kbr", combo_indices_hard, atoms_binary)`:
- `combo_indices_hard` is differentiable (grad flows here from loss)
- `atoms_binary = STEBinary.apply(atoms_U)` 
- The gradient through STEBinary is identity: grad flows to atoms_U
- BUT atoms_U is initialized small (0.01 * randn)
- After sign(), it's ±1, but the original atoms_U is tiny
- Gradient to atoms_U exists but is minuscule
- Additionally: combo_indices_hard is STE output, which may block gradients differently

## Attempted Solutions

### 1. Temperature Annealing (Implemented)
- Soft XOR: `tanh(combined / temp)` where temp starts at 1.0 and anneals to 0.01
- **Result**: Still dead atoms
- **Issue**: Temperature affects smoothness but not the STE chain

### 2. Scale Clamping (Implemented)
- Clamps scales to [-10, 10]
- Gradient clipping at 0.5 for scales
- **Result**: Scales healthy, atoms still dead

### 3. Higher LR for Atoms (Attempted)
- Atoms LR: 3e-5 vs Scales LR: 3e-3 (100x difference)
- **Result**: No change, atoms still dead

## Required Solution

We need **direct gradient supervision** on atoms, bypassing the STE chain or strengthening it.

### Option A: Commitment Loss on Atoms
Add auxiliary loss that directly compares atoms to their target values:

```python
# Add to forward:
atom_commitment_loss = ||atoms_U - EMA_target||^2
```

### Option B: Simplify Level 3
Remove XOR atoms entirely, use direct codebook:

```python
# Instead of C[j] = XOR(atoms), use:
C = nn.Parameter(torch.randn(k, r))  # Direct codebook
```

This keeps Level 1+2, removes Level 3 (XOR).

### Option C: Initialize Atoms from Pre-computed
Use random orthogonal atoms from initialization, freeze them:

```python
# Initialize once with QR, then freeze
atoms_U = torch.linalg.qr(torch.randn(32, r))[0]
atoms_U.requires_grad = False
```

Then only learn combo_weights and combo_indices.

### Option D: Grad Bypass via Skip Connection
Add skip connection that preserves gradients:

```python
selected_atoms = combo_indices_hard @ atoms_binary + atoms_U * 0.01
```

This adds a small gradient path directly to atoms_U.

## Recommendation Request

Claude, which option do you recommend?

**Context**:
- We want to keep ~0.005 BPW if possible
- Can accept up to 0.01 BPW for working solution
- RTX 3090 with 24GB VRAM
- First goal: Proof of concept that sub-1-bit works
- Dataset: Synthetic for now, real SlimPajama next

**Questions**:
1. Is the three-level design fundamentally flawed for gradient flow?
2. Should we remove Level 3 and keep only Level 1+2?
3. Is there a better way to structure the STE chain?
4. Should we pre-train atoms in a separate stage?

## Current Hyperparameters

```yaml
Model: TRILIX Nano
  hidden_size: 1024
  num_layers: 16
  rank_r: 64
  codebook_k: 64
  num_atoms: 16
  xor_arity: 2  # Using b=2 not 3

Training:
  batch_size: 8
  seq_len: 256
  gradient_accumulation: 16
  effective_batch: 128
  scale_lr: 3e-3
  binary_lr: 3e-5
  weight_decay: 0.1
  grad_clip: 0.5 (scales), 1.0 (others)

Optimizer: AdamW with differential LR groups
```

## Working Code Location

All code is in `/home/evgeny/Mi progi githab/Experiment 0.1 bit-weight/trilix/`

Quick test:
```python
from trilix.config import TRILIXConfig
from trilix.model import TRILIXTransformer
import torch

config = TRILIXConfig.nano()
model = TRILIXTransformer(config).cuda()
# Check gradients:
for name, p in model.named_parameters():
    if 'atoms' in name and p.grad is not None:
        print(f"{name}: grad_norm={p.grad.abs().mean():.6f}")
```

---

**Question for Claude**: What's the most pragmatic fix to get gradients flowing to atoms? Should we redesign Level 3 or remove it entirely for v1?
