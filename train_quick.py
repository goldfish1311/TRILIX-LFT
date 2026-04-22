#!/usr/bin/env python3
"""
TRILIX Quick Start - Minimal Training Loop
For RTX 3090, ~4GB VRAM with Nano config
"""

import sys

sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from trilix.config import TRILIXConfig
from trilix.model import TRILIXTransformer
import time

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"TRILIX Training - Starting on {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

# Config - Nano for RTX 3090
config = TRILIXConfig.nano()
print(f"Target BPW: {config.effective_bpw:.4f}")
print(
    f"Model: {config.hidden_size}d, {config.num_hidden_layers}L, rank={config.rank_r}\n"
)

# Model
model = TRILIXTransformer(config).to(device)
model.train()

print("Model created successfully!")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

# Optimizer - differential learning rates
scale_params = []
binary_params = []
other_params = []

for name, param in model.named_parameters():
    if "scale" in name and "latent" not in name:
        scale_params.append(param)
    elif "atoms" in name or "idx_" in name or "combo_" in name:
        binary_params.append(param)
    else:
        other_params.append(param)

optimizer = torch.optim.AdamW(
    [
        {"params": scale_params, "lr": 3e-3, "weight_decay": 0.0},  # 10x faster
        {"params": binary_params, "lr": 3e-5, "weight_decay": 0.0},  # Slow (STE)
        {"params": other_params, "lr": 3e-4, "weight_decay": 0.1},  # Normal
    ],
    betas=(0.9, 0.95),
)

# Training loop
batch_size = 8
seq_len = 256
grad_accumulation = 16  # Effective batch = 128
max_steps = 1000


# AGI Phase Scheduler (Claude's innovation)
def set_agi_phase(model, step):
    """
    Phase-based AGI activation:
    - Phase 0 (0-300): Atoms frozen, combo_indices stabilize
    - Phase 1 (300-500): Atoms unfrozen, AGI starts at 0.01
    - Phase 2 (500+): Full AGI at 0.1
    """
    for layer in model.modules():
        if hasattr(layer, "atoms_U") and hasattr(layer, "agi_phase"):
            if step < 300:
                # Phase 0: Stabilization
                layer.atoms_U.requires_grad = False
                layer.atoms_V.requires_grad = False
                layer.agi_phase = 0
                layer.agi_weight = 0.0
            elif step < 500:
                # Phase 1: Gradual activation
                layer.atoms_U.requires_grad = True
                layer.atoms_V.requires_grad = True
                layer.agi_phase = 1
                progress = (step - 300) / 200
                layer.agi_weight = 0.01 * progress
            else:
                # Phase 2: Full AGI
                layer.agi_phase = 1
                layer.agi_weight = min(0.1, 0.01 + (step - 500) * 0.0001)


print(f"Training config:")
print(f"  Batch size: {batch_size}")
print(f"  Seq length: {seq_len}")
print(f"  Gradient accumulation: {grad_accumulation}")
print(f"  Effective batch: {batch_size * grad_accumulation}")
print(f"  Max steps: {max_steps}")
print(f"  AGI Phases: 0-300 frozen, 300-500 gradual, 500+ full\n")

print(f"{'=' * 60}")
print("TRILIX Training with AGI - Atom Gradient Injection")
print("Phases: 0-300 combo stabilization → 300-500 atom warmup → 500+ full AGI")
print(f"{'=' * 60}\n")

writer = SummaryWriter(log_dir="./trilix_logs")
step = 0
accumulated_loss = 0
start_time = time.time()

# Initialize AGI phase
set_agi_phase(model, step)

while step < max_steps:
    for _ in range(grad_accumulation):
        # Update AGI phase before forward
        set_agi_phase(model, step)

        # Generate synthetic batch (replace with real data)
        input_ids = torch.randint(
            0, config.vocab_size, (batch_size, seq_len), device=device
        )
        labels = torch.randint(
            0, config.vocab_size, (batch_size, seq_len), device=device
        )

        # Forward
        outputs = model(input_ids, labels=labels)
        ce_loss = outputs["loss"]

        # Collect AGI loss from all TRILIXLinear layers
        agi_loss_total = 0.0
        for layer in model.modules():
            if hasattr(layer, "_cached_agi_loss"):
                agi_loss_total += layer._cached_agi_loss

        # Total loss: CE + AGI
        total_loss = (ce_loss + agi_loss_total) / grad_accumulation

        # Backward
        total_loss.backward()
        accumulated_loss += total_loss.item()

    # Gradient clip for scales (P0!)
    torch.nn.utils.clip_grad_norm_(scale_params, max_norm=0.5)
    torch.nn.utils.clip_grad_norm_(binary_params, max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(other_params, max_norm=1.0)

    optimizer.step()
    optimizer.zero_grad()

    step += 1

    # Logging
    if step % 10 == 0:
        elapsed = time.time() - start_time
        tokens_per_sec = (step * batch_size * seq_len * grad_accumulation) / elapsed

        # Get CE loss from outputs
        ce_loss_val = outputs["aux_losses"].get("ce_loss", 0)
        if isinstance(ce_loss_val, torch.Tensor):
            ce_loss_val = ce_loss_val.item()

        print(
            f"Step {step:4d}/{max_steps} | "
            f"Loss: {accumulated_loss:.4f} | "
            f"CE: {ce_loss_val:.2f} | "
            f"Tok/s: {tokens_per_sec:.0f}"
        )

        writer.add_scalar("train/loss", accumulated_loss, step)
        writer.add_scalar("train/ce_loss", ce_loss_val, step)

        # Check scale health
        scale_max = max(p.abs().max().item() for p in scale_params)
        scale_mean = sum(p.abs().mean().item() for p in scale_params) / len(
            scale_params
        )
        writer.add_scalar("health/scale_max", scale_max, step)
        writer.add_scalar("health/scale_mean", scale_mean, step)

        # Scale diagnostics (from Gemini)
        if scale_max > 9.0:
            print(
                f"  ⚠️  Scales near boundary ({scale_max:.2f}) — model may need more capacity"
            )
        elif scale_max > 7.0:
            print(f"  🟡 Scale at {scale_max:.2f} — acceptable but watch closely")
        elif 3.0 < scale_max < 7.0:
            print(f"  ✅ Scale at {scale_max:.2f} — healthy range")

        # XOR temperature monitoring (from DeepSeek)
        xor_temps = []
        for layer in model.layers:
            for name, mod in layer.self_attn.named_modules():
                if hasattr(mod, "xor_temperature"):
                    xor_temps.append(mod.xor_temperature.item())
        if xor_temps:
            avg_temp = sum(xor_temps) / len(xor_temps)
            writer.add_scalar("health/xor_temperature", avg_temp, step)
            if avg_temp < 0.06 and avg_temp > 0.04:
                print(f"  🔴 XOR transition imminent! temp={avg_temp:.3f}")
            if avg_temp <= 0.01:
                print(f"  ⚡ Hard XOR mode active")

        # Dead atoms monitoring (from DeepSeek)
        dead_atoms = 0
        total_atoms = 0
        for layer in model.layers:
            for name, mod in layer.named_modules():
                if hasattr(mod, "atoms_U") and hasattr(mod, "usage_counter_U"):
                    with torch.no_grad():
                        # Check which atoms are actually used (non-zero gradient)
                        atom_grad_U = (
                            mod.atoms_U.grad is not None
                            and mod.atoms_U.grad.abs().sum(dim=1).gt(0).sum().item()
                        )
                        atom_grad_V = (
                            mod.atoms_V.grad is not None
                            and mod.atoms_V.grad.abs().sum(dim=1).gt(0).sum().item()
                        )
                        dead_atoms += (mod.num_atoms - atom_grad_U) + (
                            mod.num_atoms - atom_grad_V
                        )
                        total_atoms += mod.num_atoms * 2
        if total_atoms > 0:
            writer.add_scalar("health/dead_atoms", dead_atoms, step)
            if dead_atoms > total_atoms * 0.6:
                print(
                    f"  ⚠️  Dead atoms: {dead_atoms}/{total_atoms} — codebook collapse risk!"
                )

        # Loss milestones (from Gemini)
        if ce_loss > 8.0:
            pass  # Still random
        elif 5.0 <= ce_loss <= 8.0:
            print(f"  🌱 Language structure emerging!")
        elif ce_loss < 5.0:
            print(f"  🎯 BREAKTHROUGH! Loss below 5.0 — TRILIX is learning!")

        accumulated_loss = 0

    # Save checkpoint
    if step % 100 == 0:
        torch.save(
            {
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            f"trilix_checkpoint_step_{step}.pt",
        )
        print(f"  💾 Checkpoint saved")

print(f"\n{'=' * 60}")
print("Training complete!")
print(f"{'=' * 60}")
writer.close()

# Final save
torch.save(model.state_dict(), "trilix_final.pt")
print(f"\n✓ Model saved to trilix_final.pt")
print(f"✓ Logs saved to ./trilix_logs")
print(f"\nTo view training metrics:")
print(f"  tensorboard --logdir=./trilix_logs")
