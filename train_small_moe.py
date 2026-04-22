#!/usr/bin/env python3
"""
TRILIX Training - Small Config with MoE
2048d, 24L, MoE-Codebook (4 experts, top-2)
Target BPW: 0.0024 (~2.4 bits per weight)
For RTX 3090 with 24GB VRAM
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
print(f"TRILIX Small + MoE Training - Starting on {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

# Config - Small with MoE
config = TRILIXConfig.small()
config.use_moe = True
config.num_experts = 4
config.moe_top_k = 2

print(f"Target BPW: {config.effective_bpw:.4f}")
print(f"Config: {config.hidden_size}d, {config.num_hidden_layers}L")
print(f"MoE: {config.num_experts} experts, top-{config.moe_top_k}")
print(
    f"Rank: {config.rank_r}, Codebook: {config.codebook_k}, Atoms: {config.num_atoms_A}\n"
)

# Model
model = TRILIXTransformer(config).to(device)
model.train()

print("Model created successfully!")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

# Optimizer - differential learning rates
scale_params = []
binary_params = []
moe_params = []
other_params = []

for name, param in model.named_parameters():
    if "scale" in name and "latent" not in name:
        scale_params.append(param)
    elif "moe" in name or "expert" in name or "router" in name:
        moe_params.append(param)
    elif "atoms" in name or "idx_" in name or "combo_" in name:
        binary_params.append(param)
    else:
        other_params.append(param)

optimizer = torch.optim.AdamW(
    [
        {"params": scale_params, "lr": 3e-3, "weight_decay": 0.0},  # 10x faster
        {"params": moe_params, "lr": 1e-4, "weight_decay": 0.1},  # MoE routing
        {"params": binary_params, "lr": 3e-5, "weight_decay": 0.0},  # Slow (STE)
        {"params": other_params, "lr": 3e-4, "weight_decay": 0.1},  # Normal
    ],
    betas=(0.9, 0.95),
)

# Training config
batch_size = 4  # Reduced for larger model
seq_len = 256
grad_accumulation = 16  # Effective batch = 64
max_steps = 5000  # Longer training for convergence

print(f"Training config:")
print(f"  Batch size: {batch_size}")
print(f"  Seq length: {seq_len}")
print(f"  Gradient accumulation: {grad_accumulation}")
print(f"  Effective batch: {batch_size * grad_accumulation}")
print(f"  Max steps: {max_steps}")
print(f"  MoE enabled: {config.use_moe}")
print(f"  AGI Phases: 0-300 frozen, 300-500 gradual, 500+ full\n")

print(f"{'=' * 60}")
print("TRILIX Small + MoE Training")
print("AGI: 0-300 combo stabilization → 300-500 atom warmup → 500+ full AGI")
print(f"{'=' * 60}\n")


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


writer = SummaryWriter(log_dir="./trilix_small_moe_logs")
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
        moe_aux_loss = 0.0
        for layer in model.modules():
            if hasattr(layer, "_cached_agi_loss"):
                agi_loss_total += layer._cached_agi_loss
            if hasattr(layer, "moe_codebook_U") and layer.moe_codebook_U is not None:
                # MoE aux loss (load balancing)
                if hasattr(layer.moe_codebook_U, "_cached_moe_aux_loss"):
                    moe_aux_loss += layer.moe_codebook_U._cached_moe_aux_loss

        # Total loss: CE + AGI + MoE aux
        total_loss = (
            ce_loss + agi_loss_total + 0.01 * moe_aux_loss
        ) / grad_accumulation

        # Backward
        total_loss.backward()
        accumulated_loss += total_loss.item()

    # Gradient clip
    torch.nn.utils.clip_grad_norm_(scale_params, max_norm=0.5)
    torch.nn.utils.clip_grad_norm_(moe_params, max_norm=1.0)
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

        # Get AGI status
        agi_active = step >= 300
        agi_weight_avg = 0
        layer_count = 0
        for layer in model.modules():
            if hasattr(layer, "agi_weight"):
                agi_weight_avg += layer.agi_weight
                layer_count += 1
        if layer_count > 0:
            agi_weight_avg /= layer_count

        # Get MoE status
        moe_gates = []
        for layer in model.modules():
            if hasattr(layer, "moe_codebook_U") and layer.moe_codebook_U is not None:
                if hasattr(layer.moe_codebook_U, "_cached_gates"):
                    moe_gates.append(layer.moe_codebook_U._cached_gates.mean().item())
        moe_gate_avg = sum(moe_gates) / len(moe_gates) if moe_gates else 0

        print(
            f"Step {step:4d}/{max_steps} | "
            f"Loss: {accumulated_loss:.4f} | "
            f"CE: {ce_loss_val:.2f} | "
            f"AGI: {'ON' if agi_active else 'OFF'} (w={agi_weight_avg:.3f}) | "
            f"MoE: {moe_gate_avg:.3f} | "
            f"Tok/s: {tokens_per_sec:.0f}"
        )

        writer.add_scalar("train/loss", accumulated_loss, step)
        writer.add_scalar("train/ce_loss", ce_loss_val, step)
        writer.add_scalar("train/agi_weight", agi_weight_avg, step)
        writer.add_scalar("train/moe_gate", moe_gate_avg, step)

        # Check scale health
        scale_max = max(p.abs().max().item() for p in scale_params)
        scale_mean = sum(p.abs().mean().item() for p in scale_params) / len(
            scale_params
        )
        writer.add_scalar("health/scale_max", scale_max, step)
        writer.add_scalar("health/scale_mean", scale_mean, step)

        # Scale diagnostics
        if scale_max > 9.0:
            print(f"  ⚠️  Scales near boundary ({scale_max:.2f})")
        elif 3.0 < scale_max < 7.0:
            print(f"  ✅ Scales healthy ({scale_max:.2f})")

        # Memory check
        if device.type == "cuda":
            memory_gb = torch.cuda.memory_allocated() / 1e9
            writer.add_scalar("health/memory_gb", memory_gb, step)
            if memory_gb > 20:
                print(f"  ⚠️  Memory high: {memory_gb:.1f} GB")

        # Loss milestones
        if ce_loss_val < 8.0 and ce_loss_val > 5.0:
            print(f"  🌱 Language structure emerging!")
        elif ce_loss_val < 5.0:
            print(f"  🎯 BREAKTHROUGH! Loss below 5.0!")

        accumulated_loss = 0

    # Save checkpoint
    if step % 500 == 0:
        torch.save(
            {
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
            },
            f"trilix_small_moe_step_{step}.pt",
        )
        print(f"  💾 Checkpoint saved at step {step}")

print(f"\n{'=' * 60}")
print("Training complete!")
print(f"{'=' * 60}")
writer.close()

# Final save
torch.save(model.state_dict(), "trilix_small_moe_final.pt")
print(f"\n✓ Model saved to trilix_small_moe_final.pt")
print(f"✓ Logs saved to ./trilix_small_moe_logs")
print(f"\nTo view training metrics:")
print(f"  tensorboard --logdir=./trilix_small_moe_logs")
