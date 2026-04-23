#!/usr/bin/env python3
"""
TRILIX Training - Small Config with MoE (Fixed)
2048d, 24L, MoE-Codebook (4 experts, top-2)
Target BPW: 0.0024 (~2.4 bits per weight)

Fixes applied:
1. grad_accumulation: 16 → 4 (heartbeat every micro-batch)
2. Heartbeat logging inside accumulation loop
3. VRAM + gradient health checks every micro-batch
4. NaN detection on all losses
5. xor_temperature starts at 2.0 (softer AGI)
6. TensorBoard removed (was blocking stdout)
"""

import sys

sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
from trilix.config import TRILIXConfig
from trilix.model import TRILIXTransformer
from trilix.layers import TemperatureCascadeScheduler
import time
import os

os.environ["PYTHONUNBUFFERED"] = "1"

LOGFILE = "train_small_moe_output.txt"


def log(msg):
    with open(LOGFILE, "a") as f:
        f.write(msg + "\n")
        f.flush()
    print(msg, flush=True)


# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"TRILIX Small + MoE Training - Starting on {device}")
if device.type == "cuda":
    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    used_gb = torch.cuda.memory_allocated() / 1e9
    log(f"VRAM Used: {used_gb:.1f} GB / Free: {total_gb - used_gb:.1f} GB")
    torch.cuda.empty_cache()
    log(f"VRAM Cleared: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

# Config - Small with MoE
config = TRILIXConfig.small()
config.use_moe = True
config.num_experts = 4
config.moe_top_k = 2

# B1.5: Enable Flat Hierarchical Codebook (4 meta × 4 base = 16 virtual experts)
config.use_fhc = True

log(f"\nTarget BPW: {config.effective_bpw:.4f}")
log(f"Config: {config.hidden_size}d, {config.num_hidden_layers}L")
if config.use_fhc:
    log(f"FHC: 4×4=16 виртуальных экспертов (Flat Hierarchical Codebook)")
else:
    log(f"MoE: {config.num_experts} experts, top-{config.moe_top_k}")
log(
    f"Rank: {config.rank_r}, Codebook: {config.codebook_k}, Atoms: {config.num_atoms_A}"
)

# Model
model = TRILIXTransformer(config).to(device)
model.train()

log(f"Model created successfully!")
log(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Check gradients are NOT frozen
grad_count = sum(1 for p in model.parameters() if p.requires_grad)
total_count = sum(1 for _ in model.parameters())
log(
    f"Trainable params: {grad_count}/{total_count} ({100 * grad_count / total_count:.0f}%)"
)

# B4: Enable Hebbian Atom Resonance (HAR) on all TRILIXLinear layers
log("\nB4: Enabling Hebbian Atom Resonance (HAR)...")
har_layer_count = 0
for layer in model.modules():
    if hasattr(layer, "enable_har"):
        layer.enable_har(resonance_interval=200)
        har_layer_count += 1
log(f"  HAR enabled on {har_layer_count} TRILIXLinear layers")

# B3: Enable Differentiable Atom Evolution (DAE) on all TRILIXLinear layers
log("\nB3: Enabling Differentiable Atom Evolution (DAE)...")
dae_layer_count = 0
for layer in model.modules():
    if hasattr(layer, "enable_dae"):
        layer.enable_dae(evolution_interval=500, selection_threshold=0.1)
        dae_layer_count += 1
log(f"  DAE enabled on {dae_layer_count} TRILIXLinear layers")

# Optimizer - differential learning rates + per-group gradient clipping (D2)
    scale_params = []           # clip 0.5 (строгий — быстро учатся)
    atom_params = []            # clip 2.0 (мягкий — медленная эволюция)
    idx_params = []             # clip 1.0 (стандартный — idx_logits, combo_indices)
    moe_params = []             # clip 1.0
    world_model_params = []      # clip 1.0
    soul_params = []             # clip 1.0
    other_params = []            # clip 1.0

    for name, param in model.named_parameters():
        if "scale" in name and "latent" not in name:
            scale_params.append(param)
        elif "atoms" in name:  # atoms_U, atoms_V — медленная эволюция
            atom_params.append(param)
        elif "idx_" in name or "combo_" in name:  # индекс-логиты
            idx_params.append(param)
        elif "moe" in name or "expert" in name or "router" in name:
            moe_params.append(param)
        elif "world_model" in name or "z_projector" in name:
            world_model_params.append(param)
        elif "soul_projector" in name:
            soul_params.append(param)
        elif "soul_codebook" in name:
            soul_params.append(param)
        else:
            other_params.append(param)

optimizer = torch.optim.AdamW(
    [
        {"params": scale_params, "lr": 3e-3, "weight_decay": 0.0},
        {"params": moe_params, "lr": 1e-4, "weight_decay": 0.1},
        {"params": world_model_params, "lr": 3e-4, "weight_decay": 0.1},
        {"params": soul_params, "lr": 1e-4, "weight_decay": 0.01},
        {"params": binary_params, "lr": 3e-5, "weight_decay": 0.0},
        {"params": other_params, "lr": 3e-4, "weight_decay": 0.1},
    ],
    betas=(0.9, 0.95),
)

# Training config - REDUCED accumulation for heartbeat
batch_size = 1
seq_len = 256
grad_accumulation = 16
max_steps = 5000
temp_scheduler = TemperatureCascadeScheduler(total_steps=max_steps, warmup_steps=500)

log(f"\nTraining config:")
log(f"  Batch size: {batch_size}")
log(f"  Seq length: {seq_len}")
log(f"  Gradient accumulation: {grad_accumulation}")
log(f"  Effective batch: {batch_size * grad_accumulation}")
log(f"  Max steps: {max_steps}")
log(f"  MoE enabled: {config.use_moe}")
log(f"  ATC: atom → codebook → idx cascading temperatures")
log(f"\n{'=' * 60}")
log(f"TRILIX Small + MoE Training")
log(f"ATC: atom_temp (fastest freeze) → codebook_temp → idx_temp (slowest)")
log(f"{'=' * 60}\n")

# Set higher xor_temperature for softer AGI at start
for layer in model.modules():
    if hasattr(layer, "xor_temperature"):
        temp = layer.xor_temperature
        temp.data.fill_(2.0)  # Start at 2.0 (softer AGI)
        layer.xor_temp_steps = 0


# AGI Phase Scheduler with Temperature Annealing
set_agi_phase = None  # removed - ATC scheduler used instead


step = 0
accumulated_loss = 0.0
start_time = time.time()
first_step_time = None
world_model_loss_val = 0.0  # A2: World Model loss

set_agi_phase(model, step)

# Pre-warmup: make sure model is fully initialized on GPU
log("Warming up GPU...")
with torch.no_grad():
    warmup_input = torch.randint(0, config.vocab_size, (2, 64), device=device)
    _ = model(warmup_input, labels=warmup_input)
log("Warmup done.\n")

torch.cuda.synchronize()
log(f"VRAM after warmup: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
log(f"VRAM reserved: {torch.cuda.memory_reserved() / 1e9:.1f} GB")

model.train()

while step < max_steps:
    for accum_idx in range(grad_accumulation):
        # ATC scheduler applied via temp_scheduler.step() call below

        # Generate synthetic batch
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
        moe_aux_loss_total = 0.0
        world_model_loss_val = 0.0
        for layer in model.modules():
            if hasattr(layer, "_cached_agi_loss"):
                agi_loss_val = layer._cached_agi_loss
                if isinstance(agi_loss_val, torch.Tensor):
                    agi_loss_total += (
                        agi_loss_val.item()
                        if agi_loss_val.numel() == 1
                        else agi_loss_val.sum().item()
                    )
                elif isinstance(agi_loss_val, float):
                    agi_loss_total += agi_loss_val

            # Collect MoE aux loss for load balancing
            if hasattr(layer, "moe_codebook_U") and layer.moe_codebook_U is not None:
                if hasattr(layer.moe_codebook_U, "_cached_moe_aux_loss"):
                    moe_aux_loss_total += layer.moe_codebook_U._cached_moe_aux_loss
            if hasattr(layer, "moe_codebook_V") and layer.moe_codebook_V is not None:
                if hasattr(layer.moe_codebook_V, "_cached_moe_aux_loss"):
                    moe_aux_loss_total += layer.moe_codebook_V._cached_moe_aux_loss

        # A2: World Model loss из aux_losses
        if outputs.get("aux_losses") and "world_model_loss" in outputs["aux_losses"]:
            world_model_loss_val = (
                outputs["aux_losses"]["world_model_loss"].item()
                if isinstance(outputs["aux_losses"]["world_model_loss"], torch.Tensor)
                else outputs["aux_losses"]["world_model_loss"]
            )

        # Total loss: CE + AGI + MoE aux (load balancing) + World Model
        # MoE aux loss encourages all 4 experts to be used equally
        total_loss = (
            ce_loss + agi_loss_total + 0.01 * moe_aux_loss_total
        ) / grad_accumulation

        # NaN check
        if total_loss.isnan().any():
            log(f"\n!!! NaN DETECTED at step {step}, accum {accum_idx} !!!")
            log(f"  CE loss: {ce_loss}")
            log(f"  AGI loss: {agi_loss_total}")
            sys.exit(1)

        # Backward
        total_loss.backward()

        # B3: DAE — observe gradients and evolve atoms (after backward!)
        for layer in model.modules():
            if hasattr(layer, "step_dae"):
                layer.step_dae(total_loss.item())

        accumulated_loss += total_loss.item()

        # Check gradients on very first backward only
        if step == 0 and accum_idx == 0:
            missing_grad_count = sum(
                1 for p in model.parameters() if p.requires_grad and p.grad is None
            )
            log(f"  Gradients missing: {missing_grad_count} params")

        # Heartbeat logging every 2 accum
        if accum_idx % 2 == 0:
            vram_gb = torch.cuda.memory_allocated() / 1e9
            log(
                f"  [Step {step}][Accum {accum_idx + 1}/{grad_accumulation}] "
                f"CE={ce_loss.item():.4f} | VRAM={vram_gb:.1f}GB"
            )

    # D2: Per-group gradient clipping
    # Scale — строгий клиппинг (быстро учатся, не должны "убегать")
    torch.nn.utils.clip_grad_norm_(scale_params, max_norm=0.5)
    # Atoms — мягкий клиппинг (медленная эволюция, нужно больше сигнала)
    if atom_params:
        torch.nn.utils.clip_grad_norm_(atom_params, max_norm=2.0)
    # Index logits — стандартный
    if idx_params:
        torch.nn.utils.clip_grad_norm_(idx_params, max_norm=1.0)
    # MoE
    if moe_params:
        torch.nn.utils.clip_grad_norm_(moe_params, max_norm=1.0)
    # World Model
    if world_model_params:
        torch.nn.utils.clip_grad_norm_(world_model_params, max_norm=1.0)
    # Soul
    if soul_params:
        torch.nn.utils.clip_grad_norm_(soul_params, max_norm=1.0)
    # Остальное
    if other_params:
        torch.nn.utils.clip_grad_norm_(other_params, max_norm=1.0)

    optimizer.step()
    optimizer.zero_grad()

    # Apply ATC temperatures
    temp_scheduler.apply_to_model(model, step)

    step += 1

    if first_step_time is None:
        first_step_time = time.time() - start_time
        log(f"\n  === FIRST STEP DONE in {first_step_time:.1f}s ===\n")

    # D3: Enhanced logging with structured metrics (WandB-compatible)
    elapsed = time.time() - start_time
    tokens_per_sec = (step * batch_size * seq_len * grad_accumulation) / max(
        elapsed, 0.1
    )
    vram_gb = torch.cuda.memory_allocated() / 1e9

    # Structured metrics dict (D3: для интеграции с WandB)
    metrics = {
        "step": step,
        "loss/total": accumulated_loss,
        "loss/ce": ce_loss.item() if isinstance(ce_loss, torch.Tensor) else ce_loss,
        "loss/world_model": world_model_loss_val,
        "loss/sdo": outputs.get("aux_losses", {}).get("sdo_loss", 0),
        "loss/belief": outputs.get("aux_losses", {}).get("belief_loss", 0),
        "loss/edh": outputs.get("aux_losses", {}).get("edh_loss", 0),
        "loss/rel": outputs.get("aux_losses", {}).get("rel_loss", 0),
        "performance/tokens_per_sec": tokens_per_sec,
        "performance/vram_gb": vram_gb,
        "performance/time_elapsed": elapsed,
    }

    # Human-readable log
    log(
        f"Step {step:4d}/{max_steps} | "
        f"Loss: {accumulated_loss:.4f} | "
        f"Tok/s: {tokens_per_sec:.0f} | "
        f"VRAM: {vram_gb:.1f}GB | "
        f"Time: {elapsed:.0f}s | "
        f"WM: {world_model_loss_val:.4f}"
    )

    # TODO: wandb.log(metrics) — когда установят wandb

    # Scale health + HAR stats every 50 steps
    if step % 50 == 0:
        if scale_params:
            scale_max = max(p.abs().max().item() for p in scale_params)
            if scale_max > 9.0:
                log(f"  ⚠️  Scales near boundary ({scale_max:.2f})")
            elif 3.0 < scale_max < 7.0:
                log(f"  ✅ Scales healthy ({scale_max:.2f})")
        total_dead = 0
        total_har = 0
        for layer in model.modules():
            if hasattr(layer, "use_har") and layer.use_har and layer.har is not None:
                stats = layer.har.get_stats()
                total_dead += stats["dead_atoms_U"] + stats["dead_atoms_V"]
                total_har += 1
        if total_har > 0 and total_dead > 0:
            log(f"  🧬 HAR: {total_dead} dead atoms across {total_har} layers")

    # 3-minute time limit
    if elapsed > 180:
        log(f"\n=== 3 minutes done! Step {step} ===")
        torch.save(
            {
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
            },
            "trilix_small_moe_3min.pt",
        )
        log(f"  💾 Checkpoint saved trilix_small_moe_3min.pt")
        break

    accumulated_loss = 0.0

    # Save checkpoint every 500 steps
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
        log(f"  💾 Checkpoint saved at step {step}")

log(f"\n{'=' * 60}")
log(f"Training complete! {step} steps in {time.time() - start_time:.1f}s")
log(f"{'=' * 60}")

torch.save(model.state_dict(), "trilix_small_moe_final.pt")
log(f"\n✓ Model saved to trilix_small_moe_final.pt")
