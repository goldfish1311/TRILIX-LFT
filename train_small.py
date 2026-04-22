#!/usr/bin/env python3
"""
TRILIX Training - Small Config WITHOUT MoE (AGI-only)
2048d, 24L, AGI enabled
Target BPW: 0.0024 (~2.4 bits per weight)
For RTX 3090 with 24GB VRAM
"""

import sys

sys.path.insert(0, ".")

import torch
from trilix.config import TRILIXConfig
from trilix.model import TRILIXTransformer
import time

LOGFILE = "train_small_output.txt"


def log(msg):
    with open(LOGFILE, "a") as f:
        f.write(msg + "\n")
        f.flush()
    print(msg, flush=True)


device = torch.device("cuda")
log(f"TRILIX Small Training (AGI-only) on {device}")
log(f"GPU: {torch.cuda.get_device_name(0)}")

config = TRILIXConfig.small()
config.use_moe = False  # AGI-only (MoE forward path needs fixing)

log(f"\nConfig: {config.hidden_size}d, {config.num_hidden_layers}L")
log(f"Target BPW: {config.effective_bpw:.4f}")
log(
    f"Rank: {config.rank_r}, Codebook: {config.codebook_k}, Atoms: {config.num_atoms_A}"
)

model = TRILIXTransformer(config).to(device)
model.train()
log(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Set higher xor_temperature for softer AGI at start
for layer in model.modules():
    if hasattr(layer, "xor_temperature"):
        temp = layer.xor_temperature
        temp.data.fill_(2.0)
        layer.xor_temp_steps = 0


# Set AGI phase
def set_agi_phase(model, step):
    for layer in model.modules():
        if hasattr(layer, "atoms_U") and hasattr(layer, "agi_phase"):
            if step < 300:
                layer.atoms_U.requires_grad = False
                layer.atoms_V.requires_grad = False
                layer.agi_phase = 0
                layer.agi_weight = 0.0
            elif step < 500:
                layer.atoms_U.requires_grad = True
                layer.atoms_V.requires_grad = True
                layer.agi_phase = 1
                layer.agi_weight = 0.01 * (step - 300) / 200
            else:
                layer.agi_phase = 1
                layer.agi_weight = min(0.1, 0.01 + (step - 500) * 0.0001)


optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

batch_size = 4
seq_len = 256
max_steps = 5000
grad_accum = 4

log(f"\nTraining: bs={batch_size}, seq={seq_len}, accum={grad_accum}")
log(f"={'=' * 60}\n")

# Warmup
with torch.no_grad():
    _ = model(
        torch.randint(0, config.vocab_size, (2, 64), device=device),
        labels=torch.randint(0, config.vocab_size, (2, 64), device=device),
    )
log(f"VRAM after warmup: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

model.train()
step = 0
accum_loss = 0.0
start_time = time.time()

while step < max_steps:
    for ai in range(grad_accum):
        set_agi_phase(model, step)
        input_ids = torch.randint(
            0, config.vocab_size, (batch_size, seq_len), device=device
        )
        labels = torch.randint(
            0, config.vocab_size, (batch_size, seq_len), device=device
        )

        out = model(input_ids, labels=labels)
        (out["loss"] / grad_accum).backward()
        accum_loss += out["loss"].item()

        if ai == 0:
            vram = torch.cuda.memory_allocated() / 1e9
            grad_ok = all(
                p.grad is not None for p in model.parameters() if p.requires_grad
            )
            agi_on = step >= 300
            log(
                f"  [S{step} A1/{grad_accum}] CE={out['loss'].item():.4f} VRAM={vram:.1f}G Grad={'OK' if grad_ok else 'MISS'} AGI={'ON' if agi_on else 'OFF'}"
            )

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()
    step += 1

    elapsed = time.time() - start_time
    tps = (step * batch_size * seq_len * grad_accum) / max(elapsed, 0.1)
    vram = torch.cuda.memory_allocated() / 1e9

    log(
        f"Step {step:4d}/{max_steps} | Loss: {accum_loss:.4f} | Tok/s: {tps:.0f} | VRAM: {vram:.1f}GB | Time: {elapsed:.0f}s"
    )

    if elapsed > 180:
        log(f"\n=== 3 MINUTES DONE! Step {step} ===")
        torch.save(
            {"step": step, "model": model.state_dict(), "config": config},
            "trilix_small_agi_3min.pt",
        )
        log(f"Checkpoint saved: trilix_small_agi_3min.pt")
        break

    accum_loss = 0.0

log(f"\nTraining complete! {step} steps in {time.time() - start_time:.1f}s")
torch.save(model.state_dict(), "trilix_small_agi_final.pt")
log(f"Model saved: trilix_small_agi_final.pt")
