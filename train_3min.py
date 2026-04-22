#!/usr/bin/env python3
"""TRILIX 3-minute test - write to file"""

import sys

sys.path.insert(0, ".")

import torch
from trilix.config import TRILIXConfig
from trilix.model import TRILIXTransformer
import time
import os

LOGFILE = "train_3min_output.txt"

with open(LOGFILE, "w") as f:
    f.write(f"TRILIX 3-min test\n")
    f.flush()

    device = torch.device("cuda")
    f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
    f.flush()

    config = TRILIXConfig.small()
    config.use_moe = True
    config.num_experts = 4
    config.moe_top_k = 2

    f.write(
        f"Config: {config.hidden_size}d, {config.num_hidden_layers}L, MoE={config.num_experts}\n"
    )
    f.write(f"Target BPW: {config.effective_bpw:.4f}\n")
    f.flush()

    model = TRILIXTransformer(config).to(device)
    model.train()
    f.write(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    f.flush()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    batch_size = 4
    seq_len = 256
    grad_accum = 16
    max_steps = 5000

    start_time = time.time()
    step = 0

    f.write(f"Training: bs={batch_size}, seq={seq_len}, accum={grad_accum}\n")
    f.write(f"{'=' * 60}\n")
    f.flush()

    while step < max_steps:
        for _ in range(grad_accum):
            input_ids = torch.randint(
                0, config.vocab_size, (batch_size, seq_len), device=device
            )
            out = model(input_ids, labels=input_ids)
            (out["loss"] / grad_accum).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        step += 1

        if step % 10 == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = (step * batch_size * seq_len * grad_accum) / max(
                elapsed, 0.1
            )
            loss_val = out["loss"].item()

            line = f"Step {step:4d}/{max_steps} | Loss: {loss_val:.4f} | Tok/s: {tokens_per_sec:.0f} | VRAM: {torch.cuda.memory_allocated() / 1e9:.1f}GB\n"
            f.write(line)
            f.flush()

            if loss_val < 600 and loss_val > 400:
                f.write("  🌱 Language structure emerging!\n")
                f.flush()
            elif loss_val < 400:
                f.write("  🎯 BREAKTHROUGH! Loss below 400!\n")
                f.flush()

        if time.time() - start_time > 180:
            f.write(f"\n=== 3 minutes done! Step {step} ===\n")
            f.flush()
            break

    f.write(f"\nTraining complete! {step} steps in {time.time() - start_time:.1f}s\n")
    f.flush()

torch.save(model.state_dict(), "trilix_small_moe_3min.pt")
with open(LOGFILE, "a") as f:
    f.write("Model saved to trilix_small_moe_3min.pt\n")
