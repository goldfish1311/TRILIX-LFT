#!/usr/bin/env python3
"""
TRILIX Training Script for SlimPajama
Optimized for RTX 3090 (24GB VRAM)
"""

import os
import sys
import math
import time
import argparse
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent))

from trilix.config import TRILIXConfig
from trilix.model import TRILIXTransformer


class SimpleTextDataset(torch.utils.data.Dataset):
    """Simple text dataset for testing"""

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int = 100000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

        # Generate synthetic data (in practice, load from SlimPajama)
        print(f"Generating {num_samples} synthetic samples...")
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len + 1))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Return input_ids and labels (shifted by 1)
        sample = self.data[idx]
        return sample[:-1], sample[1:]


def get_optimizer(model: nn.Module, config: TRILIXConfig):
    """Create optimizer with differential learning rates"""

    scale_params = []
    binary_params = []
    embedding_params = []
    norm_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "row_scale" in name or "col_scale" in name or "latent_scale" in name:
            scale_params.append(param)
        elif "atoms" in name or "idx_" in name or "combo_" in name:
            binary_params.append(param)
        elif "embed" in name or "lm_head" in name:
            embedding_params.append(param)
        elif "norm" in name and "weight" in name:
            norm_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {
            "params": scale_params,
            "lr": config.base_lr * config.scale_lr_multiplier,
            "weight_decay": 0.0,
            "max_grad_norm": 0.5,  # HARD CLIP for scales!
            "name": "scales",
        },
        {
            "params": binary_params,
            "lr": config.base_lr * 0.01,
            "weight_decay": 0.0,
            "name": "binary_factors",
        },
        {
            "params": embedding_params,
            "lr": config.base_lr,
            "weight_decay": 0.0,
            "name": "embeddings",
        },
        {
            "params": norm_params,
            "lr": config.base_lr,
            "weight_decay": 0.0,
            "name": "norms",
        },
        {
            "params": other_params,
            "lr": config.base_lr,
            "weight_decay": config.weight_decay,
            "name": "other",
        },
    ]

    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)

    return optimizer


def get_lr_scheduler(optimizer, config: TRILIXConfig, total_steps: int):
    """Cosine schedule with warmup"""

    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / config.warmup_steps
        else:
            progress = (step - config.warmup_steps) / (
                total_steps - config.warmup_steps
            )
            return (1 - config.min_lr / config.base_lr) * (
                0.5 * (1 + math.cos(math.pi * progress))
            ) + config.min_lr / config.base_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def log_metrics(
    writer: Optional[SummaryWriter],
    step: int,
    model: TRILIXTransformer,
    loss: float,
    aux_losses: Dict,
    lr: float,
):
    """Log training metrics"""

    if writer is None:
        return

    # Main loss
    writer.add_scalar("train/loss", loss, step)
    writer.add_scalar("train/ce_loss", aux_losses.get("ce_loss", 0), step)
    writer.add_scalar("train/total_aux", aux_losses.get("total_aux", 0), step)
    writer.add_scalar(
        "train/diversity_total", aux_losses.get("diversity_total", 0), step
    )

    # Learning rate
    writer.add_scalar("train/lr", lr, step)

    # Scale statistics (critical for P0!)
    for name, param in model.named_parameters():
        if "scale" in name and "latent" not in name:
            writer.add_scalar(f"scales/{name}_mean", param.mean().item(), step)
            writer.add_scalar(f"scales/{name}_std", param.std().item(), step)
            writer.add_scalar(f"scales/{name}_max", param.abs().max().item(), step)

        # Check if scale is hitting clamp boundary
        if "scale" in name:
            max_val = param.abs().max().item()
            if max_val > 9.5:
                writer.add_scalar(f"scales/{name}_near_boundary", 1.0, step)

    # Codebook/Atom statistics (dead atoms)
    # Access first layer's attention projection
    first_layer = model.layers[0]
    if hasattr(first_layer, "self_attn") and hasattr(first_layer.self_attn, "q_proj"):
        q_proj = first_layer.self_attn.q_proj

        # Track XOR temperature
        writer.add_scalar("xor/temperature", q_proj.xor_temperature.item(), step)
        writer.add_scalar("xor/step", q_proj.training_step.item(), step)


def train_epoch(
    model: TRILIXTransformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    writer: Optional[SummaryWriter],
    epoch: int,
    gradient_accumulation_steps: int = 16,
    max_steps: int = 10000,
):
    """Train for one epoch with gradient accumulation"""

    model.train()

    step = epoch * len(dataloader)
    accumulated_loss = 0
    accumulated_steps = 0

    pbar = enumerate(dataloader)

    for batch_idx, (input_ids, labels) in pbar:
        if step >= max_steps:
            break

        input_ids = input_ids.to(device)
        labels = labels.to(device)

        # Forward
        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"] / gradient_accumulation_steps

        # Backward
        loss.backward()

        accumulated_loss += loss.item()
        accumulated_steps += 1

        # Gradient accumulation
        if accumulated_steps % gradient_accumulation_steps == 0:
            # Clip gradients differently per group
            for group in optimizer.param_groups:
                max_norm = group.get("max_grad_norm", 1.0)
                torch.nn.utils.clip_grad_norm_(group["params"], max_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Update XOR temperature
            for layer in model.layers:
                for proj in [
                    layer.self_attn.q_proj,
                    layer.self_attn.k_proj,
                    layer.self_attn.v_proj,
                    layer.self_attn.o_proj,
                    layer.mlp.gate_proj,
                    layer.mlp.up_proj,
                    layer.mlp.down_proj,
                ]:
                    if hasattr(proj, "training_step"):
                        proj.training_step += 1
                        if (
                            proj.training_step % 100 == 0
                            and proj.xor_temperature > 0.01
                        ):
                            proj.xor_temperature *= 0.99

            step += 1

            # Logging
            if step % 10 == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Step {step}/{max_steps} | Loss: {accumulated_loss:.4f} | LR: {lr:.6f}"
                )

                log_metrics(
                    writer,
                    step,
                    model,
                    accumulated_loss * gradient_accumulation_steps,
                    outputs["aux_losses"],
                    lr,
                )

            accumulated_loss = 0

            # Save checkpoint
            if step % 500 == 0:
                checkpoint_path = Path("checkpoints")
                checkpoint_path.mkdir(exist_ok=True)
                torch.save(
                    {
                        "step": step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    checkpoint_path / f"trilix_step_{step}.pt",
                )


def main():
    parser = argparse.ArgumentParser(description="Train TRILIX on SlimPajama")
    parser.add_argument("--config", type=str, default="nano", choices=["nano", "small"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--log-dir", type=str, default="./logs")

    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Config
    if args.config == "nano":
        config = TRILIXConfig.nano()
    else:
        config = TRILIXConfig.small()

    print(f"\n{'=' * 60}")
    print(f"TRILIX Training - {args.config.upper()} Config")
    print(f"{'=' * 60}")
    print(f"Target BPW: {config.effective_bpw:.4f}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Layers: {config.num_hidden_layers}")
    print(f"Rank: {config.rank_r}")
    print(f"Codebook size: {config.codebook_k}")
    print(f"Atoms: {config.num_atoms_A}")
    print(f"{'=' * 60}\n")

    # Model
    print("Creating model...")
    model = TRILIXTransformer(config).to(device)

    memory_stats = model.get_memory_stats()
    print(f"\nMemory footprint:")
    print(f"  Eval: {memory_stats.get('bpw_eval', 0):.4f} BPW")
    print(f"  Size: ~{memory_stats.get('trilix_eval_mb', 0):.1f} MB")

    # Dataset
    print(f"\nLoading dataset (synthetic for now)...")
    dataset = SimpleTextDataset(
        vocab_size=config.vocab_size, seq_len=args.seq_len, num_samples=100000
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # RTX 3090 single GPU
        pin_memory=True if device.type == "cuda" else False,
    )

    # Optimizer
    optimizer = get_optimizer(model, config)

    total_steps = args.max_steps
    scheduler = get_lr_scheduler(optimizer, config, total_steps)

    # TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir)

    print(f"\n{'=' * 60}")
    print(f"Starting training!")
    print(f"Total steps: {total_steps}")
    print(
        f"Batch size: {args.batch_size} (effective: {args.batch_size * args.gradient_accumulation})"
    )
    print(f"Gradient accumulation: {args.gradient_accumulation}")
    print(f"{'=' * 60}\n")

    # Train
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_epoch(
            model,
            dataloader,
            optimizer,
            scheduler,
            device,
            writer,
            epoch,
            args.gradient_accumulation,
            total_steps,
        )

    # Final save
    final_path = Path(args.output_dir) / "final_model.pt"
    final_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), final_path)
    print(f"\n✓ Training complete! Model saved to {final_path}")

    writer.close()


if __name__ == "__main__":
    main()
