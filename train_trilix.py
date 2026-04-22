#!/usr/bin/env python3
"""
TRILIX Training Script
Train TRILIX-LFT model from scratch with native 0.04-0.09 BPW
"""

import os
import sys
import math
import json
import argparse
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Add trilix to path
sys.path.insert(0, str(Path(__file__).parent))

from trilix.config import TRILIXConfig
from trilix.model import TRILIXTransformer


def setup_distributed():
    """Setup distributed training"""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank = 0
        local_rank = 0
        world_size = 1

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    return rank, local_rank, world_size, device


def get_optimizer(model: nn.Module, config: TRILIXConfig) -> optim.Optimizer:
    """
    Create optimizer with differential learning rates for different parameter groups
    """
    # Separate parameters into groups
    scale_params = []
    binary_params = []
    embedding_params = []
    norm_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Scale factors (highest LR)
        if "row_scale" in name or "col_scale" in name or "latent_scale" in name:
            scale_params.append(param)
        # Binary latent factors
        elif "atoms" in name or "idx_" in name or "combo_" in name:
            binary_params.append(param)
        # Embeddings
        elif "embed" in name or "lm_head" in name:
            embedding_params.append(param)
        # Norm weights
        elif "norm" in name and "weight" in name:
            norm_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {
            "params": scale_params,
            "lr": config.base_lr * config.scale_lr_multiplier,
            "weight_decay": 0.0,
            "name": "scales",
        },
        {
            "params": binary_params,
            "lr": config.base_lr * 0.01,  # Very small for STE
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

    optimizer = optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)

    return optimizer


def get_lr_scheduler(
    optimizer: optim.Optimizer, config: TRILIXConfig, total_steps: int
):
    """Cosine learning rate schedule with warmup"""

    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / config.warmup_steps
        else:
            progress = (step - config.warmup_steps) / (
                total_steps - config.warmup_steps
            )
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            decayed = (
                1 - config.min_lr / config.base_lr
            ) * cosine_decay + config.min_lr / config.base_lr
            return decayed

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_step(
    model: nn.Module,
    batch: torch.Tensor,
    optimizer: optim.Optimizer,
    device: torch.device,
    gradient_accumulation_steps: int = 1,
) -> Dict:
    """Single training step"""

    batch = batch.to(device)

    # Forward
    outputs = model(batch, labels=batch)
    loss = outputs["loss"]

    # Scale for gradient accumulation
    loss = loss / gradient_accumulation_steps

    # Backward
    loss.backward()

    metrics = {
        "loss": loss.item() * gradient_accumulation_steps,
        "ce_loss": outputs["aux_losses"].get("ce_loss", 0),
        "commitment_U": sum(
            v for k, v in outputs["aux_losses"].items() if "commitment_U" in k
        ),
        "commitment_V": sum(
            v for k, v in outputs["aux_losses"].items() if "commitment_V" in k
        ),
        "diversity": sum(
            v for k, v in outputs["aux_losses"].items() if "diversity" in k
        ),
    }

    return metrics


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int = 100,
) -> Dict:
    """Evaluate model on validation set"""
    model.eval()

    total_loss = 0
    total_ce = 0
    num_batches = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break

            batch = batch.to(device)
            outputs = model(batch, labels=batch)

            total_loss += outputs["loss"].item()
            total_ce += outputs["aux_losses"].get("ce_loss", 0)
            num_batches += 1

    model.train()

    return {
        "loss": total_loss / num_batches,
        "ce_loss": total_ce / num_batches,
        "perplexity": math.exp(total_ce / num_batches),
    }


def main():
    parser = argparse.ArgumentParser(description="Train TRILIX model")
    parser.add_argument(
        "--config", type=str, default="small", choices=["nano", "small", "medium"]
    )
    parser.add_argument(
        "--data-path", type=str, required=True, help="Path to training data"
    )
    parser.add_argument("--output-dir", type=str, default="./trilix_outputs")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=100000)

    args = parser.parse_args()

    # Setup
    rank, local_rank, world_size, device = setup_distributed()

    # Config
    if args.config == "nano":
        config = TRILIXConfig.nano()
    elif args.config == "small":
        config = TRILIXConfig.small()
    else:
        config = TRILIXConfig.medium()

    if rank == 0:
        print(f"TRILIX Training - {args.config} config")
        print(f"Effective BPW: {config.effective_bpw:.4f}")
        print(f"Device: {device}")
        print(f"World size: {world_size}")

    # Model
    model = TRILIXTransformer(config).to(device)

    if rank == 0:
        memory_stats = model.get_memory_stats()
        print(f"Memory stats:")
        for k, v in memory_stats.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    # Distributed
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Optimizer
    optimizer = get_optimizer(model, config)

    # Scheduler
    total_steps = args.max_steps
    scheduler = get_lr_scheduler(optimizer, config, total_steps)

    # TODO: Data loading
    # For now, create dummy data
    # Replace with actual dataset

    if rank == 0:
        print(f"Starting training for {total_steps} steps...")

    model.train()
    step = 0
    accumulated_steps = 0

    while step < total_steps:
        # Dummy batch (replace with actual data)
        batch = torch.randint(0, config.vocab_size, (args.batch_size, args.seq_len))

        # Train step
        metrics = train_step(
            model,
            batch,
            optimizer,
            device,
            args.gradient_accumulation,
        )

        accumulated_steps += 1

        # Gradient update
        if accumulated_steps % args.gradient_accumulation == 0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            step += 1

            # Logging
            if rank == 0 and step % 10 == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Step {step}/{total_steps} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"CE: {metrics['ce_loss']:.4f} | "
                    f"Commit: {metrics['commitment_U']:.4f} | "
                    f"Diversity: {metrics['diversity']:.4f} | "
                    f"LR: {lr:.6f}"
                )

            # Save checkpoint
            if step % args.save_every == 0 and rank == 0:
                save_path = Path(args.output_dir) / f"checkpoint_step_{step}.pt"
                save_path.parent.mkdir(parents=True, exist_ok=True)

                state = {
                    "step": step,
                    "model": model.module.state_dict()
                    if isinstance(model, DDP)
                    else model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config.__dict__,
                }
                torch.save(state, save_path)
                print(f"Saved checkpoint to {save_path}")

            # Evaluate
            if step % args.eval_every == 0:
                pass  # TODO: evaluation

    if rank == 0:
        print("Training complete!")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
