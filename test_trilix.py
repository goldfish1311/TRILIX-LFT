#!/usr/bin/env python3
"""
Quick test of TRILIX layer functionality
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from trilix.config import TRILIXConfig
from trilix.model import TRILIXTransformer


def test_trilix():
    print("Testing TRILIX implementation...")

    # Create nano config
    config = TRILIXConfig.nano()
    print(f"\nConfig: {config}")
    print(f"Effective BPW: {config.effective_bpw:.4f}")

    # Create model
    print("\nCreating model...")
    model = TRILIXTransformer(config)

    # Memory stats
    print("\nMemory statistics:")
    stats = model.get_memory_stats()
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 2
    seq_len = 128

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    print(f"Input shape: {input_ids.shape}")

    outputs = model(input_ids, labels=labels)

    print(f"Loss: {outputs['loss']:.4f}")
    print(f"Logits shape: {outputs['logits'].shape}")

    # Check aux losses
    print("\nAuxiliary losses:")
    for k, v in outputs["aux_losses"].items():
        if isinstance(v, (int, float)) and v > 0:
            print(f"  {k}: {v:.4f}")

    # Test parameter counting
    print("\nParameter statistics:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_trilix()
