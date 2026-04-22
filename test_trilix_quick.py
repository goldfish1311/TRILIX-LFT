#!/usr/bin/env python3
"""Quick sanity test for TRILIX"""

import sys
import traceback

sys.path.insert(0, "/home/evgeny/Mi progi githab/Experiment 0.1 bit-weight")

try:
    print("Importing PyTorch...")
    import torch

    print(f"✓ PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")

    print("\nImporting TRILIX modules...")
    from trilix.config import TRILIXConfig
    from trilix.model import TRILIXTransformer

    print("✓ Imports successful")

    print("\nCreating nano config...")
    config = TRILIXConfig.nano()
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Effective BPW: {config.effective_bpw:.4f}")

    print("\nCreating model...")
    model = TRILIXTransformer(config)
    print("✓ Model created")

    print("\nTesting forward pass...")
    batch_size = 2
    seq_len = 32

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    outputs = model(input_ids, labels=labels)

    print(f"✓ Forward pass successful!")
    print(f"  Loss: {outputs['loss'].item():.4f}")
    print(f"  Logits shape: {outputs['logits'].shape}")
    print(f"  Past KV length: {len(outputs['past_key_values'])}")

    print("\n✓✓✓ All tests passed! TRILIX is working! ✓✓✓")

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    print("\nTraceback:")
    traceback.print_exc()
    sys.exit(1)
