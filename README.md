# TRILIX-LFT: Triple-Level Indexed eXtreme Latent Factorized Transformer

**Версия анализа:** 1.0 от 22.04.26  
**Статус:** Анализ проекта — может содержать ошибки в выводах

---

## ⚠️ ВАЖНОЕ ПРЕДУПРЕЖДЕНИЕ

**Этот документ — результат анализа исходного кода. Выводы могут быть неполными или ошибочными.**  

Реальное состояние проекта проверяй запуском:
```bash
python3 train_small_moe.py
```

---

<p align="center">
  <img src="https://img.shields.io/badge/BPW-0.0024-blue" alt="BPW">
  <img src="https://img.shields.io/badge/GPU-RTX%203090-green" alt="GPU">
  <img src="https.shields.io/badge/PyTorch-2.0+-red" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

**TRILIX-LFT** is a transformer architecture designed for extreme weight compression to **0.0024 bits per weight (BPW)** while maintaining full capability. Built on the principles of LittleBit (NeurIPS 2025), TRILIX introduces four levels of compression and a novel **Atom Gradient Injection (AGI)** technique to solve the "dead atoms" problem inherent in ultra-compressed models.

## 🎯 Key Features

- **4-Level Compression Architecture**
  - Level 1: Latent Factorization (W ≈ U·V^T)
  - Level 2: Vector Quantization Codebook
  - Level 3: XOR Atoms with AGI
  - Level 4: MoE-Codebook (4 experts, top-2)

- **Atom Gradient Injection (AGI)**: Dual-path forward pass that creates a fully differentiable gradient flow to atoms, solving the gradient collapse problem

- **MoE-Codebook**: Multiple codebook experts (syntax, semantics, rare tokens, context) with only ~2KB memory overhead

- **Small Config**: 2048d, 24L, fits comfortably on RTX 3090 (3.96 GB VRAM)

- **Target BPW**: 0.0024 (~2.4 bits per weight)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/goldfish1311/TRILIX-LFT.git
cd TRILIX-LFT

# Install dependencies
pip install torch>=2.0 numpy
```

### Training (Small + MoE)

```bash
python train_small_moe.py
```

This will:
- Train a 2048d, 24L model with MoE-Codebook
- Use AGI with phase-based activation (0-300 → 300-500 → 500+)
- Save checkpoints every 500 steps
- Log to TensorBoard

### Monitor Training

```bash
tensorboard --logdir=./trilix_small_moe_logs
```

## 📊 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    TRILIX-LFT Stack                         │
├─────────────────────────────────────────────────────────────┤
│ Level 4: MoE-Codebook (4 experts, top-2 routing)           │
│          ┌─────────┬─────────┬─────────┬─────────┐         │
│          │ Syntax  │Semantic │  Rare   │ Context │         │
│          │ Expert  │ Expert  │ Expert  │ Expert  │         │
│          └─────────┴─────────┴─────────┴─────────┘         │
├─────────────────────────────────────────────────────────────┤
│ Level 3: XOR Atoms (binary atomic patterns)                │
│          C[j] = sign(Σ α[j,b] · Atom[β[j,b]])              │
│          AGI: Dual-path for gradient flow                   │
├─────────────────────────────────────────────────────────────┤
│ Level 2: Vector Quantization (codebook indices)            │
│          U[i] = Codebook[idx_U[i]], idx_U ∈ {0..k-1}       │
├─────────────────────────────────────────────────────────────┤
│ Level 1: Latent Factorization                               │
│          W_eff = diag(h) · U · diag(l) · V^T · diag(g)     │
└─────────────────────────────────────────────────────────────┘
```

## 🧠 AGI: Atom Gradient Injection

The "dead atoms" problem occurs because gradients die through long STE chains. AGI solves this by creating a parallel, fully differentiable path:

- **Hard Path**: Uses STE for actual computation (sign, argmax)
- **Soft Path**: Uses tanh, softmax for gradient flow
- **Alignment Loss**: MSE(soft_codebook, hard_codebook.detach())

This creates direct gradient flow from loss → soft_codebook → atoms_U (NO STE!)

### Phase-Based Training

```python
Steps 0-300:  Atoms frozen (requires_grad=False)
              → combo_indices stabilize

Steps 300-500: Atoms unfrozen, AGI weight 0.01 → 0.1
              → atoms receive gradients

Steps 500+:   Full AGI (weight = 0.1)
              → atoms learn with coherent signals
```

## 📁 Project Structure

```
TRILIX-LFT/
├── trilix/
│   ├── __init__.py
│   ├── config.py          # TRILIXConfig (nano, small, medium)
│   ├── layers.py          # TRILIXLinear, MoECodebook, AGI
│   └── model.py           # TRILIXTransformer, Attention, SwiGLU
├── train_small_moe.py     # Training script (Small + MoE)
├── train_quick.py         # Quick training (Nano)
├── README.md
├── ARCHITECTURE.md        # Detailed architecture docs
└── TRAINING.md            # Training guide and recommendations
```

## 🎓 Citation

```bibtex
@article{trilix2025,
  title={TRILIX-LFT: Triple-Level Indexed eXtreme Latent Factorized Transformer},
  author={[Your Name]},
  year={2025},
  url={https://github.com/goldfish1311/TRILIX-LFT}
}
```

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Knowledge distillation from larger models
- Gradient checkpointing for Medium (4096d) config
- Real dataset integration (SlimPajama)
- Kernel optimization for XOR operations

---

<p align="center">
  <b>Built with ❤️ and extreme compression</b>
</p>
