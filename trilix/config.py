"""
TRILIX Configuration
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class TRILIXConfig:
    """Configuration for TRILIX model"""

    # Model architecture
    hidden_size: int = 2048
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    intermediate_size: int = 8192
    vocab_size: int = 32000
    max_position_embeddings: int = 4096

    # TRILIX-specific compression parameters
    rank_r: int = 100  # Latent factorization rank
    codebook_k: int = 128  # Number of codewords in codebook
    num_atoms_A: int = 32  # Number of XOR atoms
    xor_arity_b: int = 3  # Number of atoms XORed for each codeword

    # MoE-Codebook parameters (Level 4)
    use_moe: bool = False  # Enable Mixture of Experts for codebook
    num_experts: int = 4  # Number of codebook experts
    moe_top_k: int = 2  # Top-k experts per token

    # Scaling factors
    scale_lr_multiplier: float = 10.0

    # Training parameters
    base_lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    gradient_clip: float = 1.0

    # Loss coefficients
    commitment_beta: float = 0.25
    atom_diversity_weight: float = 0.01

    # EMA for atoms
    atom_ema_decay: float = 0.99

    # Codebook restart threshold
    codebook_restart_threshold: int = 1000

    # Autoresearch compatibility
    autoresearch_params: Optional[List[str]] = None

    def __post_init__(self):
        if self.autoresearch_params is None:
            self.autoresearch_params = [
                "rank_r",
                "codebook_k",
                "commitment_beta",
                "scale_lr_multiplier",
                "atom_diversity_weight",
                "atom_ema_decay",
            ]

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def effective_bpw(self) -> float:
        """Calculate approximate BPW for this config"""
        d = self.hidden_size
        r = self.rank_r
        k = self.codebook_k
        A = self.num_atoms_A
        b = self.xor_arity_b
        L = self.num_hidden_layers

        # Indices storage
        bits_indices = 2 * d * (k.bit_length() - 1)  # log2(k) bits per row

        # Scales storage (BF16)
        bits_scales = 2 * (d + d + r) * 16

        # Codebook amortized (atoms + combinations)
        bits_codebook_amortized = (A * r + k * b * A.bit_length()) / L

        total_bits = bits_indices + bits_scales + bits_codebook_amortized
        original_bits = d * d * 16

        return total_bits / original_bits

    @classmethod
    def nano(cls) -> "TRILIXConfig":
        """Nano config for RTX 3090 quick experiments"""
        return cls(
            hidden_size=1024,
            num_hidden_layers=16,
            num_attention_heads=16,
            num_key_value_heads=4,
            intermediate_size=4096,
            rank_r=64,
            codebook_k=64,
            num_atoms_A=16,
            xor_arity_b=2,
        )

    @classmethod
    def small(cls) -> "TRILIXConfig":
        """Small config for RTX 3090 serious training"""
        return cls(
            hidden_size=2048,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_key_value_heads=4,
            intermediate_size=8192,
            rank_r=100,
            codebook_k=128,
            num_atoms_A=32,
            xor_arity_b=3,
        )

    @classmethod
    def medium(cls) -> "TRILIXConfig":
        """Medium config for inference on RTX 3090, training on A100"""
        return cls(
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            intermediate_size=14336,
            rank_r=200,
            codebook_k=128,
            num_atoms_A=32,
            xor_arity_b=3,
        )
