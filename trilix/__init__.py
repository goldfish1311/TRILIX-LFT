"""
TRILIX-LFT: Triple-Level Indexed eXtreme Latent Factorized Transformer
Native 0.04-0.09 BPW training from scratch on RTX 3090
"""

__version__ = "1.0.0"

from .config import TRILIXConfig
from .layers import (
    TRILIXLinear,
    HebbianAtomResonance,
    DifferentiableAtomEvolver,
    FlatHierarchicalMoE,
    EmergentAgentSwarm,
    BeliefGate,
    ErrorDrivenHypernetwork,
    ReflectiveErrorLoop,
    TemperatureCascadeScheduler,
    # E1-E4: Innovations from Claude
    BinaryApproximateAttention,
    ShadowDistillationHead,
    AdaptiveRankSchedule,
    ConfidenceWeightedLoss,
)
from .model import TRILIXTransformer

__all__ = [
    "TRILIXConfig",
    "TRILIXLinear",
    "HebbianAtomResonance",
    "DifferentiableAtomEvolver",
    "FlatHierarchicalMoE",
    "EmergentAgentSwarm",
    "BeliefGate",
    "ErrorDrivenHypernetwork",
    "ReflectiveErrorLoop",
    "TemperatureCascadeScheduler",
    "TRILIXTransformer",
    # E1-E4
    "BinaryApproximateAttention",
    "ShadowDistillationHead",
    "AdaptiveRankSchedule",
    "ConfidenceWeightedLoss",
]
