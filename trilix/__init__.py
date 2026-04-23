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
    # E1-E4: Quality innovations
    BinaryApproximateAttention,
    ShadowDistillationHead,
    AdaptiveRankSchedule,
    ConfidenceWeightedLoss,
    # F1-F3: Architecture innovations
    HierarchicalPositionalAtomEncoding,
    CrossLayerAtomSharing,
    SpeculativeDecoder,
    # G1-G3: Final form innovations
    DiscreteSemanticAlgebra,
    LatentDiffusionCodebook,
    DynamicBPWAllocator,
    # H1-H5: Second-wave critical innovations
    MuonOptimizer,
    SequencePacker,
    CosineLoss,
    AGIWarmup,
    CodebookStatsTracker,
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
    # F1-F3
    "HierarchicalPositionalAtomEncoding",
    "CrossLayerAtomSharing",
    "SpeculativeDecoder",
    # G1-G3
    "DiscreteSemanticAlgebra",
    "LatentDiffusionCodebook",
    "DynamicBPWAllocator",
    # H1-H5: Second-wave critical innovations
    "MuonOptimizer",
    "SequencePacker",
    "CosineLoss",
    "AGIWarmup",
    "CodebookStatsTracker",
]
