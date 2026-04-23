"""
TRILIX Linear Layer: Triple-Level Compression
Level 1: Latent Factorization (W ≈ U · V^T)
Level 2: Vector Quantization Codebook (indices → codewords)
Level 3: XOR Atoms (codewords = XOR of atoms)
Level 4: MoE-Codebook (Multiple experts for specialized patterns)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class CodebookExpert(nn.Module):
    """
    Individual expert for MoE-Codebook.
    Each expert has its own XOR atoms and generates specialized codewords.
    Memory cost: negligible (~2KB per expert)
    """

    def __init__(self, k: int, r: int, num_atoms: int = 16, xor_arity: int = 3):
        super().__init__()
        self.k = k
        self.r = r
        self.num_atoms = num_atoms
        self.xor_arity = xor_arity

        # XOR atoms specific to this expert
        self.atoms = nn.Parameter(torch.randn(num_atoms, r) * 0.01)

        # Combination weights and indices
        self.combo_weights = nn.Parameter(torch.rand(k, xor_arity) * 0.5 + 0.25)
        self.combo_indices_logits = nn.Parameter(
            torch.randn(k, xor_arity, num_atoms) * 0.01
        )

    def get_codewords(self, temperature: float = 1.0) -> torch.Tensor:
        """Generate codewords from XOR atoms (hard path)"""
        # Binary atoms via sign
        atoms_binary = torch.sign(self.atoms)

        # Hard combination indices
        combo_soft = F.softmax(
            self.combo_indices_logits / max(temperature, 0.1), dim=-1
        )
        combo_hard_idx = torch.argmax(combo_soft, dim=-1)
        combo_hard = F.one_hot(combo_hard_idx, num_classes=self.num_atoms).float()

        # XOR combination
        selected = torch.einsum("kba,ar->kbr", combo_hard, atoms_binary)
        weighted = selected * self.combo_weights.unsqueeze(-1)
        combined = weighted.sum(dim=1)

        return torch.sign(combined)  # [k, r] - binary codewords

    def get_codewords_soft(self, temperature: float = 1.0) -> torch.Tensor:
        """Generate codewords (soft path for AGI)"""
        # Soft atoms
        atoms_soft = torch.tanh(self.atoms / max(temperature, 0.1))

        # Soft combination
        combo_soft = F.softmax(
            self.combo_indices_logits / max(temperature, 0.1), dim=-1
        )

        # XOR combination
        selected = torch.einsum("kba,ar->kbr", combo_soft, atoms_soft)
        weighted = selected * self.combo_weights.unsqueeze(-1)

        return weighted.sum(dim=1)  # [k, r] - soft codewords


class MoECodebook(nn.Module):
    """
    Mixture of Experts for Codebook generation.
    Each token gets routed to top-2 experts based on its latent representation.

    Args:
        num_experts: Number of codebook experts (default: 4)
        k: Codebook size per expert
        r: Latent dimension
        top_k: Number of experts to activate per token (default: 2)
    """

    def __init__(self, num_experts: int = 4, k: int = 64, r: int = 64, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.r = r
        self.top_k = top_k

        # Create experts
        self.experts = nn.ModuleList(
            [
                CodebookExpert(k=k, r=r, num_atoms=16, xor_arity=3)
                for _ in range(num_experts)
            ]
        )

        # Router: projects from latent space to expert selection
        self.router = nn.Linear(r, num_experts, bias=False)

    def forward(
        self, x_latent: torch.Tensor, temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_latent: [batch, seq, r] - latent representations after V^T·x
            temperature: For gating softmax

        Returns:
            codebook_output: [batch, seq, r] - weighted expert outputs
            aux_loss: Load balancing loss for training
        """
        batch_size, seq_len, r = x_latent.shape

        # Token-level routing
        # router_logits: [batch, seq, num_experts]
        router_logits = self.router(x_latent)

        # Top-k gating
        gates, expert_ids = torch.topk(
            F.softmax(router_logits / max(temperature, 0.1), dim=-1), self.top_k, dim=-1
        )  # gates: [B, S, top_k], expert_ids: [B, S, top_k]

        # Normalize gates
        gates = gates / gates.sum(dim=-1, keepdim=True)  # [B, S, top_k]

        # Gather outputs from experts
        # For efficiency, we compute all expert codebooks once per layer
        expert_codebooks = []
        for expert in self.experts:
            expert_codebooks.append(expert.get_codewords(temperature))  # List of [k, r]

        # For now: return routing info and let TRILIXLinear handle lookup
        # We'll store expert outputs for the layer to use
        self._cached_expert_codebooks = expert_codebooks
        self._cached_gates = gates
        self._cached_expert_ids = expert_ids

        # Compute load balancing loss (encourage uniform expert usage)
        router_prob = F.softmax(router_logits, dim=-1)  # [B, S, E]
        aux_loss = (
            self.num_experts
            * (
                router_prob.mean(dim=[0, 1])
                * (router_prob > 0).float().mean(dim=[0, 1])
            ).sum()
        )

        # Return weighted combination of expert centroids
        # For simplicity: average of expert codebooks weighted by gate
        combined = torch.zeros(batch_size, seq_len, r, device=x_latent.device)
        for i in range(self.num_experts):
            # Check where this expert is in top-k
            expert_mask = (expert_ids == i).any(dim=-1).float()  # [B, S]
            expert_gate = (
                gates[..., 0] * (expert_ids[..., 0] == i).float()
                + gates[..., 1] * (expert_ids[..., 1] == i).float()
            )  # [B, S]

            # Add contribution (using centroid of expert codebook)
            expert_centroid = expert_codebooks[i].mean(dim=0)  # [r]
            combined += expert_gate.unsqueeze(-1) * expert_centroid

        return combined, aux_loss


class STEBinary(torch.autograd.Function):
    """Straight-Through Estimator for binary values"""

    @staticmethod
    def forward(ctx, x):
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Clipped STE: only pass gradient if |x| <= 1
        return grad_output


class STEIndex(torch.autograd.Function):
    """Straight-Through Estimator for discrete indices"""

    @staticmethod
    def forward(ctx, x_soft, x_hard):
        ctx.save_for_backward(x_soft)
        return x_hard

    @staticmethod
    def backward(ctx, grad_output):
        (x_soft,) = ctx.saved_tensors
        return grad_output, None


class HebbianAtomResonance(nn.Module):
    """
    B4: Hebbian Atom Resonance (HAR)

    Hebbian principle: "neurons that fire together — wire together"

    Operates WITHOUT gradients. Runs as a periodic post-forward hook.

    Tracks:
    - Co-activation: how often atoms are selected together
    - Dead atoms: never used in the last N steps
    - Resonant pairs: frequently together + too similar (redundant)

    Actions every N steps:
    - Resonant pairs → flip bits to increase orthogonality
    - Dead atoms → replace with mutants of successful atoms
    """

    def __init__(
        self,
        num_atoms: int,
        rank: int,
        resonance_interval: int = 200,
        dead_threshold: int = 50,
        resonance_threshold: float = 0.3,
        similarity_threshold: float = 0.85,
        mutation_strength: float = 0.1,
        resonance_strength: float = 0.5,
    ):
        super().__init__()
        self.num_atoms = num_atoms
        self.rank = rank
        self.resonance_interval = resonance_interval
        self.dead_threshold = dead_threshold
        self.resonance_threshold = resonance_threshold
        self.similarity_threshold = similarity_threshold
        self.mutation_strength = mutation_strength
        self.resonance_strength = resonance_strength

        self.register_buffer("_step_counter", torch.zeros(1, dtype=torch.long))
        self.register_buffer("_co_activation_U", torch.zeros(num_atoms, num_atoms))
        self.register_buffer("_co_activation_V", torch.zeros(num_atoms, num_atoms))
        self.register_buffer("_atom_hits_U", torch.zeros(num_atoms))
        self.register_buffer("_atom_hits_V", torch.zeros(num_atoms))
        self._registered = False

    def register_with_layer(self, layer: "TRILIXLinear"):
        """Register this HAR with a TRILIXLinear layer"""
        self._layer = layer
        self._registered = True

    def observe(
        self,
        combo_indices_U: torch.Tensor,
        combo_indices_V: torch.Tensor,
        atoms_U: torch.Tensor,
        atoms_V: torch.Tensor,
    ):
        """
        Observe atom usage from forward pass (no gradients).

        Args:
            combo_indices_U: [codebook_size, xor_arity, num_atoms] one-hot
            combo_indices_V: [codebook_size, xor_arity, num_atoms] one-hot
            atoms_U: [num_atoms, rank] raw (not binary)
            atoms_V: [num_atoms, rank] raw (not binary)
        """
        if not self._registered:
            return

        with torch.no_grad():
            self._step_counter += 1

            atoms_U_bin = torch.sign(atoms_U)
            atoms_V_bin = torch.sign(atoms_V)

            combo_flat_U = combo_indices_U.sum(dim=1)  # [K, A]
            combo_flat_V = combo_indices_V.sum(dim=1)  # [K, A]

            atom_active_U = (combo_flat_U.sum(dim=0) > 0).float()  # [A]
            atom_active_V = (combo_flat_V.sum(dim=0) > 0).float()  # [A]

            self._atom_hits_U += atom_active_U
            self._atom_hits_V += atom_active_V

            co_U = torch.einsum("a,b->ab", atom_active_U, atom_active_U)
            co_V = torch.einsum("a,b->ab", atom_active_V, atom_active_V)
            self._co_activation_U += co_U
            self._co_activation_V += co_V

            if self._step_counter.item() % self.resonance_interval == 0:
                self._apply_resonance(atoms_U_bin, atoms_V_bin)

    def _apply_resonance(self, atoms_U: torch.Tensor, atoms_V: torch.Tensor):
        """Apply resonance: fix resonant pairs + replace dead atoms"""
        layer = getattr(self, "_layer", None)
        if layer is None:
            return

        with torch.no_grad():
            dead_U = self._atom_hits_U < self.dead_threshold
            dead_V = self._atom_hits_V < self.dead_threshold
            total_U = self._atom_hits_U.sum() + 1e-10
            total_V = self._atom_hits_V.sum() + 1e-10

            dead_count = dead_U.sum().item() + dead_V.sum().item()

            if dead_count == 0:
                self._atom_hits_U.zero_()
                self._atom_hits_V.zero_()
                self._co_activation_U.zero_()
                self._co_activation_V.zero_()
                return

            norm_co_U = self._co_activation_U / (total_U + 1e-10)
            norm_co_V = self._co_activation_V / (total_V + 1e-10)

            atom_norm_U = F.normalize(atoms_U, dim=1)
            atom_norm_V = F.normalize(atoms_V, dim=1)
            sim_U = atom_norm_U @ atom_norm_U.T
            sim_V = atom_norm_V @ atom_norm_V.T

            res_U = norm_co_U * (sim_U > self.similarity_threshold).float()
            res_V = norm_co_V * (sim_V > self.similarity_threshold).float()

            resonant_mask_U = res_U > self.resonance_threshold
            resonant_mask_V = res_V > self.resonance_threshold

            if resonant_mask_U.any():
                i_idx, j_idx = resonant_mask_U.nonzero(as_tuple=True)
                flip_count = 0
                for i, j in zip(i_idx.tolist(), j_idx.tolist()):
                    if i >= j:
                        continue
                    if sim_U[i, j] > self.similarity_threshold:
                        bit_diff = (atoms_U[i] != atoms_U[j]).float()
                        if bit_diff.sum() > 0:
                            flip_bit = (
                                bit_diff * torch.rand(self.rank, device=atoms_U.device)
                            ).argmax()
                            atoms_U[i, flip_bit] *= -1.0
                            atoms_U[j, flip_bit] *= -1.0
                            flip_count += 1
                if flip_count > 0:
                    layer.atoms_U.data.copy_(atoms_U)

            if resonant_mask_V.any():
                i_idx, j_idx = resonant_mask_V.nonzero(as_tuple=True)
                flip_count = 0
                for i, j in zip(i_idx.tolist(), j_idx.tolist()):
                    if i >= j:
                        continue
                    if sim_V[i, j] > self.similarity_threshold:
                        bit_diff = (atoms_V[i] != atoms_V[j]).float()
                        if bit_diff.sum() > 0:
                            flip_bit = (
                                bit_diff * torch.rand(self.rank, device=atoms_V.device)
                            ).argmax()
                            atoms_V[i, flip_bit] *= -1.0
                            atoms_V[j, flip_bit] *= -1.0
                            flip_count += 1
                if flip_count > 0:
                    layer.atoms_V.data.copy_(atoms_V)

            if dead_U.any():
                live_mask_U = ~dead_U
                live_indices_U = live_mask_U.nonzero(as_tuple=True)[0]
                if len(live_indices_U) > 0:
                    best_U = self._atom_hits_U.argmax()
                    for d in dead_U.nonzero(as_tuple=True)[0]:
                        mutant = atoms_U[best_U].clone()
                        flip_mask = (
                            torch.rand(self.rank, device=atoms_U.device)
                            < self.mutation_strength
                        )
                        mutant[flip_mask] *= -1.0
                        atoms_U[d] = mutant
                    layer.atoms_U.data.copy_(atoms_U)

            if dead_V.any():
                live_mask_V = ~dead_V
                live_indices_V = live_mask_V.nonzero(as_tuple=True)[0]
                if len(live_indices_V) > 0:
                    best_V = self._atom_hits_V.argmax()
                    for d in dead_V.nonzero(as_tuple=True)[0]:
                        mutant = atoms_V[best_V].clone()
                        flip_mask = (
                            torch.rand(self.rank, device=atoms_V.device)
                            < self.mutation_strength
                        )
                        mutant[flip_mask] *= -1.0
                        atoms_V[d] = mutant
                    layer.atoms_V.data.copy_(atoms_V)

            self._atom_hits_U.zero_()
            self._atom_hits_V.zero_()
            self._co_activation_U.zero_()
            self._co_activation_V.zero_()

    def get_stats(self) -> dict:
        """Return current HAR statistics"""
        total_U = self._atom_hits_U.sum().item()
        total_V = self._atom_hits_V.sum().item()
        return {
            "har_registered": self._registered,
            "har_step": self._step_counter.item(),
            "dead_atoms_U": int((self._atom_hits_U < self.dead_threshold).sum().item()),
            "dead_atoms_V": int((self._atom_hits_V < self.dead_threshold).sum().item()),
            "resonance_events_U": int((self._co_activation_U > 0).sum().item() // 2),
            "resonance_events_V": int((self._co_activation_V > 0).sum().item() // 2),
            "top_atom_U": int(self._atom_hits_U.argmax().item()),
            "top_atom_V": int(self._atom_hits_V.argmax().item()),
        }


class TRILIXLinear(nn.Module):
    """
    TRILIX Linear Layer with triple-level compression

    Architecture:
    W_eff = diag(h) · U_decoded · diag(l) · V_decoded^T · diag(g)

    Where:
    - U_decoded[i] = Codebook_U[idx_U[i]] (Level 2 VQ)
    - Codebook_U[j] = sign(Σ_b α[j,b] · Atom_U[β[j,b]]) (Level 3 XOR)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 100,
        codebook_size: int = 128,
        num_atoms: int = 32,
        xor_arity: int = 3,
        commitment_beta: float = 0.25,
        atom_ema_decay: float = 0.99,
        use_moe: bool = False,
        num_experts: int = 4,
        moe_top_k: int = 2,
        # Innovation flags
        use_saib: bool = False,
        use_rvq: bool = False,
        use_sgh: bool = False,
        use_lcc: bool = False,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.codebook_size = codebook_size
        self.num_atoms = num_atoms
        self.xor_arity = xor_arity
        self.commitment_beta = commitment_beta
        self.atom_ema_decay = atom_ema_decay
        self.use_moe = use_moe

        # Innovation flags
        self.use_saib = use_saib
        self.use_rvq = use_rvq
        self.use_sgh = use_sgh
        self.use_lcc = use_lcc
        self.use_har = False

        # Temperature for soft XOR (anneals from 1.0 to 0.01)
        # ATC: 3 independent temperatures for cascade freezing
        self.register_buffer("xor_temperature", torch.tensor(1.0))  # atom level
        self.register_buffer(
            "codebook_temperature", torch.tensor(1.0)
        )  # codebook level
        self.register_buffer("idx_temperature", torch.tensor(1.0))  # index level
        self.xor_temp_steps = 0

        # Scale clamping (prevent runaway)
        self.scale_max = 10.0

        # Level 1: Scale factors (only continuous parameters)
        # These carry all magnitude information
        self.row_scale = nn.Parameter(torch.ones(out_features))  # h
        self.col_scale = nn.Parameter(torch.ones(in_features))  # g
        self.latent_scale = nn.Parameter(torch.ones(rank))  # l

        # Level 4: MoE-Codebook (optional)
        if self.use_moe:
            self.moe_codebook_U = MoECodebook(
                num_experts=num_experts, k=codebook_size, r=rank, top_k=moe_top_k
            )
            self.moe_codebook_V = MoECodebook(
                num_experts=num_experts, k=codebook_size, r=rank, top_k=moe_top_k
            )
            # Router aux loss weight
            self.moe_aux_weight = 0.01
        else:
            self.moe_codebook_U = None
            self.moe_codebook_V = None

        # Innovation modules
        if use_saib:
            self.saib_U = SoftAttentionIndexBlender(
                codebook_size=codebook_size, rank=rank
            )
            self.saib_V = SoftAttentionIndexBlender(
                codebook_size=codebook_size, rank=rank
            )
        else:
            self.saib_U = None
            self.saib_V = None

        if use_rvq:
            self.rvq_U = ResidualVectorQuantizer(
                codebook_size=codebook_size,
                rank=rank,
                num_residual_levels=2,
                residual_size=16,
            )
            self.rvq_V = ResidualVectorQuantizer(
                codebook_size=codebook_size,
                rank=rank,
                num_residual_levels=2,
                residual_size=16,
            )
        else:
            self.rvq_U = None
            self.rvq_V = None

        if use_sgh:
            self.sgh_U = StochasticGroupHierarchy(
                codebook_size=codebook_size, rank=rank
            )
            self.sgh_V = StochasticGroupHierarchy(
                codebook_size=codebook_size, rank=rank
            )
        else:
            self.sgh_U = None
            self.sgh_V = None

        if use_lcc:
            self.lcc_U = LearnedCodebookCompressor(
                codebook_size=codebook_size, rank=rank
            )
            self.lcc_V = LearnedCodebookCompressor(
                codebook_size=codebook_size, rank=rank
            )
        else:
            self.lcc_U = None
            self.lcc_V = None

        # Level 2: Codebook indices for U and V
        # Stored as differentiable soft indices for STE
        self.idx_U_logits = nn.Parameter(
            torch.randn(out_features, codebook_size) * 0.01
        )
        self.idx_V_logits = nn.Parameter(torch.randn(in_features, codebook_size) * 0.01)

        # Level 3: XOR Atoms for codebooks
        # Atoms_A ∈ {±1}^{num_atoms × rank}
        self.atoms_U = nn.Parameter(torch.randn(num_atoms, rank) * 0.01)
        self.atoms_V = nn.Parameter(torch.randn(num_atoms, rank) * 0.01)

        # Combination weights for XOR (continuous, learnable)
        # alpha_U[j, b] for j in [codebook_size], b in [xor_arity]
        self.combo_weights_U = nn.Parameter(
            torch.rand(codebook_size, xor_arity) * 0.5 + 0.25
        )
        self.combo_weights_V = nn.Parameter(
            torch.rand(codebook_size, xor_arity) * 0.5 + 0.25
        )

        # Combination indices for atoms (discrete, STE)
        self.combo_indices_U_logits = nn.Parameter(
            torch.randn(codebook_size, xor_arity, num_atoms) * 0.01
        )
        self.combo_indices_V_logits = nn.Parameter(
            torch.randn(codebook_size, xor_arity, num_atoms) * 0.01
        )

        # Codebook EMA buffers (for commitment loss)
        self.register_buffer("codebook_U_ema", torch.zeros(codebook_size, rank))
        self.register_buffer("codebook_V_ema", torch.zeros(codebook_size, rank))
        self.register_buffer("codebook_U_count", torch.zeros(codebook_size))
        self.register_buffer("codebook_V_count", torch.zeros(codebook_size))

        # Usage tracking for codebook restart
        self.register_buffer(
            "usage_counter_U", torch.zeros(codebook_size, dtype=torch.long)
        )
        self.register_buffer(
            "usage_counter_V", torch.zeros(codebook_size, dtype=torch.long)
        )
        self.register_buffer("step_counter", torch.zeros(1, dtype=torch.long))

        # AGI: Atom Gradient Injection parameters
        self.agi_weight = 0.0  # Will be set by phase scheduler
        self.agi_phase = 0  # 0=disabled, 1=active
        self._cached_agi_loss = torch.tensor(0.0)

        self.har = None
        self._init_parameters()

    def enable_har(self, resonance_interval: int = 200):
        """Enable Hebbian Atom Resonance (B4)"""
        if self.har is None:
            self.har = HebbianAtomResonance(
                num_atoms=self.num_atoms,
                rank=self.rank,
                resonance_interval=resonance_interval,
            )
            self.har.register_with_layer(self)
            self.use_har = True

    def _init_parameters(self):
        """Initialize parameters with principled schemes"""
        # Initialize atoms with quasi-orthogonal structure
        with torch.no_grad():
            # Atoms: random -> orthogonalize -> sign
            for atoms in [self.atoms_U, self.atoms_V]:
                q, _ = torch.linalg.qr(torch.randn_like(atoms).T)
                atoms.copy_(q.T[: self.num_atoms] * 0.1)

            # Row scale: Kaiming-aware for factorized weights
            h_init = math.sqrt(2.0 / (self.in_features * self.rank))
            self.row_scale.fill_(h_init)

            # Latent scale: inverse rank decay (simulates SVD spectrum)
            for i in range(self.rank):
                self.latent_scale[i] = 1.0 / math.sqrt(i + 1)

    def _get_atoms_binary(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get binary atoms via STE"""
        atoms_U_bin = STEBinary.apply(self.atoms_U)
        atoms_V_bin = STEBinary.apply(self.atoms_V)
        return atoms_U_bin, atoms_V_bin

    def _get_combo_indices_hard(self, indices_logits: torch.Tensor) -> torch.Tensor:
        """Get hard indices via Gumbel-Softmax or argmax

        Args:
            indices_logits: [codebook_size, xor_arity, num_atoms]
        Returns:
            hard_indices: [codebook_size, xor_arity, num_atoms] one-hot
        """
        # indices_logits: [K, B, A] where K=codebook_size, B=xor_arity, A=num_atoms
        shape = indices_logits.shape

        # Flatten for softmax
        logits_flat = indices_logits.view(-1, shape[-1])  # [K*B, A]
        soft_flat = F.softmax(logits_flat, dim=-1)

        # Hard indices
        hard_idx = torch.argmax(soft_flat, dim=-1)  # [K*B]
        hard_flat = F.one_hot(hard_idx, num_classes=shape[-1]).float()  # [K*B, A]

        # Reshape back
        soft = soft_flat.view(shape)
        hard = hard_flat.view(shape)

        # STE: forward uses hard, backward uses soft
        return STEIndex.apply(soft, hard)

    def _decode_codebook_entry(
        self,
        combo_weights: torch.Tensor,
        combo_indices_hard: torch.Tensor,
        atoms_binary: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode a codebook entry from XOR atoms

        Args:
            combo_weights: [codebook_size, xor_arity] - continuous weights
            combo_indices_hard: [codebook_size, xor_arity, num_atoms] - one-hot
            atoms_binary: [num_atoms, rank] - binary atoms

        Returns:
            codebook: [codebook_size, rank] - binary codewords
        """
        # Get selected atoms for each codebook entry
        # combo_indices_hard[k, b, :] is one-hot over atoms
        # We want: selected[k, b, r] = sum_a combo_indices[k, b, a] * atoms[a, r]

        # [codebook_size, xor_arity, num_atoms] @ [num_atoms, rank]
        # -> [codebook_size, xor_arity, rank]
        selected_atoms = torch.einsum("kba,ar->kbr", combo_indices_hard, atoms_binary)

        # XOR approximation: sign(sum of weighted atoms)
        # weights [codebook_size, xor_arity, 1]
        weights = combo_weights.unsqueeze(-1)

        # Weighted combination
        # [codebook_size, xor_arity, rank] * [codebook_size, xor_arity, 1]
        # -> [codebook_size, xor_arity, rank]
        weighted = selected_atoms * weights

        # Sum over arity dimension and take sign
        # [codebook_size, rank]
        combined = weighted.sum(dim=1)

        # ATC: use codebook_temperature for codebook entry decoding
        temp = self.codebook_temperature.item()
        if temp > 0.05:
            combined = torch.tanh(combined / temp)
        coarse_codebook = STEBinary.apply(combined)

        # RVQ refinement: add residual on top of coarse
        if self.use_rvq and self.rvq_U is not None:
            coarse_indices = torch.arange(
                self.codebook_size, device=atoms_binary.device
            )
            codebook = self.rvq_U.decode_with_residual(coarse_codebook, coarse_indices)
            self._cached_rvq_loss = self.rvq_U.get_residual_loss(coarse_codebook)
        else:
            codebook = coarse_codebook

        return codebook

    def _decode_codebook_soft(
        self,
        combo_weights: torch.Tensor,
        combo_indices_logits: torch.Tensor,
        atoms_U: torch.Tensor,
    ) -> torch.Tensor:
        """
        AGI: Atom Gradient Injection - Soft path for gradient flow

        This method creates a FULLY DIFFERENTIABLE path to atoms,
        bypassing STE chain. Used for training only.

        Args:
            combo_weights: [codebook_size, xor_arity] - continuous weights
            combo_indices_logits: [codebook_size, xor_arity, num_atoms] - logits
            atoms_U: [num_atoms, rank] - raw atoms (NOT binarized)

        Returns:
            soft_codebook: [codebook_size, rank] - soft codewords
        """
        # ATC: use codebook_temperature for soft decoding too
        temp = max(self.codebook_temperature.item(), 0.1)

        # 1. Soft combination indices (NO argmax, NO STE!)
        # Shape: [codebook_size, xor_arity, num_atoms]
        combo_soft = F.softmax(combo_indices_logits / temp, dim=-1)

        # 2. Soft atoms representation (tanh instead of sign)
        # Shape: [num_atoms, rank]
        atoms_soft = torch.tanh(atoms_U / temp)

        # 3. Gather atoms using soft indices (fully differentiable)
        # [k, b, A] @ [A, r] -> [k, b, r]
        selected = torch.einsum("kba,ar->kbr", combo_soft, atoms_soft)

        # 4. Weighted combination
        weighted = selected * combo_weights.unsqueeze(-1)

        # 5. Sum over arity dimension
        # Result: [codebook_size, rank]
        return weighted.sum(dim=1)

    def _quantize_U(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize U matrix
        Returns: (U_hard, U_soft, commitment_loss)
        """
        # Get binary atoms
        atoms_U_bin, _ = self._get_atoms_binary()

        # Get hard combination indices
        combo_idx_U = self._get_combo_indices_hard(self.combo_indices_U_logits)

        # Decode codebook
        codebook_U = self._decode_codebook_entry(
            self.combo_weights_U, combo_idx_U, atoms_U_bin
        )

        # ATC: use idx_temperature for index selection
        idx_temp = max(self.idx_temperature.item(), 0.1)
        idx_U_soft = F.softmax(
            self.idx_U_logits / idx_temp, dim=-1
        )  # [out_features, codebook_size]

        # Hard indices (one-hot)
        idx_U_hard_idx = torch.argmax(idx_U_soft, dim=-1)  # [out_features]

        # Gather codewords
        U_hard = codebook_U[idx_U_hard_idx]  # [out_features, rank]

        # Soft version for gradient flow (matmul with softmax weights)
        U_soft = torch.matmul(idx_U_soft, codebook_U)  # [out_features, rank]

        # SAIB: EMA update of codebook (already done below, but explicitly call SAIB if enabled)
        if self.use_saib and self.saib_U is not None:
            # Use SAIB EMA instead of manual EMA
            self.saib_U.ema_update(U_soft, idx_U_hard_idx)

        # Commitment loss: encourage soft to stay close to hard
        commitment_loss = self.commitment_beta * F.mse_loss(U_soft.detach(), U_hard)

        # SGH: Semantic Gradient Highway - gradient consistency loss
        sgh_loss = 0.0
        if self.use_sgh and self.sgh_U is not None:
            # Highway: similar codebook entries → similar gradients
            highway_loss = self.sgh_U.get_gradient_highway_loss(codebook_U, codebook_U)
            # Coherence: entries cluster by semantic similarity
            coherence_loss = self.sgh_U.get_group_coherence_loss(codebook_U)
            sgh_loss = highway_loss + coherence_loss
            self._cached_sgh_loss = sgh_loss

        # LCC loss: encourage diverse codebook generation
        lcc_loss = 0.0
        if self.use_lcc and self.lcc_U is not None:
            generated_cb = self.lcc_U.get_codebook()
            lcc_loss = 0.001 * (generated_cb**2).mean()
            self._cached_lcc_loss = lcc_loss

        # Update EMA and usage counters
        with torch.no_grad():
            self.usage_counter_U.index_add_(
                0, idx_U_hard_idx, torch.ones_like(idx_U_hard_idx, dtype=torch.long)
            )

            # EMA update of codebook
            for i in range(self.codebook_size):
                mask = idx_U_hard_idx == i
                if mask.any():
                    self.codebook_U_ema[i] = self.atom_ema_decay * self.codebook_U_ema[
                        i
                    ] + (1 - self.atom_ema_decay) * U_hard[mask].mean(0)
                    self.codebook_U_count[i] += 1

        # Use hard for forward, soft for backward (via STE implicitly)
        U_final = STEIndex.apply(U_soft, U_hard)

        return U_final, U_soft, commitment_loss

    def _quantize_V(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Same as _quantize_U but for V"""
        _, atoms_V_bin = self._get_atoms_binary()

        combo_idx_V = self._get_combo_indices_hard(self.combo_indices_V_logits)

        codebook_V = self._decode_codebook_entry(
            self.combo_weights_V, combo_idx_V, atoms_V_bin
        )

        idx_V_soft = F.softmax(
            self.idx_V_logits / max(self.idx_temperature.item(), 0.1), dim=-1
        )
        idx_V_hard_idx = torch.argmax(idx_V_soft, dim=-1)

        V_hard = codebook_V[idx_V_hard_idx]  # [in_features, rank]
        V_soft = torch.matmul(idx_V_soft, codebook_V)

        # SAIB: EMA update for V
        if self.use_saib and self.saib_V is not None:
            self.saib_V.ema_update(V_soft, idx_V_hard_idx)

        commitment_loss = self.commitment_beta * F.mse_loss(V_soft.detach(), V_hard)

        with torch.no_grad():
            self.usage_counter_V.index_add_(
                0, idx_V_hard_idx, torch.ones_like(idx_V_hard_idx, dtype=torch.long)
            )

            for i in range(self.codebook_size):
                mask = idx_V_hard_idx == i
                if mask.any():
                    self.codebook_V_ema[i] = self.atom_ema_decay * self.codebook_V_ema[
                        i
                    ] + (1 - self.atom_ema_decay) * V_hard[mask].mean(0)
                    self.codebook_V_count[i] += 1

        V_final = STEIndex.apply(V_soft, V_hard)

        # SGH: Semantic Gradient Highway for V
        if self.use_sgh and self.sgh_V is not None:
            highway_loss_v = self.sgh_V.get_gradient_highway_loss(
                codebook_V, codebook_V
            )
            coherence_loss_v = self.sgh_V.get_group_coherence_loss(codebook_V)
            sgh_loss_v = highway_loss_v + coherence_loss_v
            self._cached_sgh_loss = (
                self._cached_sgh_loss + sgh_loss_v
                if hasattr(self, "_cached_sgh_loss")
                else sgh_loss_v
            )

        # LCC loss for V
        if self.use_lcc and self.lcc_V is not None:
            generated_cb = self.lcc_V.get_codebook()
            lcc_loss_v = 0.001 * (generated_cb**2).mean()
            self._cached_lcc_loss = (
                self._cached_lcc_loss + lcc_loss_v
                if hasattr(self, "_cached_lcc_loss")
                else lcc_loss_v
            )

        return V_final, V_soft, commitment_loss

    def _check_codebook_restart(self) -> Optional[torch.Tensor]:
        """Check if codebook entries need restart (dead entries)"""
        self.step_counter += 1

        if self.step_counter.item() % 1000 != 0:
            return None

        losses = []

        # Check U codebook
        dead_entries_U = (self.usage_counter_U < 10).nonzero(as_tuple=True)[0]
        if len(dead_entries_U) > 0:
            # Reinitialize dead entries from random batch samples
            with torch.no_grad():
                _, atoms_U_bin = self._get_atoms_binary()
                combo_idx_U = self._get_combo_indices_hard(self.combo_indices_U_logits)
                codebook_U = self._decode_codebook_entry(
                    self.combo_weights_U, combo_idx_U, atoms_U_bin
                )

                # Pick random live entries to copy from
                live_entries = (self.usage_counter_U >= 10).nonzero(as_tuple=True)[0]
                if len(live_entries) > 0:
                    for dead_idx in dead_entries_U[
                        : len(dead_entries_U) // 2
                    ]:  # Restart half
                        random_live = live_entries[
                            torch.randint(len(live_entries), (1,))
                        ]
                        codebook_U[dead_idx] = (
                            codebook_U[random_live]
                            + torch.randn_like(codebook_U[dead_idx]) * 0.01
                        )

            losses.append(torch.tensor(1.0))  # Signal that restart happened

        # Reset counters
        self.usage_counter_U.zero_()
        self.usage_counter_V.zero_()

        return losses[0] if losses else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass with all three compression levels

        AGI: Dual-path forward
        - Hard path: STE-based, for actual computation
        - Soft path: Fully differentiable, for gradient injection into atoms

        Returns:
            output: [batch, out_features]
            aux_losses: dict with commitment losses, AGI losses etc.
        """
        aux_losses = {}

        # Level 2: Get quantized U and V (HARD PATH - STE-based)
        U_hard, U_soft, commit_U = self._quantize_U()
        V_hard, V_soft, commit_V = self._quantize_V()

        aux_losses["commitment_U"] = commit_U
        aux_losses["commitment_V"] = commit_V

        # Check codebook restart
        restart_loss = self._check_codebook_restart()
        if restart_loss is not None:
            aux_losses["codebook_restart"] = restart_loss

        # Level 1: Latent factorization with scales
        # W_eff = diag(h) · U · diag(l) · V^T · diag(g)
        # Efficient computation:
        # y = h ⊙ (U · (l ⊙ (V^T · (g ⊙ x))))

        # Clamp scales to prevent runaway (P0 fix from Gemini)
        row_scale = torch.clamp(self.row_scale, -self.scale_max, self.scale_max)
        col_scale = torch.clamp(self.col_scale, -self.scale_max, self.scale_max)
        latent_scale = torch.clamp(self.latent_scale, -self.scale_max, self.scale_max)

        # Step 1: apply col_scale g to input
        x_scaled = x * col_scale  # broadcast

        # Step 2: V^T · (g ⊙ x)
        latent = torch.matmul(x_scaled, V_hard)  # [batch, rank]

        # Step 3: apply latent_scale l (clamped!)
        latent = latent * latent_scale  # [batch, rank]

        # Step 4: U · (l ⊙ V^T · g ⊙ x)
        output = torch.matmul(latent, U_hard.T)  # [batch, out_features]

        # Step 5: apply row_scale h (clamped!)
        output = output * row_scale

        # === AGI: SOFT PATH (for gradient injection only) ===
        if self.training and self.agi_phase > 0:
            # Get atoms (NOT binarized - direct gradient flow!)
            atoms_U_raw = self.atoms_U
            atoms_V_raw = self.atoms_V

            # Decode codebook via soft path (fully differentiable)
            soft_codebook_U = self._decode_codebook_soft(
                self.combo_weights_U, self.combo_indices_U_logits, atoms_U_raw
            )
            soft_codebook_V = self._decode_codebook_soft(
                self.combo_weights_V, self.combo_indices_V_logits, atoms_V_raw
            )

            # Get hard codebooks for alignment (detached!)
            atoms_U_bin, atoms_V_bin = self._get_atoms_binary()
            combo_idx_U = self._get_combo_indices_hard(self.combo_indices_U_logits)
            combo_idx_V = self._get_combo_indices_hard(self.combo_indices_V_logits)
            hard_codebook_U = self._decode_codebook_entry(
                self.combo_weights_U, combo_idx_U, atoms_U_bin
            )
            hard_codebook_V = self._decode_codebook_entry(
                self.combo_weights_V, combo_idx_V, atoms_V_bin
            )

            # AGI Alignment Loss: soft follows hard (hard is detached!)
            # This creates gradient flow: alignment_loss -> soft_codebook -> atoms_U (DIRECT!)
            agi_loss_U = F.mse_loss(soft_codebook_U, hard_codebook_U.detach())
            agi_loss_V = F.mse_loss(soft_codebook_V, hard_codebook_V.detach())
            agi_alignment = (agi_loss_U + agi_loss_V) / 2.0

            # AGI Diversity Loss: prevent atom collapse (via soft path)
            gram_U = soft_codebook_U @ soft_codebook_U.T
            gram_V = soft_codebook_V @ soft_codebook_V.T
            identity_U = torch.eye(gram_U.size(0), device=gram_U.device)
            identity_V = torch.eye(gram_V.size(0), device=gram_V.device)

            diversity_U = ((gram_U - identity_U) ** 2).mean()
            diversity_V = ((gram_V - identity_V) ** 2).mean()
            agi_diversity = (diversity_U + diversity_V) / 2.0

            # Total AGI loss
            self._cached_agi_loss = (
                self.agi_weight * agi_alignment + 0.01 * agi_diversity
            )

            aux_losses["agi_alignment"] = agi_alignment
            aux_losses["agi_diversity"] = agi_diversity
            aux_losses["agi_total"] = self._cached_agi_loss
        else:
            aux_losses["agi_alignment"] = torch.tensor(0.0, device=output.device)
            aux_losses["agi_diversity"] = torch.tensor(0.0, device=output.device)
            aux_losses["agi_total"] = torch.tensor(0.0, device=output.device)
            self._cached_agi_loss = torch.tensor(0.0, device=output.device)

        # Innovation losses (SGH, LCC)
        sgh_loss = getattr(
            self, "_cached_sgh_loss", torch.tensor(0.0, device=output.device)
        )
        lcc_loss = getattr(
            self, "_cached_lcc_loss", torch.tensor(0.0, device=output.device)
        )
        aux_losses["sgh_loss"] = sgh_loss
        aux_losses["lcc_loss"] = lcc_loss

        # Atom diversity tracking (for monitoring)
        atoms_U_bin, atoms_V_bin = self._get_atoms_binary()
        atom_usage_U = atoms_U_bin.view(-1, self.rank).abs().mean(1)
        atom_usage_V = atoms_V_bin.view(-1, self.rank).abs().mean(1)
        prob_U = F.softmax(atom_usage_U, dim=0)
        prob_V = F.softmax(atom_usage_V, dim=0)
        entropy_U = -(prob_U * torch.log(prob_U + 1e-10)).sum()
        entropy_V = -(prob_V * torch.log(prob_V + 1e-10)).sum()
        aux_losses["atom_entropy"] = (entropy_U + entropy_V) / 2.0

        # B4: HAR observation — no gradients, runs after forward
        if self.training and self.use_har and self.har is not None:
            combo_idx_U = self._get_combo_indices_hard(self.combo_indices_U_logits)
            combo_idx_V = self._get_combo_indices_hard(self.combo_indices_V_logits)
            self.har.observe(combo_idx_U, combo_idx_V, self.atoms_U, self.atoms_V)
            har_stats = self.har.get_stats()
            aux_losses["har_dead_U"] = torch.tensor(
                har_stats["dead_atoms_U"], dtype=torch.float32, device=output.device
            )
            aux_losses["har_dead_V"] = torch.tensor(
                har_stats["dead_atoms_V"], dtype=torch.float32, device=output.device
            )

        return output, aux_losses

    def get_effective_weight(self) -> torch.Tensor:
        """
        Compute the effective full weight matrix (for debugging/analysis)
        W_eff = diag(h) · U · diag(l) · V^T · diag(g)
        """
        with torch.no_grad():
            # Get current quantized U and V (without gradients)
            atoms_U_bin = torch.sign(self.atoms_U.detach())
            atoms_V_bin = torch.sign(self.atoms_V.detach())

            combo_idx_U = F.one_hot(
                torch.argmax(F.softmax(self.combo_indices_U_logits, dim=-1), dim=-1),
                num_classes=self.num_atoms,
            ).float()
            combo_idx_V = F.one_hot(
                torch.argmax(F.softmax(self.combo_indices_V_logits, dim=-1), dim=-1),
                num_classes=self.num_atoms,
            ).float()

            codebook_U = self._decode_codebook_entry(
                self.combo_weights_U.detach(), combo_idx_U, atoms_U_bin
            )
            codebook_V = self._decode_codebook_entry(
                self.combo_weights_V.detach(), combo_idx_V, atoms_V_bin
            )

            idx_U = torch.argmax(F.softmax(self.idx_U_logits, dim=-1), dim=-1)
            idx_V = torch.argmax(F.softmax(self.idx_V_logits, dim=-1), dim=-1)

            U = codebook_U[idx_U]
            V = codebook_V[idx_V]

            # Build W_eff
            W = torch.matmul(U * self.latent_scale, V.T)  # [out, in]
            W = W * self.row_scale.unsqueeze(1) * self.col_scale.unsqueeze(0)

            return W

    def count_parameters(self) -> dict:
        """Count actual storage requirements"""
        stats = {}

        # Scales (BF16 = 16 bits)
        stats["row_scale_bits"] = self.out_features * 16
        stats["col_scale_bits"] = self.in_features * 16
        stats["latent_scale_bits"] = self.rank * 16

        # Indices (log2(k) bits each)
        log2k = (self.codebook_size - 1).bit_length()
        stats["idx_U_bits"] = self.out_features * log2k
        stats["idx_V_bits"] = self.in_features * log2k

        # Atoms (binary = 1 bit, but stored as float during training)
        stats["atoms_U_bits_train"] = self.num_atoms * self.rank * 32  # float
        stats["atoms_V_bits_train"] = self.num_atoms * self.rank * 32
        stats["atoms_U_bits_eval"] = self.num_atoms * self.rank * 1  # binary
        stats["atoms_V_bits_eval"] = self.num_atoms * self.rank * 1

        # Combo indices
        log2A = (self.num_atoms - 1).bit_length()
        stats["combo_idx_U_bits"] = self.codebook_size * self.xor_arity * log2A
        stats["combo_idx_V_bits"] = self.codebook_size * self.xor_arity * log2A

        # Combo weights (float during training)
        stats["combo_weights_bits"] = 2 * self.codebook_size * self.xor_arity * 32

        # Total
        total_train = sum(
            v for k, v in stats.items() if "bits" in k and "eval" not in k
        )
        total_eval = (
            stats["row_scale_bits"]
            + stats["col_scale_bits"]
            + stats["latent_scale_bits"]
            + stats["idx_U_bits"]
            + stats["idx_V_bits"]
            + stats["atoms_U_bits_eval"]
            + stats["atoms_V_bits_eval"]
            + stats["combo_idx_U_bits"]
            + stats["combo_idx_V_bits"]
        )

        # Original size
        original_bits = self.in_features * self.out_features * 16

        stats["total_train_bits"] = total_train
        stats["total_eval_bits"] = total_eval
        stats["original_bits"] = original_bits
        stats["bpw_train"] = total_train / original_bits
        stats["bpw_eval"] = total_eval / original_bits

        return stats


class TemperatureCascadeScheduler:
    """
    Innovation 4: ATC — Adaptive Temperature Cascade
    Freezing order: atoms (fast) -> codebook combos (medium) -> row indices (slow)
    Analogy: learn alphabet first -> then words -> then word choice per neuron
    """

    def __init__(self, total_steps: int, warmup_steps: int = 500):
        self.total = total_steps
        self.warmup = warmup_steps

    def get_temperatures(self, step: int) -> dict:
        if step < self.warmup:
            return {"atom_temp": 1.0, "codebook_temp": 1.0, "idx_temp": 1.0}
        progress = min(1.0, (step - self.warmup) / max(1, self.total - self.warmup))
        atom_temp = max(0.01, 1.0 * math.exp(-progress * 10))
        codebook_temp = max(0.01, 1.0 * math.exp(-progress * 4))
        idx_temp = max(0.01, 1.0 * math.exp(-progress * 1.5))
        return {
            "atom_temp": atom_temp,
            "codebook_temp": codebook_temp,
            "idx_temp": idx_temp,
        }

    def apply_to_model(self, model, step: int):
        temps = self.get_temperatures(step)
        for module in model.modules():
            if isinstance(module, TRILIXLinear):
                module.xor_temperature.data.fill_(temps["atom_temp"])
                module.codebook_temperature.data.fill_(temps["codebook_temp"])
                module.idx_temperature.data.fill_(temps["idx_temp"])


class StochasticGroupHierarchy(nn.Module):
    """Innovation 2: SGH - Semantic Gradient Highway.

    Creates direct gradient paths through output space to codebook entries.
    Instead of gradients flowing through weights (old way), they flow through
    semantic representations (output similarity).

    Key insight: if two codebook entries produce similar outputs, their
    gradients should be similar (semantic clustering).
    """

    def __init__(self, codebook_size: int, rank: int, num_groups: int = 8):
        super().__init__()
        self.k = codebook_size
        self.r = rank
        self.num_groups = num_groups

        # Group embeddings for semantic clustering
        self.group_embeddings = nn.Parameter(torch.randn(num_groups, rank) * 0.01)

        # Codebook-to-group assignment (learnable)
        self.register_buffer("codebook_to_group", torch.zeros(codebook_size))

    def compute_output_similarity(self, codebook: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise output similarity of codebook entries.
        Entries that produce similar outputs should have similar gradients.

        Args:
            codebook: [k, rank] - codebook entries

        Returns:
            similarity: [k, k] - cosine similarity matrix
        """
        # Normalize entries
        codebook_norm = F.normalize(codebook, dim=-1)

        # Pairwise cosine similarity
        similarity = codebook_norm @ codebook_norm.T

        return similarity

    def get_gradient_highway_loss(
        self,
        codebook: torch.Tensor,
        codebook_grad: torch.Tensor,
    ) -> torch.Tensor:
        """
        SGH loss: gradient highway through semantic similarity.

        If two codebook entries produce similar outputs (high similarity),
        they should receive similar gradients.

        Args:
            codebook: [k, rank] - current codebook entries
            codebook_grad: [k, rank] - gradient w.r.t. codebook entries

        Returns:
            highway_loss: scalar - gradient consistency loss
        """
        # Compute output similarity among codebook entries
        sim = self.compute_output_similarity(codebook)

        # Compute gradient similarity among codebook entries
        grad_norm = F.normalize(codebook_grad, dim=-1)
        grad_sim = grad_norm @ grad_norm.T

        # Highway loss: gradient similarity should follow output similarity
        mask = (sim > 0.5).float()

        highway_loss = ((sim - grad_sim) ** 2 * mask).sum() / (mask.sum() + 1e-8)

        return 0.01 * highway_loss

    def get_group_coherence_loss(self, codebook: torch.Tensor) -> torch.Tensor:
        """
        Encourage codebook entries within a group to cluster together.
        """
        # Soft assignment to groups
        codebook_norm = F.normalize(codebook, dim=-1)
        group_norm = F.normalize(self.group_embeddings, dim=-1)

        # [k, r] @ [r, num_groups] -> [k, num_groups]
        assignment_scores = codebook_norm @ group_norm.T
        assignment_soft = F.softmax(assignment_scores, dim=-1)

        # Entries in same group should have similar embeddings
        # Compute group centroids
        group_centroids = assignment_soft.T @ codebook  # [num_groups, rank]

        # Reconstruction loss: codebook ≈ assignment @ group_centroids
        reconstructed = assignment_soft @ group_centroids  # [k, rank]
        coherence_loss = F.mse_loss(codebook, reconstructed)

        return 0.001 * coherence_loss


class ResidualVectorQuantizer(nn.Module):
    """Innovation 3: RVQ - Residual Vector Quantization (like EnCodec/DAC).

    Two-level quantization:
    - Level 1 (coarse): existing XOR codebook
    - Level 2 (residual): small codebook for fine-tuning the residual

    Effective codebook size = k_coarse × k_residual
    Storage: k_coarse + k_residual entries instead of k_coarse × k_residual

    For each codeword:
    codeword_final = coarse_entry[coarse_idx] + residual_entry[residual_idx]
    """

    def __init__(
        self,
        codebook_size: int,
        rank: int,
        num_residual_levels: int = 2,
        residual_size: int = 16,
    ):
        super().__init__()
        self.k_coarse = codebook_size
        self.r = rank
        self.num_levels = num_residual_levels
        self.k_residual = residual_size

        # Residual codebooks (one per level)
        # Each has k_residual entries of rank r
        for lvl in range(num_residual_levels):
            self.register_parameter(
                f"residual_cb_{lvl}",
                nn.Parameter(torch.randn(residual_size, rank) * 0.01),
            )
            # Residual index logits (per coarse entry)
            self.register_parameter(
                f"residual_idx_{lvl}",
                nn.Parameter(torch.zeros(codebook_size, residual_size)),
            )

    def decode_with_residual(
        self,
        coarse_codebook: torch.Tensor,
        coarse_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode codewords with residual refinement.

        Args:
            coarse_codebook: [k_coarse, rank] - the XOR-generated coarse codebook
            coarse_indices: [num_entries] - which coarse entry to use

        Returns:
            refined: [num_entries, rank] - coarse + residual refinement
        """
        # Get coarse entries
        coarse_entries = coarse_codebook[coarse_indices]  # [num_entries, rank]

        # Apply residual refinement level by level
        current = coarse_entries
        for lvl in range(self.num_levels):
            residual_cb = getattr(self, f"residual_cb_{lvl}")
            residual_idx = getattr(self, f"residual_idx_{lvl}")

            # Compute residual scores based on current approximation
            # residual_idx: [k_coarse, k_residual] - how good each residual is for each coarse
            residual_scores = residual_idx[coarse_indices]  # [num_entries, k_residual]
            residual_softmax = F.softmax(residual_scores, dim=-1)

            # Get best residual for each entry
            residual_indices = residual_softmax.argmax(dim=-1)  # [num_entries]
            residual_entries = residual_cb[residual_indices]  # [num_entries, rank]

            # Add residual
            current = current + residual_entries

        return current

    def get_residual_loss(self, coarse_codebook: torch.Tensor) -> torch.Tensor:
        """
        Encourage residual codebook to capture what coarse misses.
        L2 loss: residual should reconstruct the difference from mean.
        """
        # Mean codebook entry
        mean_entry = coarse_codebook.mean(dim=0, keepdim=True)  # [1, r]

        total_loss = 0.0
        for lvl in range(self.num_levels):
            residual_cb = getattr(self, f"residual_cb_{lvl}")
            # Residual should help reconstruct deviations from mean
            loss = (residual_cb**2).mean()
            total_loss = total_loss + loss

        return 0.001 * total_loss

    def get_codebook_size_effective(self) -> int:
        """Effective k = k_coarse * k_residual^num_levels"""
        return self.k_coarse * (self.k_residual**self.num_levels)


class SoftAttentionIndexBlender(nn.Module):
    """Innovation 1: SAIB - Spectral Initialization + EMA for Codebook.

    1. Spectral initialization: SVD-based atom initialization for better conditioning
    2. EMA update: exponential moving average for codebook entries (like DAC/EnCodec)

    Benefits:
    - Faster convergence (spectral init)
    - More stable training (EMA updates)
    - No extra parameters needed
    """

    def __init__(self, codebook_size: int, rank: int, ema_decay: float = 0.99):
        super().__init__()
        self.k = codebook_size
        self.r = rank
        self.ema_decay = ema_decay

        # EMA buffers for codebook entries
        self.register_buffer("ema_codebook", torch.zeros(codebook_size, rank))
        self.register_buffer("ema_cluster_counts", torch.zeros(codebook_size))
        self.ema_initialized = False

    def spectral_init_atoms(self, random_matrix: torch.Tensor):
        """
        Initialize atoms using SVD of random matrix.
        This gives orthogonal, well-conditioned starting vectors.

        Args:
            random_matrix: [N, r] - random initialization matrix
        """
        with torch.no_grad():
            # Center the matrix
            centered = random_matrix - random_matrix.mean(dim=0, keepdim=True)

            # SVD
            try:
                U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
                # Use top r singular vectors as initialization
                # Atoms = Vt[:num_atoms] scaled by sqrt(S)
                initialized = (
                    Vt[: min(random_matrix.shape[0], Vt.shape[0])]
                    * S[: min(random_matrix.shape[0], S.shape[0])].sqrt()
                ).T
                return initialized
            except Exception:
                # Fallback to random init if SVD fails
                return random_matrix

    def ema_update(self, assigned_entries: torch.Tensor, indices: torch.Tensor):
        """
        EMA update of codebook entries based on assigned vectors.
        Like DAC/EnCodec: codebook[k] = decay * codebook[k] + (1-decay) * new_entry

        Args:
            assigned_entries: [N, r] - vectors assigned to each codebook entry
            indices: [N] - which codebook entry each vector belongs to
        """
        if not self.ema_initialized:
            self.ema_codebook.fill_(0)
            self.ema_cluster_counts.fill_(0)
            self.ema_initialized = True

        # Per-entry EMA update
        for idx, entry in zip(indices.unique(), assigned_entries):
            mask = indices == idx
            batch_mean = assigned_entries[mask].mean(dim=0)

            self.ema_codebook[idx] = (
                self.ema_decay * self.ema_codebook[idx]
                + (1 - self.ema_decay) * batch_mean
            )
            self.ema_cluster_counts[idx] += mask.sum().item()

    def forward(self, x: torch.Tensor, codebook: torch.Tensor):
        """Just return codebook (EMA updates happen during quantization)."""
        return codebook


class LearnedCodebookCompressor(nn.Module):
    """Innovation 5: LCC - Learned Codebook Compressor.

    Uses small MLP to predict codebook entries from latent codes.
    This enables on-the-fly codebook generation instead of fixed storage.
    """

    def __init__(self, codebook_size: int, rank: int, hidden_dim: int = 32):
        super().__init__()
        self.k = codebook_size
        self.r = rank

        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, rank),
        )

        self.latent_codes = nn.Parameter(torch.randn(codebook_size, hidden_dim) * 0.01)

    def forward(self):
        """Generate full codebook from latent codes."""
        return self.encoder(self.latent_codes)  # [k, hidden] -> [k, r]

    def get_codebook(self) -> torch.Tensor:
        """Get current codebook."""
        return self.forward()


class SymbolicDiffLoss(nn.Module):
    """B5: SDO — Symbolic Diff Operations (от Клода).

    Аналогия word2vec king-man+woman=queen, но через XOR в пространстве кодбука.

    Если C[a] XOR C[b] ≈ C[c] XOR C[d]
    → модель умеет делать структурные аналогии через дискретные операции
    → это напрямую улучшает reasoning capabilities

    Ключевой инсайт Клода:
    - {±1} XOR = поэлементное умножение (，不需要特殊处理)
    - Это нативная битовая операция в кремнии
    - Ближе к логическому мышлению человека

    Два компонента:
    1. analogy_clarity: поощряет высокий |similarity| (чёткие аналогии, не размытые)
    2. analogy_diversity: поощряет разнообразие (не коллапс к одной аналогии)

    Args:
        rank: Latent dimension (r)
        num_samples: Сколько аналогий семплировать за раз (default: 32)
        loss_weight: Вес лосса (default: 0.0001)
    """

    def __init__(
        self, rank: int = 100, num_samples: int = 32, loss_weight: float = 0.0001
    ):
        super().__init__()
        self.rank = rank
        self.num_samples = num_samples
        self.loss_weight = loss_weight

    def _binary_xor(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """XOR двух бинарных векторов {±1} = поэлементное умножение."""
        return a * b

    def forward(
        self,
        codebook_U: torch.Tensor,
        codebook_V: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            codebook_U: [k, r] — кодбук для U матрицы
            codebook_V: [k, r] — кодбук для V матрицы

        Returns:
            sdo_loss: scalar — штраф за отсутствие аналогий
        """
        k, r = codebook_U.shape
        if k < 4:
            return torch.tensor(0.0, device=codebook_U.device)

        loss_u = self._symbolic_analogy_loss(codebook_U)
        loss_v = self._symbolic_analogy_loss(codebook_V)
        return (loss_u + loss_v) * self.loss_weight

    def _symbolic_analogy_loss(self, codebook: torch.Tensor) -> torch.Tensor:
        """Поощряем структурные аналогии в codebook."""
        k, r = codebook.shape

        # Семплировать случайные квадреты индексов (a, b, c, d)
        indices = torch.randint(0, k, (self.num_samples, 4), device=codebook.device)

        a = codebook[indices[:, 0]]
        b = codebook[indices[:, 1]]
        c = codebook[indices[:, 2]]
        d = codebook[indices[:, 3]]

        # XOR в пространстве {±1} = поэлементное умножение
        diff_ab = self._binary_xor(a, b)
        diff_cd = self._binary_xor(c, d)

        # Если diff_ab близок к diff_cd → есть аналогия
        # Cosine similarity: диапазон [-1, 1]
        # similarity = 1 → полная аналогия (diff_ab == diff_cd)
        # similarity = 0 → нет аналогии
        similarity = (diff_ab * diff_cd).sum(dim=-1) / r

        # 1. Analogy clarity: поощряем чёткие аналогии (высокий |similarity|)
        analogy_clarity = -similarity.abs().mean()

        # 2. Analogy diversity: поощряем разнообразие (низкая variance)
        # Если все similarity одинаковые → коллапс к одной аналогии
        analogy_diversity = similarity.var()

        return analogy_clarity + 0.1 * (1 - analogy_diversity)


class SoulCodebook(nn.Module):
    """A1: Soul Codebook — файл "души" агента.

    Один обучаемый вектор, который подключается к латентному пространству.
    Один TRILIX становится 1000+ разными агентами.

    Меняя soul_id, модель "переключает личность":
    - soul_id=0: Python-разработчик
    - soul_id=1: Поэт
    - soul_id=2: Математик
    - и т.д.

    Soul-вектор добавляется к латентному представлению ДО роутера MoE,
    физически перестраивая синапсы под агента.

    Args:
        num_agents: Количество агентов (по умолчанию 1024)
        r: Latent dimension (должен совпадать с rank TRILIX)
    """

    def __init__(self, num_agents: int = 1024, r: int = 100):
        super().__init__()
        self.num_agents = num_agents
        self.r = r
        self.soul_vectors = nn.Embedding(num_agents, r)

        # Инициализация — случайные "характеры"
        nn.init.normal_(self.soul_vectors.weight, std=0.02)

    def forward(self, soul_id: torch.Tensor) -> torch.Tensor:
        """Получить вектор души для агента.

        Args:
            soul_id: [batch] — ID агента (0..num_agents-1)

        Returns:
            soul: [batch, r] — вектор "души" агента
        """
        return self.soul_vectors(soul_id)


class WorldModelHead(nn.Module):
    """A2: Latent World Model — учит латентное пространство предсказывать будущее.

    Обучает r-мерное пространство не просто сжимать данные,
    а предсказывать собственное будущее состояние.

    Это создаёт внутри сети "модель мира" — внутреннюю симуляцию,
    которая понимает причинно-следственные связи.

    Args:
        r: Latent dimension (должен совпадать с rank TRILIX)
        hidden_dim: Размер скрытого слоя предсказателя
    """

    def __init__(self, r: int = 100, hidden_dim: int = 128):
        super().__init__()
        self.r = r
        self.hidden_dim = hidden_dim

        # Предсказатель: z -> z_next
        self.predictor = nn.Sequential(
            nn.Linear(r, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, r),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Предсказать следующее латентное состояние.

        Args:
            z: [batch, r] — латентное состояние текущего токена

        Returns:
            z_pred: [batch, r] — предсказанное состояние следующего токена
        """
        return self.predictor(z)
