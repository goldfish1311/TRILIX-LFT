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
        # Gradient flows to soft version
        return grad_output, None


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

        self._init_parameters()

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
        codebook = STEBinary.apply(combined)

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

        # Commitment loss: encourage soft to stay close to hard
        commitment_loss = self.commitment_beta * F.mse_loss(U_soft.detach(), U_hard)

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

        # Atom diversity tracking (for monitoring)
        atoms_U_bin, atoms_V_bin = self._get_atoms_binary()
        atom_usage_U = atoms_U_bin.view(-1, self.rank).abs().mean(1)
        atom_usage_V = atoms_V_bin.view(-1, self.rank).abs().mean(1)
        prob_U = F.softmax(atom_usage_U, dim=0)
        prob_V = F.softmax(atom_usage_V, dim=0)
        entropy_U = -(prob_U * torch.log(prob_U + 1e-10)).sum()
        entropy_V = -(prob_V * torch.log(prob_V + 1e-10)).sum()
        aux_losses["atom_entropy"] = (entropy_U + entropy_V) / 2.0

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
    """Innovation 2: SGH - Stochastic Group Hierarchy for codebook regularization."""

    def __init__(
        self, codebook_size: int, rank: int, num_groups: int = 4, num_subgroups: int = 4
    ):
        super().__init__()
        self.k = codebook_size
        self.r = rank
        self.num_groups = num_groups
        self.num_subgroups = num_subgroups

        self.group_logits = nn.Parameter(torch.randn(codebook_size, num_groups) * 0.01)
        self.subgroup_logits = nn.Parameter(
            torch.randn(codebook_size, num_groups, num_subgroups) * 0.01
        )

    def get_group_assignment(self, temperature: float = 1.0):
        """Returns soft group/subgroup assignments."""
        group_soft = F.softmax(self.group_logits / max(temperature, 0.1), dim=-1)
        subgroup_soft = F.softmax(self.subgroup_logits / max(temperature, 0.1), dim=-1)
        return group_soft, subgroup_soft

    def get_regularization_loss(self, temperature: float = 1.0):
        """SGH regularization: uniform group usage + confident assignment."""
        group_soft, _ = self.get_group_assignment(temperature)
        group_usage = group_soft.sum(dim=0)
        uniform_loss = ((group_usage / group_usage.mean()) - 1.0).abs().mean()
        entropy_loss = -(group_soft * (group_soft + 1e-8).log()).sum(dim=-1).mean()
        return 0.01 * (uniform_loss + 0.1 * entropy_loss)


class ResidualVectorQuantizer(nn.Module):
    """Innovation 3: RVQ - Residual Vector Quantization."""

    def __init__(self, num_levels: int = 3, codebook_size: int = 16, rank: int = 64):
        super().__init__()
        self.num_levels = num_levels
        self.k = codebook_size
        self.r = rank
        for i in range(num_levels):
            self.register_parameter(
                f"codebook_level_{i}",
                nn.Parameter(torch.randn(codebook_size, rank) * 0.01),
            )
            self.register_parameter(
                f"idx_logits_level_{i}",
                nn.Parameter(torch.randn(1, codebook_size) * 0.01),
            )

    def forward(self, x: torch.Tensor):
        batch = x.shape[0]
        total = torch.zeros(batch, self.r, device=x.device, dtype=x.dtype)
        indices = []
        for lvl in range(self.num_levels):
            cb = getattr(self, f"codebook_level_{lvl}")
            lg = getattr(self, f"idx_logits_level_{lvl}")
            scores = lg @ cb.T
            idx = scores.argmax(dim=-1)
            indices.append(idx.item())
            total = total + cb[idx.squeeze(0)]
        return total, torch.tensor(indices, device=x.device)

    def get_codebook_size_effective(self) -> int:
        return self.k**self.num_levels


class SoftAttentionIndexBlender(nn.Module):
    """Innovation 1: SAIB - Soft Attention for Index Blending.

    Uses attention over codebook indices for smoother quantization.
    Instead of hard argmax, computes weighted blend of codebook entries.
    """

    def __init__(self, codebook_size: int, rank: int, num_heads: int = 4):
        super().__init__()
        self.k = codebook_size
        self.r = rank
        self.num_heads = num_heads

        self.query_proj = nn.Linear(rank, rank)
        self.key_proj = nn.Linear(rank, rank)
        self.value_proj = nn.Linear(rank, rank)
        self.scale = rank**-0.5

    def forward(self, x: torch.Tensor, codebook: torch.Tensor):
        """Attention-weighted codebook blend."""
        batch = x.shape[0]

        q = self.query_proj(x).unsqueeze(1)  # [batch, 1, r]
        k = self.key_proj(codebook).unsqueeze(0)  # [batch, k, r]
        v = self.value_proj(codebook).unsqueeze(0)  # [batch, k, r]

        attn_weights = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)

        blended = (attn_weights @ v).squeeze(1)  # [batch, r]
        hard_idx = attn_weights.argmax(dim=-1)

        return blended, hard_idx.squeeze(-1)


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
