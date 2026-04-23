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
from typing import Optional, Tuple, Dict
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


class FlatHierarchicalMoE(nn.Module):
    """
    B1.5: Flat Hierarchical Codebook (FHC)

    4 мета-эксперта × 4 базовых = 16 виртуальных специализаций.
    Один matmul — clear gradient flow. НЕ рекурсивное дерево.

    vs R-MoE Дипсика: дерево = непрозрачный gradient flow.
    FHC = один forward, легко профилировать.

    Router: [r] → [meta_k × base_k] = 4×4 = 16-way soft routing

    Каждый виртуальный эксперт = мета[b] ⊙ базовый[a]
    (element-wise product, NOT matmul)

    Args:
        meta_k: Number of meta-experts (default: 4)
        base_k: Number of base experts (default: 4)
        k: Codebook size per expert
        r: Latent dimension
        top_k: Number of virtual experts to activate per token
    """

    def __init__(
        self,
        meta_k: int = 4,
        base_k: int = 4,
        k: int = 64,
        r: int = 64,
        top_k: int = 2,
        num_atoms: int = 16,
        xor_arity: int = 3,
    ):
        super().__init__()
        self.meta_k = meta_k
        self.base_k = base_k
        self.virtual_k = meta_k * base_k
        self.k = k
        self.r = r
        self.top_k = top_k

        # Meta-experts: [meta_k, r] — "стратегии" маршрутизации
        self.meta_experts = nn.ModuleList(
            [
                CodebookExpert(k=k, r=r, num_atoms=num_atoms, xor_arity=xor_arity)
                for _ in range(meta_k)
            ]
        )

        # Base experts: [base_k, r] — "базовые паттерны"
        self.base_experts = nn.ModuleList(
            [
                CodebookExpert(k=k, r=r, num_atoms=num_atoms, xor_arity=xor_arity)
                for _ in range(base_k)
            ]
        )

        # Router: flat 1D routing over virtual experts
        # [r] → [meta_k * base_k] — one matmul, clean gradient
        self.router = nn.Linear(r, self.virtual_k, bias=False)

        # Meta affinity: how much each meta expert "likes" each base
        # [meta_k, base_k] — learned, not fixed
        self.meta_affinity = nn.Parameter(torch.ones(meta_k, base_k) * 0.5)

    def forward(
        self, x_latent: torch.Tensor, temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, r = x_latent.shape

        # D4 FIX: Get meta and base expert codebooks (D4: representative vectors, not centroids)
        meta_codebooks = [e.get_codewords(temperature) for e in self.meta_experts]
        base_codebooks = [e.get_codewords(temperature) for e in self.base_experts]
        # Each: [k, r]

        # D4 FIX: Use representative vectors (median) instead of centroids (mean)
        # mean(dim=0) схлопывает 128 кодслов в 1 вектор — теряет всю структуру!
        # median(dim=0) берёт "типичный" вектор, сохраняя больше информации
        meta_centroids = torch.stack(
            [cb.median(dim=0).values for cb in meta_codebooks], dim=0
        )
        base_centroids = torch.stack(
            [cb.median(dim=0).values for cb in base_codebooks], dim=0
        )

        meta_centroids = F.normalize(meta_centroids, dim=1)
        base_centroids = F.normalize(base_centroids, dim=1)

        affinity = torch.sigmoid(self.meta_affinity)
        virtual_centroids = torch.einsum(
            "mr,br,mr->mb", meta_centroids, base_centroids, affinity
        )
        virtual_centroids = F.normalize(virtual_centroids, dim=1)

        virtual_centroids_flat = virtual_centroids.reshape(-1, r)

        router_logits = self.router(x_latent)
        temp = max(temperature, 0.1)
        router_probs = F.softmax(router_logits / temp, dim=-1)
        gates, top_ids = torch.topk(router_probs, self.top_k, dim=-1)
        gates = gates / gates.sum(dim=-1, keepdim=True)

        selected = virtual_centroids_flat[top_ids]
        combined = (gates.unsqueeze(-1) * selected).sum(dim=1)

        router_prob = F.softmax(router_logits, dim=-1)
        aux_loss = (
            self.virtual_k
            * (router_prob.mean(dim=[0, 1]) * router_prob.mean(dim=[0, 1])).sum()
        )

        self._cached_virt_k = self.virtual_k
        self._cached_meta_k = self.meta_k
        self._cached_base_k = self.base_k
        self._cached_gates = gates
        self._cached_top_ids = top_ids
        self._cached_aux_loss = aux_loss

        return combined, aux_loss


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

        self.experts = nn.ModuleList(
            [
                CodebookExpert(k=k, r=r, num_atoms=16, xor_arity=3)
                for _ in range(num_experts)
            ]
        )
        self.router = nn.Linear(r, num_experts, bias=False)

    def forward(
        self, x_latent: torch.Tensor, temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, r = x_latent.shape
        router_logits = self.router(x_latent)
        gates, expert_ids = torch.topk(
            F.softmax(router_logits / max(temperature, 0.1), dim=-1), self.top_k, dim=-1
        )
        gates = gates / gates.sum(dim=-1, keepdim=True)
        expert_codebooks = [e.get_codewords(temperature) for e in self.experts]
        self._cached_expert_codebooks = expert_codebooks
        self._cached_gates = gates
        self._cached_expert_ids = expert_ids
        router_prob = F.softmax(router_logits, dim=-1)
        aux_loss = (
            self.num_experts
            * (router_prob.mean(dim=[0, 1]) * router_prob.mean(dim=[0, 1])).sum()
        )
        combined = torch.zeros(batch_size, seq_len, r, device=x_latent.device)
        for i in range(self.num_experts):
            expert_gate = (
                gates[..., 0] * (expert_ids[..., 0] == i).float()
                + gates[..., 1] * (expert_ids[..., 1] == i).float()
            )
            expert_centroid = expert_codebooks[i].mean(dim=0)
            combined += expert_gate.unsqueeze(-1) * expert_centroid
        return combined, aux_loss


class STEBinary(torch.autograd.Function):
    """Straight-Through Estimator for binary values"""

    @staticmethod
    def forward(ctx, x):
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
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


class DifferentiableAtomEvolver(nn.Module):
    """
    B3: Differentiable Atom Evolution (DAE)

    DIFFERENT от HAR:
    - HAR:    безградиентное, статистическое, "выживают популярные"
    - DAE:    дифференцируемое, градиентное, selection pressure

    Эволюция работает ЧЕРЕЗ ГРАДИЕНТЫ:
    1. Fitness = ||d_loss/d_atom|| — какой атом больше влияет на loss
    2. Mutation = atom + lr * grad(atom) * fitness  — двигаемся в сторону улучшения
    3. Сrossover: топ-K атомов производятoffspring через interpolate
    4. Selection: лучшие survive, худшие заменяются

    В отличие от HAR:
    - Использует GRADIENT для направления эволюции
    - Может создавать НОВЫЕ атомы (не только переиспользовать старые)
    - Эволюция идёт в пространстве параметров (atoms как genotype)
    - Selection pressure = grad_magnitude

    Ключевое: атомы — это genotype. Финальный фенотип = sign(atom).
    """

    def __init__(
        self,
        num_atoms: int,
        rank: int,
        evolution_interval: int = 500,
        selection_threshold: float = 0.1,
        elite_fraction: float = 0.2,
        mutation_scale: float = 0.05,
        crossover_prob: float = 0.3,
    ):
        super().__init__()
        self.num_atoms = num_atoms
        self.rank = rank
        self.evolution_interval = evolution_interval
        self.selection_threshold = selection_threshold
        self.elite_fraction = elite_fraction
        self.mutation_scale = mutation_scale
        self.crossover_prob = crossover_prob

        self.register_buffer("_step_counter", torch.zeros(1, dtype=torch.long))
        self.register_buffer("_atom_fitness_U", torch.zeros(num_atoms))
        self.register_buffer("_atom_fitness_V", torch.zeros(num_atoms))
        self.register_buffer("_atom_grad_mag_U", torch.zeros(num_atoms))
        self.register_buffer("_atom_grad_mag_V", torch.zeros(num_atoms))
        self.register_buffer("_evolution_count", torch.zeros(1, dtype=torch.long))
        self._registered = False
        self._last_evolution_step = 0

    def register_with_layer(self, layer: "TRILIXLinear"):
        """Register this DAE with a TRILIXLinear layer"""
        self._layer = layer
        self._registered = True

    def observe_gradient(
        self,
        atom_grad_U: torch.Tensor,
        atom_grad_V: torch.Tensor,
        combo_indices_U: torch.Tensor,
        combo_indices_V: torch.Tensor,
        loss_val: float,
    ):
        """
        Observe gradients on atoms and track fitness.

        Args:
            atom_grad_U: [num_atoms, rank] — gradient on atoms_U
            atom_grad_V: [num_atoms, rank] — gradient on atoms_V
            combo_indices_U: [codebook_size, xor_arity, num_atoms] one-hot
            combo_indices_V: [codebook_size, xor_arity, num_atoms] one-hot
            loss_val: current loss value (for normalization)
        """
        if not self._registered:
            return

        with torch.no_grad():
            self._step_counter += 1

            grad_mag_U = atom_grad_U.abs().mean(dim=1)  # [A]
            grad_mag_V = atom_grad_V.abs().mean(dim=1)  # [A]

            self._atom_grad_mag_U = 0.9 * self._atom_grad_mag_U + 0.1 * grad_mag_U
            self._atom_grad_mag_V = 0.9 * self._atom_grad_mag_V + 0.1 * grad_mag_V

            combo_flat_U = combo_indices_U.sum(dim=1)  # [K, A]
            combo_flat_V = combo_indices_V.sum(dim=1)  # [K, A]
            atom_active_U = (combo_flat_U.sum(dim=0) > 0).float()
            atom_active_V = (combo_flat_V.sum(dim=0) > 0).float()

            fitness_U = self._atom_grad_mag_U * atom_active_U
            fitness_V = self._atom_grad_mag_V * atom_active_V
            self._atom_fitness_U = 0.95 * self._atom_fitness_U + 0.05 * fitness_U
            self._atom_fitness_V = 0.95 * self._atom_fitness_V + 0.05 * fitness_V

            if (
                self._step_counter.item() - self._last_evolution_step
                >= self.evolution_interval
            ):
                self._evolve()
                self._last_evolution_step = self._step_counter.item()

    def _evolve(self):
        """Apply differentiable evolution: mutation + crossover + selection"""
        layer = getattr(self, "_layer", None)
        if layer is None:
            return

        with torch.no_grad():
            self._evolution_count += 1

            fitness_U = self._atom_fitness_U
            fitness_V = self._atom_fitness_V
            top_fitness_U = fitness_U.max().item()
            top_fitness_V = fitness_V.max().item()

            if top_fitness_U < 1e-8 and top_fitness_V < 1e-8:
                return

            norm_fitness_U = fitness_U / (top_fitness_U + 1e-10)
            norm_fitness_V = fitness_V / (top_fitness_V + 1e-10)

            elite_count = max(1, int(self.num_atoms * self.elite_fraction))
            elite_U = norm_fitness_U.topk(elite_count)[1]
            elite_V = norm_fitness_V.topk(elite_count)[1]

            new_atoms_U = layer.atoms_U.data.clone()
            new_atoms_V = layer.atoms_V.data.clone()

            for i in range(self.num_atoms):
                if norm_fitness_U[i] < self.selection_threshold:
                    parent1_idx = elite_U[torch.randint(elite_count, (1,))]
                    new_atom = new_atoms_U[parent1_idx].clone()

                    if torch.rand(1).item() < self.crossover_prob and elite_count > 1:
                        parent2_idx = elite_U[torch.randint(elite_count, (1,))]
                        parent2 = new_atoms_U[parent2_idx]
                        alpha = torch.rand(1).item()
                        new_atom = alpha * new_atom + (1 - alpha) * parent2

                    mutation_noise = torch.randn_like(new_atom) * self.mutation_scale
                    new_atom = new_atom + mutation_noise * max(norm_fitness_U[i], 0.1)
                    new_atoms_U[i] = new_atom

                if norm_fitness_V[i] < self.selection_threshold:
                    parent1_idx = elite_V[torch.randint(elite_count, (1,))]
                    new_atom = new_atoms_V[parent1_idx].clone()

                    if torch.rand(1).item() < self.crossover_prob and elite_count > 1:
                        parent2_idx = elite_V[torch.randint(elite_count, (1,))]
                        parent2 = new_atoms_V[parent2_idx]
                        alpha = torch.rand(1).item()
                        new_atom = alpha * new_atom + (1 - alpha) * parent2

                    mutation_noise = torch.randn_like(new_atom) * self.mutation_scale
                    new_atom = new_atom + mutation_noise * max(norm_fitness_V[i], 0.1)
                    new_atoms_V[i] = new_atom

            layer.atoms_U.data.copy_(new_atoms_U)
            layer.atoms_V.data.copy_(new_atoms_V)

            self._atom_fitness_U.zero_()
            self._atom_fitness_V.zero_()

    def get_stats(self) -> dict:
        """Return current DAE statistics"""
        top_fitness_U = self._atom_fitness_U.max().item()
        top_fitness_V = self._atom_fitness_V.max().item()
        return {
            "dae_registered": self._registered,
            "dae_step": self._step_counter.item(),
            "dae_evolution_count": self._evolution_count.item(),
            "top_fitness_U": top_fitness_U,
            "top_fitness_V": top_fitness_V,
            "avg_fitness_U": self._atom_fitness_U.mean().item(),
            "avg_fitness_V": self._atom_fitness_V.mean().item(),
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
        use_fhc: bool = False,
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
        self.dae = None
        self.use_dae = False
        self.use_fhc = use_fhc

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
        # B1.5: FlatHierarchicalMoE or standard MoECodebook
        if self.use_moe:
            if self.use_fhc:
                self.moe_codebook_U = FlatHierarchicalMoE(
                    meta_k=4,
                    base_k=4,
                    k=codebook_size,
                    r=rank,
                    top_k=moe_top_k,
                    num_atoms=num_atoms,
                    xor_arity=xor_arity,
                )
                self.moe_codebook_V = FlatHierarchicalMoE(
                    meta_k=4,
                    base_k=4,
                    k=codebook_size,
                    r=rank,
                    top_k=moe_top_k,
                    num_atoms=num_atoms,
                    xor_arity=xor_arity,
                )
            else:
                self.moe_codebook_U = MoECodebook(
                    num_experts=num_experts, k=codebook_size, r=rank, top_k=moe_top_k
                )
                self.moe_codebook_V = MoECodebook(
                    num_experts=num_experts, k=codebook_size, r=rank, top_k=moe_top_k
                )
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

    def enable_dae(
        self, evolution_interval: int = 500, selection_threshold: float = 0.1
    ):
        """Enable Differentiable Atom Evolution (B3)"""
        if self.dae is None:
            self.dae = DifferentiableAtomEvolver(
                num_atoms=self.num_atoms,
                rank=self.rank,
                evolution_interval=evolution_interval,
                selection_threshold=selection_threshold,
            )
            self.dae.register_with_layer(self)
            self.use_dae = True

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

        # Commitment loss: encourage soft (encoder output) to stay close to hard (codebook)
        # Gradient flows to U_soft, NOT to U_hard (codebook stays fixed)
        commitment_loss = self.commitment_beta * F.mse_loss(U_soft, U_hard.detach())

        # SGH: Semantic Gradient Highway — coherence loss only (highway loss requires gradients, not available in forward)
        sgh_loss = 0.0
        if self.use_sgh and self.sgh_U is not None:
            coherence_loss = self.sgh_U.get_group_coherence_loss(codebook_U)
            sgh_loss = coherence_loss
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

        # EMA update of codebook — VECTORIZED (D1 fix)
        # Было: for i in range(self.codebook_size) — 128 Python итераций за слой
        # Стало: один scatter_add_ — GPU kernel
        # 168 слоёв × 128 = 21,504 → 168 GPU операций
        counts = torch.bincount(idx_U_hard_idx, minlength=self.codebook_size).float()

        # Накопить суммы через scatter_add
        sums = torch.zeros(self.codebook_size, self.rank, device=U_hard.device)
        sums.scatter_add_(
            0,
            idx_U_hard_idx.unsqueeze(-1).expand(-1, self.rank),
            U_hard,
        )

        # EMA только для активных кодслов
        active = counts > 0
        if active.any():
            new_means = sums[active] / counts[active].unsqueeze(-1)
            self.codebook_U_ema[active] = (
                self.atom_ema_decay * self.codebook_U_ema[active]
                + (1 - self.atom_ema_decay) * new_means
            )
            self.codebook_U_count[active] += 1

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

        commitment_loss = self.commitment_beta * F.mse_loss(V_soft, V_hard.detach())

        with torch.no_grad():
            self.usage_counter_V.index_add_(
                0, idx_V_hard_idx, torch.ones_like(idx_V_hard_idx, dtype=torch.long)
            )

        # EMA update of codebook — VECTORIZED (D1 fix)
        counts = torch.bincount(idx_V_hard_idx, minlength=self.codebook_size).float()
        sums = torch.zeros(self.codebook_size, self.rank, device=V_hard.device)
        sums.scatter_add_(
            0,
            idx_V_hard_idx.unsqueeze(-1).expand(-1, self.rank),
            V_hard,
        )
        active = counts > 0
        if active.any():
            new_means = sums[active] / counts[active].unsqueeze(-1)
            self.codebook_V_ema[active] = (
                self.atom_ema_decay * self.codebook_V_ema[active]
                + (1 - self.atom_ema_decay) * new_means
            )
            self.codebook_V_count[active] += 1

        V_final = STEIndex.apply(V_soft, V_hard)

        # SGH: Semantic Gradient Highway for V — coherence only
        if self.use_sgh and self.sgh_V is not None:
            coherence_loss_v = self.sgh_V.get_group_coherence_loss(codebook_V)
            self._cached_sgh_loss = (
                self._cached_sgh_loss + coherence_loss_v
                if hasattr(self, "_cached_sgh_loss")
                else coherence_loss_v
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

        # B3: DAE observation — observe combo usage (gradients come after backward)
        # DAE needs gradients, so we track atom activation during forward
        # Actual gradient-based evolution happens post-backward via step()
        if self.training and self.use_dae and self.dae is not None:
            combo_idx_U = self._get_combo_indices_hard(self.combo_indices_U_logits)
            combo_idx_V = self._get_combo_indices_hard(self.combo_indices_V_logits)
            self._cached_combo_U = combo_idx_U
            self._cached_combo_V = combo_idx_V
            self._cached_loss_val = aux_losses.get(
                "commitment_U", torch.tensor(0.0)
            ).item()

        return output, aux_losses

    def step_dae(self, loss_val: float = 0.0):
        """B3: DAE step — called after backward() to observe gradients and evolve

        DAE needs gradient info which is only available after backward(),
        so this is a separate method to be called from training loop.
        """
        if not (self.use_dae and self.dae is not None):
            return

        grad_U = getattr(self.atoms_U, "grad", None)
        grad_V = getattr(self.atoms_V, "grad", None)
        combo_U = getattr(self, "_cached_combo_U", None)
        combo_V = getattr(self, "_cached_combo_V", None)

        if grad_U is not None and combo_U is not None:
            self.dae.observe_gradient(grad_U, grad_V, combo_U, combo_V, loss_val)
            dae_stats = self.dae.get_stats()
            self._cached_dae_stats = dae_stats

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


class EmergentAgentSwarm(nn.Module):
    """
    B2: Emergent Agent Swarm

    1024+ агента работают как рой. Агенты НЕ знают о существовании друг друга —
    специализация emerges из взаимодействия через мир (WorldModel).

    Как это работает:
    1. Все агенты живут в одном SoulCodebook (1024 вектора)
    2. Каждый forward: агенты получают задачу (task_embedding)
    3. Task embedding = query для attention между агентами
    4. "Лучшие" агенты для задачи получают больше внимания
    5. Специализация emergent: агент128 специализируется на Python,
       агент256 — на математику, агент512 — на перевод

    Почему emergent, а не explicit:
    - НЕ создаём 1024 отдельных "экспертов"
    - НЕ делаем routing "агент → задача"
    - Вместо этого: attention между агентами + задачей
    - Специализация emerges из поощрения "агент X хорош для задачи Y"

    Аналогия: муравьиная колония. Каждый муравей простой,
    но вместе они решают сложные задачи.

    Args:
        num_agents: 1024
        r: latent dimension
        num_heads: attention heads для agent-task interaction
    """

    def __init__(
        self,
        num_agents: int = 1024,
        r: int = 100,
        num_heads: int = 4,
        temperature: float = 1.0,
        specialization_lr: float = 0.001,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.r = r
        self.num_heads = num_heads
        self.head_dim = r // num_heads

        self.temperature = temperature
        self.specialization_lr = specialization_lr

        self.register_buffer("_agent_scores", torch.zeros(num_agents))
        self.register_buffer("_task_history", torch.zeros(num_agents, 64))
        self.register_buffer("_step", torch.zeros(1, dtype=torch.long))
        self.register_buffer("_specialization_count", torch.zeros(num_agents))

    def forward(
        self,
        soul_vectors: torch.Tensor,
        task_embedding: torch.Tensor,
        return_attention: bool = False,
    ) -> Dict:
        """
        Forward pass: agent-task attention.

        Args:
            soul_vectors: [batch, num_agents, r] — все агенты
            task_embedding: [batch, r] — текущая задача
            return_attention: вернуть attention weights для визуализации

        Returns:
            dict с:
            - boosted_soul: [batch, num_agents, r] — агенты с учётом specialized attention
            - specialization_loss: scalar — поощряет специализацию
            - agent_scores: [num_agents] — текущие score агентов
        """
        batch_size, num_agents, r = soul_vectors.shape
        task_emb = task_embedding.unsqueeze(1).expand(-1, num_agents, -1)

        q = task_embedding.unsqueeze(1)
        k = soul_vectors
        v = soul_vectors

        q_heads = q.view(batch_size, 1, self.num_heads, self.head_dim)
        k_heads = k.view(batch_size, num_agents, self.num_heads, self.head_dim)
        v_heads = v.view(batch_size, num_agents, self.num_heads, self.head_dim)

        q_heads = q_heads.transpose(1, 2)
        k_heads = k_heads.transpose(1, 2).transpose(2, 3)
        v_heads = v_heads.transpose(1, 2)

        scores = torch.matmul(q_heads, k_heads) / (self.head_dim**0.5)
        attn = F.softmax(scores / max(self.temperature, 0.1), dim=-1)

        attn_out = torch.matmul(attn, v_heads)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, num_agents, r)

        boosted_soul = soul_vectors + 0.1 * attn_out

        specialization_loss = self._specialization_loss(attn, task_embedding)

        agent_scores = attn.mean(dim=[0, 1, 2])

        result = {
            "boosted_soul": boosted_soul,
            "specialization_loss": specialization_loss,
            "agent_scores": agent_scores,
        }

        if return_attention:
            result["attention"] = attn

        self._step += 1
        return result

    def _specialization_loss(
        self, attention: torch.Tensor, task: torch.Tensor
    ) -> torch.Tensor:
        """
        Поощряем специализацию: агент X становится "лучшим для задачи Y".

        Loss = diversity_loss - concentration_loss
        - diversity_loss: разные агенты получают attention для разных задач
        - concentration_loss: не должно быть "один агент для всех"
        """
        batch_size = attention.size(0)

        top_attn = attention.max(dim=-1)[0].mean(dim=[0, 1])
        concentration = -(top_attn**2).mean()

        agent_affinity = attention.mean(dim=[0, 1])
        diversity = -(agent_affinity * torch.log(agent_affinity + 1e-10)).sum()

        return concentration + 0.01 * diversity

    def get_agent_stats(self) -> Dict:
        """Статистика агентов"""
        return {
            "total_agents": self.num_agents,
            "top_agent": int(self._agent_scores.argmax().item()),
            "avg_score": self._agent_scores.mean().item(),
            "specialization_rate": (
                self._specialization_count.float() / max(self._step.item(), 1)
            )
            .mean()
            .item(),
        }


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


class BeliefGate(nn.Module):
    """A3: Belief Gate — убеждения агента о мире.

    Belief = "что агент знает о мире" — вектор убеждений.
    Belief Gate модулирует предсказания WorldModel на основе убеждений.

    Как работает:
    1. Belief state — вектор убеждений [r], обучается вместе с моделью
    2. Belief update: предсказание - реальность = error → обновляем belief
    3. Belief gate: z_pred = z_pred * sigmoid(belief) + z * (1 - sigmoid(belief))
       - high belief → предсказание доминирует
       - low belief → текущее состояние доминирует

    Аналогия:
    - Человек с сильными убеждениями ("я знаю как это работает")
      игнорирует новую информацию и полагается на предсказание
    - Человек без убеждений ("я не знаю")
      больше полагается на текущие данные

    Args:
        r: Latent dimension
        belief_dim: Dimension of belief representation (default: r//4)
    """

    def __init__(self, r: int = 100, belief_dim: int = 25):
        super().__init__()
        self.r = r
        self.belief_dim = belief_dim

        self.belief_vector = nn.Parameter(torch.randn(belief_dim) * 0.01)

        self.belief_encoder = nn.Linear(r, belief_dim)
        self.belief_gate = nn.Sequential(
            nn.Linear(belief_dim, r),
            nn.Sigmoid(),
        )

        self.belief_predictor = nn.Sequential(
            nn.Linear(r + belief_dim, hidden_dim := r),
            nn.GELU(),
            nn.Linear(hidden_dim, r),
        )

        self.register_buffer("_belief_update_rate", torch.tensor(0.1))
        self.register_buffer("_last_belief", torch.zeros(belief_dim))

    def forward(
        self,
        z: torch.Tensor,
        z_pred: torch.Tensor,
        z_actual: Optional[torch.Tensor] = None,
        return_belief: bool = False,
    ) -> Dict:
        """
        Belief-gated prediction.

        Args:
            z: [batch, r] — текущее латентное состояние
            z_pred: [batch, r] — предсказание WorldModel
            z_actual: [batch, r] — реальное следующее состоятие (для обновления belief)
            return_belief: вернуть belief state

        Returns:
            gated_pred: [batch, r] — belief-gated предсказание
            belief_loss: scalar — loss на обновление belief (если z_actual дан)
        """
        batch_size = z.size(0)

        belief_emb = self.belief_encoder(z)
        gate = self.belief_gate(belief_emb)

        gated_pred = z_pred * gate + z * (1 - gate)

        result = {"gated_pred": gated_pred, "gate": gate}

        if z_actual is not None and self.training:
            error = (z_pred - z_actual).abs().mean()
            belief_update = error * self._belief_update_rate
            self.belief_vector.data += (
                torch.randn_like(self.belief_vector) * belief_update
            )
            self._last_belief.copy_(self.belief_vector.detach())

            result["belief_loss"] = error
        else:
            result["belief_loss"] = torch.tensor(0.0, device=z.device)

        if return_belief:
            result["belief_emb"] = belief_emb
            result["belief_vector"] = self.belief_vector

        return result

    def get_belief_stats(self) -> Dict:
        """Статистика убеждений"""
        belief_magnitude = self.belief_vector.abs().mean().item()
        return {
            "belief_magnitude": belief_magnitude,
            "belief_dim": self.belief_dim,
        }


class ErrorDrivenHypernetwork(nn.Module):
    """C1: EDH — Error-Driven Hypernetwork.

    Гиперсеть, которая получает на вход error-сигнал и генерирует
    "Builder Weights" — специализированные веса для разных паттернов ошибок.

    Проблема (от Клода): Builder Expert имел circular dependency — ему нужен
    task_embedding, который получается из выхода сети. EDH решает это:
    EDH получает ГЛОБАЛЬНЫЙ loss-сигнал, не требует task_embedding от сети.

    Как работает:
    1. EDH получает loss-сигнал (error embedding)
    2. Генерирует "error-type embedding" — какой тип ошибки произошёл
    3. Генерирует builder_weights — специализированные веса для этого типа ошибки
    4. Builder Weights используются как bias/модуляция в слоях

    Error patterns:
    - High CE loss → нужно "standard" builder
    - High WorldModel loss → нужно "prediction" builder
    - High Diversity loss → нужно "creative" builder
    - High Belief loss → нужно "skeptical" builder

    Аналогия: EDH = "консультант", который анализирует ошибки команды
    и даёт каждому совет (builder_weights), как лучше работать.

    Args:
        error_dim: Размерность error embedding (default: 64)
        builder_dim: Размерность builder специализации (default: rank_r)
        num_builders: Количество типов builders (default: 8)
    """

    def __init__(
        self, error_dim: int = 64, builder_dim: int = 100, num_builders: int = 8
    ):
        super().__init__()
        self.error_dim = error_dim
        self.builder_dim = builder_dim
        self.num_builders = num_builders

        self.error_encoder = nn.Sequential(
            nn.Linear(4, error_dim),
            nn.GELU(),
            nn.Linear(error_dim, error_dim),
        )

        self.error_type_embedding = nn.Embedding(num_builders, error_dim)
        nn.init.normal_(self.error_type_embedding.weight, std=0.01)

        self.builder_generator = nn.Sequential(
            nn.Linear(error_dim, error_dim * 2),
            nn.GELU(),
            nn.Linear(error_dim * 2, builder_dim),
        )

        self.builder_modulation = nn.Sequential(
            nn.Linear(error_dim, builder_dim),
            nn.Tanh(),
        )

        self._registered_builders = nn.ParameterList(
            [nn.Parameter(torch.randn(builder_dim) * 0.01) for _ in range(num_builders)]
        )

        self._current_builder_weights: Optional[torch.Tensor] = None
        self._last_error_type = 0

    def forward(
        self,
        ce_loss: torch.Tensor,
        world_model_loss: torch.Tensor,
        diversity_loss: torch.Tensor,
        belief_loss: torch.Tensor,
        return_builder: bool = False,
    ) -> Dict:
        """Forward pass EDH."""
        ce_val = (
            ce_loss.detach().mean().item()
            if ce_loss.numel() > 1
            else ce_loss.detach().item()
        )
        wm_val = (
            world_model_loss.detach().mean().item()
            if world_model_loss.numel() > 1
            else world_model_loss.detach().item()
        )
        div_val = (
            diversity_loss.detach().mean().item()
            if diversity_loss.numel() > 1
            else diversity_loss.detach().item()
        )
        bel_val = (
            belief_loss.detach().mean().item()
            if belief_loss.numel() > 1
            else belief_loss.detach().item()
        )

        loss_vector = torch.tensor(
            [ce_val, wm_val, div_val, bel_val], device=next(self.parameters()).device
        )
        loss_vector_norm = loss_vector / (loss_vector.max() + 1e-8)

        error_emb = self.error_encoder(loss_vector_norm.unsqueeze(0)).squeeze(0)

        error_type_raw = error_emb.sum(dim=-1) / (self.error_dim**0.5)
        error_type = int(torch.argmax(error_type_raw.abs()).item()) % self.num_builders
        self._last_error_type = error_type

        error_type_emb = self.error_type_embedding(
            torch.tensor(error_type, device=error_emb.device)
        )

        combined_emb = error_emb + error_type_emb

        builder_weights = self.builder_generator(combined_emb.unsqueeze(0)).squeeze(0)

        modulation = self.builder_modulation(combined_emb.unsqueeze(0)).squeeze(0)
        builder_weights = builder_weights * (1.0 + modulation)

        self._current_builder_weights = builder_weights.detach()

        builder_reg_loss = torch.tensor(0.0, device=error_emb.device)
        if self.training:
            target_builder = self._registered_builders[error_type]
            builder_reg_loss = F.mse_loss(builder_weights, target_builder.detach())
            current_builder_data = self._registered_builders[error_type].data
            current_builder_data.copy_(
                current_builder_data * 0.95 + builder_weights.detach() * 0.05
            )

        result = {
            "builder_weights": builder_weights,
            "error_type": error_type,
            "error_embedding": error_emb,
            "builder_loss": builder_reg_loss,
        }

        if return_builder:
            result["registered_builder"] = self._registered_builders[error_type]

        return result

    def get_builder_for_layer(
        self,
        layer_idx: int,
        num_layers: int,
    ) -> torch.Tensor:
        """Get builder weights modulated by layer position."""
        if self._current_builder_weights is None:
            return torch.zeros(self.builder_dim, device=next(self.parameters()).device)

        layer_factor = (layer_idx / max(num_layers - 1, 1)) ** 0.5
        pos_emb = torch.sin(
            torch.tensor([layer_idx], dtype=torch.float32)
            / torch.pow(
                torch.tensor(10000.0),
                torch.arange(0, self.builder_dim // 4, dtype=torch.float32)
                / (self.builder_dim // 4),
            )
        ).to(self._current_builder_weights.device)

        modulated = self._current_builder_weights * (
            1.0 + layer_factor * pos_emb[: self.builder_dim]
        )
        return modulated

    def get_edh_stats(self) -> Dict:
        """Статистика EDH."""
        builder_magnitudes = [b.abs().mean().item() for b in self._registered_builders]
        mean_mag = sum(builder_magnitudes) / len(builder_magnitudes)
        return {
            "builder_magnitude_mean": mean_mag,
            "builder_magnitude_std": (
                sum((x - mean_mag) ** 2 for x in builder_magnitudes)
                / len(builder_magnitudes)
            )
            ** 0.5,
            "num_builders": self.num_builders,
            "last_error_type": self._last_error_type,
            "current_builder_magnitude": (
                self._current_builder_weights.abs().mean().item()
                if self._current_builder_weights is not None
                else 0.0
            ),
        }


class ReflectiveErrorLoop(nn.Module):
    """C2: REL — Reflective Error Loop.

    Рефлексивный цикл ошибок: модель анализирует СВОИ ошибки
    и корректирует своё поведение ("я сомневаюсь здесь").

    Как работает:
    1. Per-token uncertainty estimation — для каждого токена вычисляем
       "уверенность" модели
    2. High-uncertainty tokens → повышенный loss при backprop
    3. Self-correction: модель учится быть увереннее в "сомнительных" местах

    Компоненты:
    - Uncertainty Estimator: предсказывает variance/entropy для каждого токена
    - Reflective Loss: CE loss * uncertainty_weight
    - Confidence Calibration: учит модель быть увереннее когда нужно

    Args:
        hidden_dim: Размерность hidden states
        uncertainty_dim: Размерность uncertainty embedding (default: 16)
    """

    def __init__(self, hidden_dim: int = 256, uncertainty_dim: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.uncertainty_dim = uncertainty_dim

        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, uncertainty_dim),
        )

        self.confidence_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

        self.uncertainty_gate = nn.Sequential(
            nn.Linear(uncertainty_dim, 1),
            nn.Softplus(),
        )

        self._uncertainty_history_avg: list[float] = []
        self._uncertainty_history_max: list[float] = []
        self._step_count = 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        return_uncertainty: bool = False,
    ) -> Dict:
        """Forward pass REL."""
        batch_size, seq_len, _ = hidden_states.shape

        uncertainty_emb = self.uncertainty_estimator(hidden_states)
        uncertainty_gate = self.uncertainty_gate(uncertainty_emb).squeeze(-1)

        confidence = self.confidence_predictor(hidden_states).squeeze(-1)

        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        entropy_normalized = entropy / (entropy.max() + 1e-8)

        uncertainty_map = entropy_normalized * 0.6 + uncertainty_gate * 0.4

        ce_loss_per_token = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction="none",
        ).view(batch_size, seq_len)

        reflective_loss_per_token = ce_loss_per_token * uncertainty_map
        reflective_loss = reflective_loss_per_token.mean()

        confidence_target = 1.0 - uncertainty_map.detach()
        confidence_loss = F.binary_cross_entropy(
            confidence, confidence_target, reduction="none"
        ).mean()

        correction_strength = uncertainty_map.mean()

        reflective_loss = reflective_loss + 0.1 * confidence_loss

        self._step_count += 1
        if self._step_count % 100 == 0:
            self._uncertainty_history_avg.append(uncertainty_map.mean().item())
            self._uncertainty_history_max.append(uncertainty_map.max().item())

        result = {
            "reflective_loss": reflective_loss,
            "uncertainty_map": uncertainty_map,
            "confidence_loss": confidence_loss,
            "correction_strength": correction_strength,
        }

        if return_uncertainty:
            result["entropy"] = entropy
            result["uncertainty_gate"] = uncertainty_gate
            result["confidence"] = confidence

        return result

    def get_rel_stats(self) -> Dict:
        """Статистика REL."""
        return {
            "step_count": self._step_count,
            "uncertainty_avg_avg": (
                sum(self._uncertainty_history_avg) / len(self._uncertainty_history_avg)
                if self._uncertainty_history_avg
                else 0.0
            ),
            "uncertainty_max_avg": (
                sum(self._uncertainty_history_max) / len(self._uncertainty_history_max)
                if self._uncertainty_history_max
                else 0.0
            ),
            "uncertainty_trend": (
                self._uncertainty_history_avg[-1] - self._uncertainty_history_avg[0]
                if len(self._uncertainty_history_avg) > 1
                else 0.0
            ),
        }



# =============================================================================
# E1-E4: Innovations from Claude (integrated at end of file)
# =============================================================================

class BinaryApproximateAttention(nn.Module):
    """E1: BinAttn — бинарное sparse attention."""
    
    def __init__(self, hidden_size, num_heads, head_dim, top_k_precise=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads  
        self.head_dim = head_dim
        self.top_k_fraction = top_k_precise
        
    def forward(self, q, k, v, mask=None):
        batch_size, num_heads, seq_len, head_dim = q.shape
        # Бинарное приближение
        q_bin = torch.sign(q).detach()
        k_bin = torch.sign(k).detach()
        approx_scores = torch.matmul(q_bin, k_bin.transpose(-2, -1)) / head_dim
        # Top-K для точного вычисления
        k_precise = max(1, int(seq_len * self.top_k_fraction))
        topk_vals, topk_idx = torch.topk(approx_scores, k_precise, dim=-1)
        # Собираем K, V для top-k
        k_gathered = torch.gather(k, 2, topk_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim))
        v_gathered = torch.gather(v, 2, topk_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim))
        # Точные скоры
        precise_scores = torch.matmul(q.unsqueeze(-2), k_gathered.transpose(-2, -1)).squeeze(-2) / math.sqrt(head_dim)
        if mask is not None:
            mask_gathered = torch.gather(mask.expand(-1, -1, seq_len, -1), -1, topk_idx)
            precise_scores = precise_scores + mask_gathered
        precise_probs = F.softmax(precise_scores, dim=-1)
        attn_output = torch.matmul(precise_probs.unsqueeze(-2), v_gathered).squeeze(-2)
        return attn_output


class ShadowDistillationHead(nn.Module):
    """E2: OKDSH — Shadow Head for Knowledge Distillation."""
    
    def __init__(self, hidden_size=256, vocab_size=32000, rank=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.rank = rank
        self.shadow_proj = nn.Linear(hidden_size, rank, bias=False)
        self.shadow_head = nn.Linear(rank, vocab_size, bias=False)
        nn.init.normal_(self.shadow_proj.weight, std=0.02)
        nn.init.normal_(self.shadow_head.weight, std=0.02)
        
    def forward(self, hidden_states):
        projected = self.shadow_proj(hidden_states)
        return self.shadow_head(projected)
    
    def distillation_loss(self, trilix_logits, shadow_logits, temperature=2.0):
        trilix_soft = F.log_softmax(trilix_logits / temperature, dim=-1)
        shadow_soft = F.softmax(shadow_logits.detach() / temperature, dim=-1)
        return F.kl_div(
            trilix_soft.view(-1, self.vocab_size),
            shadow_soft.view(-1, self.vocab_size),
            reduction='batchmean'
        ) * (temperature ** 2)


class AdaptiveRankSchedule:
    """E3: ARL — Adaptive Rank per Layer."""
    
    def __init__(self, num_layers, base_rank=100, min_factor=0.5, max_factor=2.0):
        self.num_layers = num_layers
        self.base_rank = base_rank
        self.rank_schedule = self._generate_schedule(min_factor, max_factor)
        
    def _generate_schedule(self, min_factor, max_factor):
        ranks = []
        for i in range(self.num_layers):
            progress = i / (self.num_layers - 1) if self.num_layers > 1 else 0
            parabola = 4 * progress * (1 - progress)
            factor = min_factor + (max_factor - min_factor) * parabola
            rank = int(self.base_rank * factor)
            rank = max(8, (rank // 8) * 8)
            ranks.append(rank)
        return ranks
    
    def get_rank(self, layer_idx):
        return self.rank_schedule[layer_idx]


class ConfidenceWeightedLoss(nn.Module):
    """E4: CWL — Confidence-Weighted Loss."""
    
    def __init__(self, confidence_temp=2.0, min_weight=0.1):
        super().__init__()
        self.confidence_temp = confidence_temp
        self.min_weight = min_weight
        
    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        probs = F.softmax(shift_logits.detach() / self.confidence_temp, dim=-1)
        max_prob = probs.max(dim=-1).values
        weights = (1.0 - max_prob).clamp(self.min_weight, 1.0)
        weights = weights / weights.mean()
        ce_per_token = F.cross_entropy(
            shift_logits.view(-1, logits.size(-1)),
            shift_labels.view(-1),
            reduction='none'
        ).view_as(shift_labels)
        return (ce_per_token * weights).mean()
