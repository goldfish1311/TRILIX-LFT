"""
TRILIX Transformer Model
Full transformer with TRILIX linear layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict

from .layers import (
    TRILIXLinear,
    WorldModelHead,
    BeliefGate,
    SoulCodebook,
    SymbolicDiffLoss,
    STEBinary,
    EmergentAgentSwarm,
    ErrorDrivenHypernetwork,
    ReflectiveErrorLoop,
)
from .config import TRILIXConfig


class TRILIXRMSNorm(nn.Module):
    """RMSNorm in FP32 for stability"""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Always compute in FP32
        orig_dtype = x.dtype
        x_fp32 = x.float()
        rms = torch.sqrt(torch.mean(x_fp32**2, dim=-1, keepdim=True) + self.eps)
        x_norm = x_fp32 / rms
        return (x_norm * self.weight).to(orig_dtype)


def get_trilix_kwargs(config: TRILIXConfig) -> Dict:
    """Get common kwargs for TRILIXLinear from config"""
    return {
        "rank": config.rank_r,
        "codebook_size": config.codebook_k,
        "num_atoms": config.num_atoms_A,
        "xor_arity": config.xor_arity_b,
        "commitment_beta": config.commitment_beta,
        "atom_ema_decay": config.atom_ema_decay,
        "use_moe": config.use_moe,
        "num_experts": config.num_experts if config.use_moe else 4,
        "moe_top_k": config.moe_top_k if config.use_moe else 2,
        "use_saib": config.use_saib,
        "use_rvq": config.use_rvq,
        "use_sgh": config.use_sgh,
        "use_lcc": config.use_lcc,
        "use_fhc": config.use_fhc,
    }


class TRILIXAttention(nn.Module):
    """Grouped Query Attention with TRILIX linear layers"""

    def __init__(self, config: TRILIXConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        trilix_kwargs = get_trilix_kwargs(config)

        self.q_proj = TRILIXLinear(
            config.hidden_size, self.num_heads * self.head_dim, **trilix_kwargs
        )

        self.k_proj = TRILIXLinear(
            config.hidden_size, self.num_kv_heads * self.head_dim, **trilix_kwargs
        )

        self.v_proj = TRILIXLinear(
            config.hidden_size, self.num_kv_heads * self.head_dim, **trilix_kwargs
        )

        self.o_proj = TRILIXLinear(
            self.num_heads * self.head_dim, config.hidden_size, **trilix_kwargs
        )

        # RoPE
        self.rope_theta = 500000.0
        self._init_rope()

    def _init_rope(self):
        """Initialize RoPE frequencies"""
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def _apply_rope(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply Rotary Positional Embedding"""
        # x: [batch, num_heads, seq_len, head_dim]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, head_dim//2]

        # Duplicate for full head_dim
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, head_dim]

        cos = emb.cos()[None, None, :, :]  # [1, 1, seq_len, head_dim]
        sin = emb.sin()[None, None, :, :]

        # Rotate
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)

        return x * cos + x_rotated * sin

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple], Dict]:
        """
        Forward pass with TRILIX compression

        Returns:
            hidden_states: [batch, seq_len, hidden_size]
            past_key_value: cached KV for generation
            aux_losses: dict with commitment losses
        """
        batch_size, seq_len, _ = hidden_states.shape

        aux_losses = {}

        # Project with TRILIX layers
        q, q_losses = self.q_proj(hidden_states)
        k, k_losses = self.k_proj(hidden_states)
        v, v_losses = self.v_proj(hidden_states)

        for k_loss in [q_losses, k_losses, v_losses]:
            for key, val in k_loss.items():
                aux_losses[f"attn_{key}"] = aux_losses.get(f"attn_{key}", 0) + val

        # Reshape for attention
        # Q: [batch, seq_len, num_heads * head_dim] -> [batch, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(
            1, 2
        )
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(
            1, 2
        )

        # Apply RoPE
        q = self._apply_rope(q, seq_len)
        k = self._apply_rope(k, seq_len)

        # Handle KV cache for generation
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_key_value = (k, v) if past_key_value is not None else None

        # Group query attention: repeat KV heads
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = F.softmax(attn_scores, dim=-1)

        # Attention output
        attn_output = torch.matmul(
            attn_probs, v
        )  # [batch, num_heads, seq_len, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)

        # Output projection
        output, o_losses = self.o_proj(attn_output)
        for key, val in o_losses.items():
            aux_losses[f"attn_o_{key}"] = val

        return output, present_key_value, aux_losses


class TRILIXSwiGLU(nn.Module):
    """SwiGLU FFN with TRILIX layers"""

    def __init__(self, config: TRILIXConfig):
        super().__init__()

        self.gate_proj = TRILIXLinear(
            config.hidden_size,
            config.intermediate_size,
            rank=config.rank_r,
            codebook_size=config.codebook_k,
            num_atoms=config.num_atoms_A,
            xor_arity=config.xor_arity_b,
            commitment_beta=config.commitment_beta,
            atom_ema_decay=config.atom_ema_decay,
            use_moe=config.use_moe,
            num_experts=config.num_experts if config.use_moe else 4,
            moe_top_k=config.moe_top_k if config.use_moe else 2,
        )

        self.up_proj = TRILIXLinear(
            config.hidden_size,
            config.intermediate_size,
            rank=config.rank_r,
            codebook_size=config.codebook_k,
            num_atoms=config.num_atoms_A,
            xor_arity=config.xor_arity_b,
            commitment_beta=config.commitment_beta,
            atom_ema_decay=config.atom_ema_decay,
            use_moe=config.use_moe,
            num_experts=config.num_experts if config.use_moe else 4,
            moe_top_k=config.moe_top_k if config.use_moe else 2,
        )

        self.down_proj = TRILIXLinear(
            config.intermediate_size,
            config.hidden_size,
            rank=config.rank_r,
            codebook_size=config.codebook_k,
            num_atoms=config.num_atoms_A,
            xor_arity=config.xor_arity_b,
            commitment_beta=config.commitment_beta,
            atom_ema_decay=config.atom_ema_decay,
            use_moe=config.use_moe,
            num_experts=config.num_experts if config.use_moe else 4,
            moe_top_k=config.moe_top_k if config.use_moe else 2,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """SwiGLU: gate * silu(up) -> down"""
        aux_losses = {}

        gate, gate_losses = self.gate_proj(x)
        up, up_losses = self.up_proj(x)

        for k_loss in [gate_losses, up_losses]:
            for key, val in k_loss.items():
                aux_losses[f"ffn_{key}"] = aux_losses.get(f"ffn_{key}", 0) + val

        # SwiGLU
        intermediate = F.silu(gate) * up

        output, down_losses = self.down_proj(intermediate)
        for key, val in down_losses.items():
            aux_losses[f"ffn_down_{key}"] = val

        return output, aux_losses


class TRILIXLayer(nn.Module):
    """Single TRILIX transformer layer"""

    def __init__(self, config: TRILIXConfig):
        super().__init__()

        self.input_layernorm = TRILIXRMSNorm(config.hidden_size)
        self.self_attn = TRILIXAttention(config)
        self.post_attention_layernorm = TRILIXRMSNorm(config.hidden_size)
        self.mlp = TRILIXSwiGLU(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple], Dict]:

        aux_losses = {}

        # Self attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_kv, attn_losses = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
        )
        aux_losses.update(attn_losses)
        hidden_states = residual + hidden_states

        # FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output, mlp_losses = self.mlp(hidden_states)
        aux_losses.update(mlp_losses)
        hidden_states = residual + mlp_output

        return hidden_states, present_kv, aux_losses


class TRILIXTransformer(nn.Module):
    """Full TRILIX Transformer model"""

    def __init__(self, config: TRILIXConfig):
        super().__init__()

        self.config = config

        # Embeddings (kept in BF16 - not compressed)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer layers
        self.layers = nn.ModuleList(
            [TRILIXLayer(config) for _ in range(config.num_hidden_layers)]
        )

        # Final norm
        self.norm = TRILIXRMSNorm(config.hidden_size)

        # LM head (kept in BF16 - not compressed)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # A2: World Model head для предсказания следующего латентного состояния
        # Проецируем hidden_size -> rank_r для предсказания z_next
        self.world_model_head = WorldModelHead(
            r=config.rank_r, hidden_dim=config.rank_r * 2
        )
        self.z_projector = nn.Linear(config.hidden_size, config.rank_r)

        # A3: Belief Gate — убеждения агента о мире
        self.belief_gate = BeliefGate(r=config.rank_r, belief_dim=config.rank_r // 4)

        # C1: EDH — Error-Driven Hypernetwork
        self.edh = ErrorDrivenHypernetwork(
            error_dim=64,
            builder_dim=config.rank_r,
            num_builders=8,
        )

        # C2: REL — Reflective Error Loop
        self.rel = ReflectiveErrorLoop(
            hidden_dim=config.hidden_size,
            uncertainty_dim=16,
        )

        # A1: Soul Codebook — файл "души" агента
        self.soul_codebook = SoulCodebook(num_agents=1024, r=config.rank_r)
        self.soul_projector = nn.Linear(config.rank_r, config.hidden_size)

        # B2: Emergent Agent Swarm — 1024 агента работают как рой
        # Все агенты доступны через SoulCodebook, Swarm добавляет attention
        self.agent_swarm = EmergentAgentSwarm(
            num_agents=1024, r=config.rank_r, num_heads=4
        )

        # B5: SDO — Symbolic Diff Operations для reasoning
        self.symbolic_diff_loss = SymbolicDiffLoss(rank=config.rank_r)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        # Embeddings
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=0.006)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.006)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        labels: Optional[torch.Tensor] = None,
        soul_id: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Forward pass

        Args:
            soul_id: [batch] — ID агента (0..1023). Если None — случайный.

        Returns dict with:
        - logits: [batch, seq_len, vocab_size]
        - loss: total loss including aux losses
        - aux_losses: breakdown of auxiliary losses
        """
        batch_size, seq_len = input_ids.shape

        all_aux_losses = {}

        # Embeddings
        hidden_states = self.embed_tokens(input_ids)

        # A1: Soul Codebook — добавляем "душу" агента к латентному пространству
        # Если soul_id не передан — случайный агент
        if soul_id is None:
            soul_id = torch.randint(
                0, self.soul_codebook.num_agents, (batch_size,), device=input_ids.device
            )
        soul_vector = self.soul_codebook(soul_id)  # [batch, rank]
        soul_vector = self.soul_projector(soul_vector)  # [batch, hidden_size]

        # B2: Emergent Agent Swarm — агент получает attention от других агентов
        if self.training:
            all_soul = self.soul_codebook.soul_vectors.weight  # [1024, rank]
            task_emb = hidden_states.mean(dim=1)  # [batch, hidden]
            task_emb_low = self.z_projector(task_emb)  # [batch, rank]
            task_emb_expanded = task_emb_low.unsqueeze(1).expand(
                -1, all_soul.size(0), -1
            )
            all_soul_expanded = all_soul.unsqueeze(0).expand(batch_size, -1, -1)
            swarm_result = self.agent_swarm(all_soul_expanded, task_emb_low)
            swarm_boost = swarm_result["boosted_soul"]
            agent_idx = (
                soul_id.unsqueeze(-1).expand(-1, self.config.rank_r).unsqueeze(1)
            )
            swarm_soul = torch.gather(swarm_boost, 1, agent_idx).squeeze(1)
            swarm_soul_proj = self.soul_projector(swarm_soul)
            hidden_states = hidden_states + swarm_soul_proj.unsqueeze(1) * 0.05
            all_aux_losses["swarm_specialization"] = swarm_result["specialization_loss"]
        else:
            all_aux_losses["swarm_specialization"] = torch.tensor(
                0.0, device=hidden_states.device
            )

        hidden_states = hidden_states + soul_vector.unsqueeze(1) * 0.1

        # Prepare attention mask
        if attention_mask is not None:
            # Convert to additive mask
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Transformer layers
        present_key_values = [] if past_key_values is None else past_key_values

        for i, layer in enumerate(self.layers):
            past_kv = present_key_values[i] if past_key_values is not None else None

            hidden_states, present_kv, aux_losses = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_kv,
            )

            for key, val in aux_losses.items():
                all_aux_losses[f"layer_{i}_{key}"] = val

            if past_key_values is not None:
                present_key_values.append(present_kv)

        # Final norm
        hidden_states = self.norm(hidden_states)

        # LM head
        logits = self.lm_head(hidden_states)

        # Calculate losses
        total_loss = None
        if labels is not None:
            # Main CE loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            ce_loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)
            )

        # A2: World Model loss — предсказание следующего латентного состояния
        # Среднее по sequence для каждого элемента батча
        z = hidden_states.mean(dim=1)  # [batch, hidden_size]
        z_proj = self.z_projector(z)  # [batch, rank]

        # z_next — следующий токен (среднее по sequence на 1 шаг вперёд)
        z_next = hidden_states[:, 1:, :].mean(dim=1)  # [batch, hidden_size]
        z_next_proj = self.z_projector(z_next)  # [batch, rank]

        # Предсказание
        z_pred = self.world_model_head(z_proj)  # [batch, rank]

        # A3: Belief Gate — модулируем предсказание на основе убеждений агента
        belief_result = self.belief_gate(
            z_proj, z_pred, z_next_proj, return_belief=False
        )
        gated_pred = belief_result["gated_pred"]
        belief_loss = belief_result["belief_loss"]

        # World Model loss — MSE между предсказанным и реальным следующим состоянием
        world_model_loss = F.mse_loss(
            gated_pred, z_next_proj.detach()
        )  # detach чтобы не мешать CE

        # Auxiliary losses
        aux_loss_sum = sum(all_aux_losses.values())

        # Note: diversity loss is already included in aux_loss_sum as atom_diversity
        # We extract it for logging only
        diversity_loss = sum(
            all_aux_losses.get(f"layer_{i}_atom_diversity", torch.tensor(0.0))
            for i in range(self.config.num_hidden_layers)
        )

        # A2: World Model loss в сумму aux losses (с весом 0.1 чтобы не доминировать)
        aux_loss_sum = aux_loss_sum + 0.1 * world_model_loss

        # A3: Belief Gate loss
        aux_loss_sum = aux_loss_sum + 0.05 * belief_loss

        # B5: SDO — Symbolic Diff Operations для reasoning
        # Семплируем codebook из первого слоя
        codebook_U_sdo = None
        codebook_V_sdo = None
        for layer in self.layers:
            if codebook_U_sdo is None:
                trilix_layer = layer.self_attn.q_proj
                if hasattr(trilix_layer, "_get_combo_indices_hard"):
                    atoms_U_bin = STEBinary.apply(trilix_layer.atoms_U)
                    atoms_V_bin = STEBinary.apply(trilix_layer.atoms_V)
                    combo_idx_U = trilix_layer._get_combo_indices_hard(
                        trilix_layer.combo_indices_U_logits
                    )
                    combo_idx_V = trilix_layer._get_combo_indices_hard(
                        trilix_layer.combo_indices_V_logits
                    )
                    codebook_U_sdo = trilix_layer._decode_codebook_entry(
                        trilix_layer.combo_weights_U, combo_idx_U, atoms_U_bin
                    )
                    codebook_V_sdo = trilix_layer._decode_codebook_entry(
                        trilix_layer.combo_weights_V, combo_idx_V, atoms_V_bin
                    )
                break

        sdo_loss = torch.tensor(0.0, device=hidden_states.device)
        if codebook_U_sdo is not None:
            sdo_loss = self.symbolic_diff_loss(codebook_U_sdo, codebook_V_sdo)
            aux_loss_sum = aux_loss_sum + sdo_loss

        # C1: EDH — Error-Driven Hypernetwork
        edh_result = self.edh(
            ce_loss=ce_loss,
            world_model_loss=world_model_loss,
            diversity_loss=diversity_loss,
            belief_loss=belief_loss,
        )
        edh_loss = edh_result["builder_loss"]
        aux_loss_sum = aux_loss_sum + 0.05 * edh_loss

        # C2: REL — Reflective Error Loop
        if labels is not None:
            shift_logits_for_rel = logits[..., :-1, :].contiguous()
            shift_labels_for_rel = labels[..., 1:].contiguous()
            rel_result = self.rel(
                hidden_states=hidden_states,
                logits=shift_logits_for_rel,
                labels=shift_labels_for_rel,
            )
            rel_loss = rel_result["reflective_loss"]
        else:
            rel_loss = torch.tensor(0.0, device=hidden_states.device)
        aux_loss_sum = aux_loss_sum + 0.05 * rel_loss

        total_loss = ce_loss + aux_loss_sum

        all_aux_losses["ce_loss"] = ce_loss
        all_aux_losses["total_aux"] = aux_loss_sum
        all_aux_losses["diversity_total"] = diversity_loss
        all_aux_losses["world_model_loss"] = world_model_loss
        all_aux_losses["sdo_loss"] = sdo_loss
        all_aux_losses["belief_loss"] = belief_loss
        all_aux_losses["edh_loss"] = edh_loss
        all_aux_losses["rel_loss"] = rel_loss

        return {
            "logits": logits,
            "loss": total_loss,
            "aux_losses": all_aux_losses,
            "past_key_values": present_key_values,
        }

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Simple greedy generation"""

        past_key_values = None

        for _ in range(max_new_tokens):
            # Forward
            outputs = self.forward(
                input_ids if past_key_values is None else input_ids[:, -1:],
                past_key_values=past_key_values,
            )

            logits = outputs["logits"][:, -1, :]  # Last token

            # Temperature scaling
            logits = logits / temperature

            # Top-k
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if past_key_values is not None:
                past_key_values = outputs["past_key_values"]

        return input_ids

    def get_memory_stats(self) -> Dict:
        """Get memory statistics for the model"""
        stats = {
            "embeddings_mb": self.config.vocab_size
            * self.config.hidden_size
            * 2
            / 1e6,  # BF16
            "lm_head_mb": self.config.vocab_size * self.config.hidden_size * 2 / 1e6,
        }

        # TRILIX layers stats
        total_eval_bits = 0
        total_train_bits = 0

        for layer in self.layers:
            for proj in [
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
                layer.self_attn.o_proj,
            ]:
                proj_stats = proj.count_parameters()
                total_eval_bits += proj_stats["total_eval_bits"]
                total_train_bits += proj_stats["total_train_bits"]

            for proj in [layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj]:
                proj_stats = proj.count_parameters()
                total_eval_bits += proj_stats["total_eval_bits"]
                total_train_bits += proj_stats["total_train_bits"]

        # Original size
        num_params = self.config.num_hidden_layers * (
            4 * self.config.hidden_size * self.config.hidden_size  # attn
            + 3 * self.config.hidden_size * self.config.intermediate_size  # ffn
        )
        original_bits = num_params * 16

        stats["trilix_eval_mb"] = total_eval_bits / 8 / 1e6
        stats["trilix_train_mb"] = total_train_bits / 8 / 1e6
        stats["original_mb"] = original_bits / 8 / 1e6
        stats["bpw_eval"] = total_eval_bits / original_bits
        stats["bpw_train"] = total_train_bits / original_bits
        stats["compression_ratio"] = original_bits / total_eval_bits

        return stats
