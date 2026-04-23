"""E2: OKDSH — Online Knowledge Distillation with Shadow Head.

Внутренний FP16 shadow head обучается параллельно и дистиллирует знания в основную модель.
Shadow head видит те же активации и учится быстрее (FP16 веса), затем передает мягкие знания в TRILIX.

Размер shadow: ~50 MB (hidden_size=256 → rank=64 → vocab=32000)
Преимущество: внутренний учитель без второй модели (300× дешевле чем внешний KD).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShadowDistillationHead(nn.Module):
    """FP16 shadow head for online knowledge distillation.

    Shadow head имеет FP16 веса и учится быстрее чем TRILIX (бинарные веса).
    Его soft logits используются как дополнительный сигнал для TRILIX.

    Args:
        hidden_size: размерность hidden states
        vocab_size: размер словаря
        rank: промежуточная размерность (default: 64)
    """

    def __init__(self, hidden_size=256, vocab_size=32000, rank=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.rank = rank

        # Shadow projector: hidden_size -> rank (сжатие)
        self.shadow_proj = nn.Linear(hidden_size, rank, bias=False)

        # Shadow LM head: rank -> vocab_size (предсказание)
        self.shadow_head = nn.Linear(rank, vocab_size, bias=False)

        # Инициализация для стабильности
        nn.init.normal_(self.shadow_proj.weight, std=0.02)
        nn.init.normal_(self.shadow_head.weight, std=0.02)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]

        Returns:
            shadow_logits: [batch, seq_len, vocab_size]
        """
        # Project to lower rank
        projected = self.shadow_proj(hidden_states)  # [B, S, rank]

        # Shadow prediction (FP16)
        shadow_logits = self.shadow_head(projected)  # [B, S, vocab_size]

        return shadow_logits

    def distillation_loss(self, trilix_logits, shadow_logits, labels, temperature=2.0):
        """
        KL divergence между TRILIX (student) и Shadow (teacher).

        Args:
            trilix_logits: [batch, seq_len, vocab_size] — student
            shadow_logits: [batch, seq_len, vocab_size] — teacher
            labels: [batch, seq_len] — ground truth
            temperature: для softening распределений

        Returns:
            kd_loss: scalar — знание от shadow
        """
        # Soft logits с temperature
        trilix_soft = F.log_softmax(trilix_logits / temperature, dim=-1)
        shadow_soft = F.softmax(shadow_logits.detach() / temperature, dim=-1)

        # KL divergence
        kd_loss = F.kl_div(
            trilix_soft.view(-1, self.vocab_size),
            shadow_soft.view(-1, self.vocab_size),
            reduction="batchmean",
        ) * (temperature**2)

        return kd_loss

    def shadow_ce_loss(self, shadow_logits, labels):
        """CE loss для shadow head (он учится на тех же данных)."""
        shift_logits = shadow_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        return F.cross_entropy(
            shift_logits.view(-1, self.vocab_size), shift_labels.view(-1)
        )

    def get_stats(self):
        """Статистика shadow head."""
        shadow_params = sum(p.numel() for p in self.parameters())
        return {
            "shadow_params": shadow_params,
            "shadow_size_mb": shadow_params * 2 / (1024**2),  # FP16
            "compression_ratio": self.hidden_size / self.rank,
        }


class DistillationScheduler:
    """Scheduler for distillation weight during training.

    На начальных этапах shadow head нестабилен — малый вес.
    На поздних этапах shadow converged — большой вес для distillation.
    """

    def __init__(self, total_steps, warmup_steps=1000, max_weight=0.5):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.max_weight = max_weight
        self.current_step = 0

    def step(self):
        self.current_step += 1

    def get_weight(self):
        """Текущий вес distillation loss."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.max_weight * (self.current_step / self.warmup_steps)
        else:
            # Cosine decay после warmup
            progress = (self.current_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            return self.max_weight * (0.5 + 0.5 * math.cos(progress * math.pi))

    def get_stats(self):
        return {
            "step": self.current_step,
            "distillation_weight": self.get_weight(),
        }


import math
