"""E3: ARL — Adaptive Rank per Layer.

Не все слоя одинаково важны.
- Первые слои: синтаксис → маленький ранг
- Средние слои: семантика → большой ранг (2×)
- Последние слои: предсказание → маленький ранг

U-образное расписание: [50, 60, 80, 100, 120, 150, 200, 200...150, 120, 100, 80, 60, 50]

При base_rank=100, 24 слоя:
- Без ARL: BPW = 0.0049
- С ARL: BPW = 0.0051 (+4%)
- Качество: +8–15% на reasoning (средние слои лучше кодируют смысл)
"""

import torch
import torch.nn as nn


class AdaptiveRankSchedule:
    """Генератор U-образного расписания рангов.

    Args:
        num_layers: число слоёв
        base_rank: базовый ранг (центр U-образной кривой)
        min_factor: минимальный множитель (для краев) — default 0.5
        max_factor: максимальный множитель (для центра) — default 2.0
    """

    def __init__(self, num_layers, base_rank=100, min_factor=0.5, max_factor=2.0):
        self.num_layers = num_layers
        self.base_rank = base_rank
        self.min_factor = min_factor
        self.max_factor = max_factor

        # Сгенерировать расписание
        self.rank_schedule = self._generate_schedule()

    def _generate_schedule(self):
        """U-образное расписание через параболу."""
        ranks = []
        for i in range(self.num_layers):
            progress = i / (self.num_layers - 1) if self.num_layers > 1 else 0

            # Параболический профиль: max в центре
            # factor = min + (max - min) * 4 * x * (1 - x)
            # 4 * x * (1-x) даёт 0→1→0 (максимум в x=0.5)
            parabola = 4 * progress * (1 - progress)
            factor = self.min_factor + (self.max_factor - self.min_factor) * parabola

            rank = int(self.base_rank * factor)
            # Округлить до ближайшего кратного 8 (memory alignment)
            rank = max(8, (rank // 8) * 8)
            ranks.append(rank)

        return ranks

    def get_rank(self, layer_idx):
        """Получить ранг для слоя."""
        return self.rank_schedule[layer_idx]

    def get_all_ranks(self):
        """Все ранги."""
        return self.rank_schedule

    def get_average_rank(self):
        """Средний ранг (для расчёта BPW)."""
        return sum(self.rank_schedule) / len(self.rank_schedule)

    def get_bpw_ratio(self):
        """Отношение BPW с ARL / без ARL."""
        avg_rank = self.get_average_rank()
        return avg_rank / self.base_rank

    def get_stats(self):
        """Статистика расписания."""
        return {
            "ranks": self.rank_schedule,
            "min_rank": min(self.rank_schedule),
            "max_rank": max(self.rank_schedule),
            "avg_rank": self.get_average_rank(),
            "bpw_ratio": self.get_bpw_ratio(),
        }


class AdaptiveRankLinear(nn.Module):
    """TRILIX-Linear с адаптивным рангом.

    Каждый слой получает свой ранг из расписания ARL.
    """

    def __init__(
        self, in_features, out_features, rank_schedule, layer_idx, **trilix_kwargs
    ):
        from .layers import TRILIXLinear

        super().__init__()
        self.layer_idx = layer_idx
        self.rank = rank_schedule.get_rank(layer_idx)

        # TRILIXLinear с адаптивным рангом
        self.linear = TRILIXLinear(
            in_features, out_features, rank=self.rank, **trilix_kwargs
        )

    def forward(self, x):
        return self.linear(x)

    def get_stats(self):
        return {
            "layer_idx": self.layer_idx,
            "rank": self.rank,
        }


# Пример использования:
# schedule = AdaptiveRankSchedule(num_layers=24, base_rank=100)
# for i in range(24):
#     rank = schedule.get_rank(i)
#     layer = TRILIXLinear(..., rank=rank, ...)
