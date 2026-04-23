# TRILIX-LFT — Roadmap инноваций

> **Проект**: TRILIX-LFT — Трансформер с экстремальным сжатием до 0.0048 BPW  
> **Автор**: Evgeny  
> **Дата**: 2026-04-23  
> **Статус**: Активная разработка

---

## Что сделано

### Инновации Claude (реализованы, работают)

| # | Инновация | Статус | Коммит |
|---|-----------|--------|--------|
| 1 | **SAIB** — Spectral Atom Init + EMA | ✅ Работает | `e39abf7` |
| 2 | **RVQ** — Residual Vector Quantization | ✅ Работает | `e39abf7` |
| 3 | **SGH** — Semantic Gradient Highway | ✅ Работает | `e39abf7` |
| 4 | **ATC** — Adaptive Temperature Cascade | ✅ Работает (был в оригинале) | — |
| 5 | **LCC** — Learned Codebook Compressor | ✅ Работает | `e39abf7` |

### Исправления

| # | Что | Было | Стало | Коммит |
|---|-----|------|-------|--------|
| 1 | commitment_beta | 0.25 (взрывало loss=1127) | 0.0001 (loss=~592) | `0fc0be4` |

---

## Очередь на внедрение

### Группа A — Простые и мощные (1-2 дня на внедрение)

#### A1: Soul Codebook
**Источник**: Гемини 3.1 (Инновация 1)  
**Автор**: Гемини 3.1 Pro  
**Статус**: ✅ ЗАВЕРШЕНО (коммит 1bbb49f) — 2026-04-23

**Суть**: Файл "души" агента — отдельный обучаемый вектор, подключаемый к латентному пространству.  
Один TRILIX становится 1000+ разными агентами.

**Как внедрить**:
```python
# В layers.py добавить класс SoulCodebook

class SoulCodebook(nn.Module):
    def __init__(self, num_agents: int = 1000, r: int = 100):
        super().__init__()
        self.soul_vectors = nn.Embedding(num_agents, r)  # [1000, r]
    
    def forward(self, soul_id: int) -> torch.Tensor:
        return self.soul_vectors(soul_id)

# В TRILIXLinear.forward():
# x_latent = x_latent + soul_vector  # до роутера MoE
```

**Чем улучшит TRILIX**:
- Мгновенное переключение контекста (Python ↔ Шекспир ↔ Математика)
- Один и тот же набор весов выдаёт разные "личности"
- 1000 агентов в 1 модели

**Файлы**: `layers.py`, `model.py`, `train.py`

**Приоритет**: 🔴 ВЫСОКИЙ

---

#### A2: Latent World Model
**Источник**: Дипсик (Инновация 3)  
**Автор**: Дипсик  
**Статус**: ✅ ЗАВЕРШЕНО (коммит bc4006b) — 2026-04-23

**Суть**: Обучить латентное пространство r предсказывать собственное будущее.  
Латентное пространство становится "физическим движком", а не просто складом.

**Как внедрить**:
```python
# В model.py добавить WorldModelHead

class WorldModelHead(nn.Module):
    def __init__(self, r: int):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(r, r * 2),
            nn.ReLU(),
            nn.Linear(r * 2, r)
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [batch, r] — латентное состояние
        return self.predictor(z)

# В training loop:
# loss_world = F.mse_loss(world_model(z), z_next)
```

**Чем улучшит TRILIX**:
- Улучшенное обобщение (сеть понимает причинно-следственную динамику)
- Галлюцинации упадут до нуля
- SGH работает лучше (градиенты станут осмысленными)

**Файлы**: `layers.py` (WorldModelHead), `model.py`, `train.py`

**Приоритет**: 🔴 ВЫСОКИЙ

---

### Группа B — Архитектурные улучшения (3-4 дня)

#### B1: Recursive MoE
**Источник**: Дипсик (Инновация 1)  
**Автор**: Дипсик

**Суть**: Иерархия экспертов. Мета-роутер → дочерний роутер.  
4×4 = 16 "виртуальных" экспертов при тех же затратах памяти.

**Как внедрить**:
```python
class RecursiveMoECodebook(nn.Module):
    def __init__(self, depth: int, num_experts_per_level: int, k: int, r: int, top_k: int):
        super().__init__()
        self.depth = depth
        if depth == 1:
            self.experts = nn.ModuleList([
                CodebookExpert(k=k, r=r) for _ in range(num_experts_per_level)
            ])
            self.router = nn.Linear(r, num_experts_per_level)
        else:
            self.children = nn.ModuleList([
                RecursiveMoECodebook(depth-1, num_experts_per_level, k, r, top_k)
                for _ in range(num_experts_per_level)
            ])
            self.router = nn.Linear(r, num_experts_per_level)
    
    def forward(self, x_latent: torch.Tensor):
        # Мета-роутер выбирает группу экспертов
        meta_weights = F.softmax(self.router(x_latent), dim=-1)
        # Дочерний роутер уточняет выбор
        ...
```

**Чем улучшит TRILIX**:
- Экспоненциальный рост ёмкости (depth=2, 4 эксперта → 16 комбинаций)
- Более качественный роутинг (синтаксис → подлежащее → глагол)
- Уменьшает load balancing loss

**Файлы**: `layers.py` (RecursiveMoECodebook)

**Приоритет**: 🟡 СРЕДНИЙ

---

#### B2: Emergent Agent Swarm
**Источник**: Дипсик (Инновация 2)  
**Автор**: Дипсик

**Суть**: Gumbel-Sigmoid вместо softmax. Эксперты конкурируют за выживание.  
"Сильные" получают больше градиента, "мёртвые" заменяются.

**Как внедрить**:
```python
# В MoECodebook.forward() заменить softmax роутер на:

logits = self.router(x_latent)
gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
y_soft = torch.sigmoid((logits + gumbel_noise) / temperature)
y_hard = (y_soft > 0.5).float()
mask = y_hard - y_soft.detach() + y_soft  # STE

# Evolutionary Pressure Loss:
loss_ep = -sum(usage_count[e] * reward[e] for e in experts)
# reward[e] = -commitment_loss эксперта
```

**Чем улучшит TRILIX**:
- Решает проблему "мёртвых атомов"
- Динамическая архитектура (0, 1, 3 эксперта на токен)
- Сеть сама выращивает специализации

**Файлы**: `layers.py` (MoECodebook), `train.py`

**Приоритет**: 🟡 СРЕДНИЙ

---

#### B3: Differentiable Atom Evolution (DAE)
**Источник**: Дипсик (Инновация 4)  
**Автор**: Дипсик

**Суть**: Эволюция атомов раз в N шагов. Мутанты заменяют originals если лучше.  
STE для дифференцируемости.

**Как внедрить**:
```python
# В новый файл evolution.py

class EvolutionScheduler:
    def __init__(self, evolution_freq: int = 1000, mutation_rate: float = 0.01):
        self.evolution_freq = evolution_freq
        self.mutation_rate = mutation_rate
    
    def evolution_step(self, model):
        # 1. Создать мутантов
        atoms_mutated = atoms + noise * mutation_rate
        
        # 2. Вычислить loss для мутантов
        loss_mutated = compute_loss(model, atoms_mutated)
        loss_original = compute_loss(model, atoms)
        
        # 3. STE замена
        atoms_new = atoms + (atoms_mutated - atoms).detach() * (loss_mutated < loss_original).float()
```

**Чем улучшит TRILIX**:
- Обход локальных минимумов (эволюция выпрыгивает из плато)
- Глобально оптимальный базис (не зависит от инициализации)
- Решает проблему "мёртвых атомов" навсегда

**Файлы**: `evolution.py` (новый), `train.py`

**Приоритет**: 🟡 СРЕДНИЙ

---

#### B4: Darwinian Gradient Swarm
**Источник**: Гемини (Инновация 3)  
**Автор**: Гемини 3.1 Pro

**Суть**: Биологическая эволюция прямо в латентном пространстве.  
Конкуренция экспертов за градиент. Выживает сильнейший.

**Как внедрить**:
```python
# В MoECodebook добавить:
self.register_buffer("usage_counter", torch.zeros(num_experts))

# Раз в N шагов:
# 1. Отсортировать эксперты по Utility Score
# 2. Убить нижние 10% (заменить на копии топ-10% + мутация)
# 3. Обнулить счётчики использования
```

**Чем улучшит TRILIX**:
- Сеть становится самоочищающейся
- Автоматическая специализация экспертов
- Шаг к истинному AGI

**Файлы**: `layers.py` (MoECodebook), `train.py`

**Приоритет**: 🟡 СРЕДНИЙ (пересекается с B2)

---

#### B5: Physics-Constrained Manifold
**Источник**: Гемини (Инновация 5)  
**Автор**: Гемини 3.1 Pro

**Суть**: Contrastive Logic Loss. Латентное пространство подчиняется законам логики.  
Если "яблоко в коробке" + "коробка перевёрнута" → яблоко должно быть вне коробки.

**Как внедрить**:
```python
# Contrastive Logic Loss:
# Обучаем на тройках: (состояние, действие, результат)
# loss_logic = || latent(state) + latent(action) - latent(result) ||^2
# Если результат нарушает логику → штраф ДО генерации текста
```

**Чем улучшит TRILIX**:
- Галлюцинации → 0
- Внутренняя "модель мира" (World Model)
- Генерация текста через симуляцию, а не "слова рядом в интернете"

**Файлы**: `train.py`

**Приоритет**: 🟡 СРЕДНИЙ

---

### Группа C — Продвинутые (5+ дней)

#### C1: Recursive Self-Compilation Loops
**Источник**: Дипсик (Инновация 5)  
**Автор**: Дипсик

**Суть**: Builder Expert — гиперсеть, генерирующая кодбуки на лету.  
Zero-shot специализация (новый язык, новая задача без обучения).

**Как внедрить**:
```python
class BuilderExpert(nn.Module):
    def __init__(self, task_embed_dim: int, k: int, r: int):
        super().__init__()
        self.hypernet = nn.Sequential(
            nn.Linear(task_embed_dim, r * 2),
            nn.ReLU(),
            nn.Linear(r * 2, k * r)
        )
    
    def forward(self, task_embedding: torch.Tensor) -> torch.Tensor:
        return self.hypernet(task_embedding).view(k, r)
```

**Чем улучшит TRILIX**:
- Zero-shot способности (новые языки, задачи)
- Бесконечная масштабируемость (кодбуки создаются и уничтожаются)
- Суб-агенты с уникальными способностями

**Файлы**: `layers.py`

**Приоритет**: 🟠 ПРОДВИНУТЫЙ

---

#### C2: Dynamic Ephemeral Experts
**Источник**: Гемини (Инновация 2)  
**Автор**: Гемини 3.1 Pro

**Суть**: "Фабрика суб-агентов" — Meta-Router генерирует новые комбинации атомов прямо во время inference.  
Одноразовый эксперт для конкретного токена.

**Проблема**: TRILIX использует дискретные индексы — inference-time генерация НЕ дифференцируема.

**Решение**: Сделать отдельный forward pass для "фабрики":
```python
# Фаза 1: Обычный forward для простых токенов
# Фаза 2: Генерация нового эксперта для сложных токенов (если задача требует)
```

**Чем улучшит TRILIX**:
- Бесконечная масштабируемость агентов при нулевом росте VRAM
- Микро-агенты для проверки фактов, расчётов — в одном forward pass

**Файлы**: `layers.py`

**Приоритет**: 🟠 ПРОДВИНУТЫЙ

---

#### C3: Meta-Reflective Mutation
**Источник**: Гемини (Инновация 4)  
**Автор**: Гемини 3.1 Pro

**Суть**: Рефлексивный контур — 2 последних слоя предсказывают Loss сети.  
Сеть "думает": "Сейчас скажу чушь, надо перестроить веса".

**Проблема**: Требует 2 forward pass — предсказать loss до генерации.

**Как внедрить**:
```python
# Выделить последние 2 слоя в "Рефлексивный контур"
# Рефлексивный контур: Loss = model.predict_loss(hidden_states)
# gradient_injection: корректирует combo_indices ещё ДО генерации
```

**Чем улучшит TRILIX**:
- Зачатки искусственного сознания
- Сеть учится "перестраиваться" перед ответом

**Файлы**: `layers.py`, `model.py`

**Приоритет**: 🟠 ПРОДВИНУТЫЙ

---

## Мои 5 инноваций (не от Гемини/Дипсика)

### M1: Temporal Atom Memory (TAM)

**Суть**: Запоминание успешных комбинаций атомов во времени.  
Атомы, которые "выиграли" в прошлом, получают буст в будущем.

**Как внедрить**:
```python
# Добавить Temporal Memory Buffer
class TemporalAtomMemory:
    def __init__(self, num_atoms: int, memory_size: int = 1000):
        self.success_count = torch.zeros(num_atoms)
        self.memory_queue = deque(maxlen=memory_size)
    
    def record_success(self, atom_indices: torch.Tensor, reward: float):
        for idx in atom_indices:
            self.success_count[idx] += reward
    
    def get_boost(self) -> torch.Tensor:
        # Буст для популярных атомов (но не слишком популярных)
        boost = 1.0 / (1.0 + self.success_count)
        return boost
```

**Чем улучшит TRILIX**:
- Быстрое восстановление после плохих комбинаций
- "Обучение из опыта" без градиентов
- Решает проблему "забывания" успешных стратегий

**Приоритет**: 🟡 СРЕДНИЙ

---

### M2: Hierarchical Codebook Pruning (HCP)

**Суть**: Иерархический прунинг codebook. Сначала удаляем "бесполезные" группы атомов, потом — внутри групп.

**Как внедрить**:
```python
# Group-level pruning
# Если группа атомов не использовалась 10K шагов → вся группа в резерв
# Резервные группы активируются только при катастрофическом забывании

group_usage = torch.zeros(num_groups)
group_threshold = 10000

if (step % group_check_interval == 0):
    dead_groups = group_usage < 1.0
    reserve_groups[dead_groups] = atoms[dead_groups]
    atoms[dead_groups] = 0.0
```

**Чем улучшит TRILIX**:
- Автоматическая "уборка мусора" в codebook
- Быстрое переобучение на новые данные
- Резерв для катастрофического забывания

**Приоритет**: 🟡 СРЕДНИЙ

---

### M3: Cross-Layer Codebook Sharing (CLCS)

**Суть**: Разные слои используют общие codebook entries для общих понятий.  
"Слово" на слое 1 то же самое, что "слово" на слое 24.

**Как внедрить**:
```python
# Глобальный codebook для всех слоёв
shared_codebook_U = nn.Parameter(torch.randn(k, r))

# Per-layer refinement
class SharedCodebookLayer(nn.Module):
    def __init__(self, layer_id: int):
        self.refinement = nn.Parameter(torch.randn(k, r) * 0.01)
        # final_codebook = shared_codebook + alpha * refinement
    
    def forward(self):
        alpha = 0.1  # уменьшается с глубиной
        return shared_codebook_U + alpha * self.refinement
```

**Чем улучшит TRILIX**:
- Меньше памяти (shared codebook)
- Лучшее обобщение (общие понятия = общие веса)
- transfer learning между слоями

**Приоритет**: 🟡 СРЕДНИЙ

---

### M4: Adversarial Codebook Perturbation (ACP)

**Суть**: Adversarial training для codebook.  
Добавляем "враждебный шум" к codebook entries, чтобы сделать их устойчивыми к атакам.

**Как внедрить**:
```python
# FGSM-style perturbation
def adversarial_perturbation(codebook: torch.Tensor, epsilon: float = 0.1):
    noise = torch.randn_like(codebook) * epsilon
    codebook_adv = codebook + noise
    return codebook_adv

# Adversarial loss
loss_adv = F.mse_loss(model(x, codebook_adv), target) - F.mse_loss(model(x, codebook), target)
```

**Чем улучшит TRILIX**:
- Устойчивость к adversarial attacks
- Более "размазанные" codebook entries (лучше generalization)
- Неожиданные комбинации атомов

**Приоритет**: 🟠 ПРОДВИНУТЫЙ

---

### M5: Curriculum Atom Learning (CAL)

**Суть**: Curriculum learning для атомов.  
Сначала учим простые атомы (2 атома → слово), потом сложные (16 атомов → смысл).

**Как внедрить**:
```python
class CurriculumScheduler:
    def __init__(self, total_steps: int):
        self.total = total_steps
    
    def get_xor_arity(self, step: int) -> int:
        progress = step / self.total
        # От 2 до 3 атомов за 50% обучения
        if progress < 0.5:
            return 2
        else:
            return 3
    
    def get_num_atoms(self, step: int) -> int:
        progress = step / self.total
        # От 16 до 32 атомов за 100% обучения
        return int(16 + progress * 16)
```

**Чем улучшит TRILIX**:
- Более стабильное обучение
- Быстрая сходимость на простых паттернах
- Постепенный переход к сложным

**Приоритет**: 🟡 СРЕДНИЙ

---

## Что НЕ реализуем пока

| Идея | Почему | Когда можно |
|------|--------|-------------|
| **Dynamic Ephemeral Experts** (Гемини) | TRILIX использует дискретные индексы — inference-time генерация НЕ дифференцируема. Нужен отдельный forward для "фабрики". НЕ совместимо с текущим пайплайном. | После стабилизации базового TRILIX |
| **Meta-Reflective Mutation** (Гемини) | Требует 2 forward pass — предсказать loss до генерации. Это возможно, но сложно интегрировать. Добавляет 2x latency. | После внедрения World Model |
| **Recursive Self-Compilation Loops** (Дипсик) | Builder Expert — это hypernet, генерирующий кодбуки. Требует отдельного обучения. Оверхед. | После внедрения всех инноваций A и B |
| **Zero-shot Agent Switching** (Гемини) | Требует Soul Codebook (A1) — зависимость | После A1 |

---

## План реализации

```
Приоритет 1 (начать завтра):
├── A1: Soul Codebook          (~2 дня)
│   └── files: layers.py, model.py, train.py
│
├── A2: Latent World Model     (~2 дня)
│   └── files: layers.py, model.py, train.py
│
Приоритет 2 (через 1-2 недели):
├── B1: Recursive MoE          (~4 дня)
├── B2: Emergent Agent Swarm   (~3 дня)
├── B3: Differentiable Atom Evolution (~4 дня)
│
Приоритет 3 (через месяц):
├── B4: Darwinian Gradient Swarm
├── B5: Physics-Constrained Manifold
├── M1: Temporal Atom Memory
├── M2: Hierarchical Codebook Pruning
├── M3: Cross-Layer Codebook Sharing
│
Приоритет 4 (продвинутые):
├── C1: Recursive Self-Compilation Loops
├── C2: Dynamic Ephemeral Experts
├── C3: Meta-Reflective Mutation
├── M4: Adversarial Codebook Perturbation
└── M5: Curriculum Atom Learning
```

---

## Метрики успеха

| Инновация | Метрика | Целевое значение |
|-----------|---------|------------------|
| Все | Loss | < 8.0 (сейчас ~592) |
| Все | BPW | < 0.0048 |
| Все | Скорость | 3-5 сек/шаг |
| Soul Codebook | Количество агентов | 1000+ |
| World Model | Logic Accuracy | > 90% |
| Recursive MoE | Эффективная ёмкость | ×16 |
| Agent Swarm | Dead atoms | < 5% |
| DAE | Улучшение loss после evolution | > 5% |

---

## Версии файлов

- `trilix/layers.py` — основной код TRILIXLinear + инновации
- `trilix/config.py` — конфигурация
- `trilix/model.py` — TRILIXTransformer
- `train_small_moe.py` — скрипт обучения

---

*Документ создан для отслеживания идей и их статуса. Обновлять после каждого коммита.*