# TRILIX-LFT — Roadmap инноваций

> **Проект**: TRILIX-LFT — Трансформер с экстремальным сжатием до 0.0048 BPW  
> **Автор**: Evgeny  
> **Дата**: 2026-04-23 (обновлён 2026-04-23 вечер, A3 BeliefGate)  
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
| A1 | **Soul Codebook** — 1024 агента в 1 модели | ✅ Работает | `1bbb49f` |
| A2 | **Latent World Model** — предсказание следующего состояния | ✅ Работает | `bc4006b` |
| A3 | **Belief Gate** — убеждения агента о мире | ✅ Работает | — |
| B5 | **SDO** — Symbolic Diff Operations | ✅ Работает | `5421917` |
| B4 | **HAR** — Hebbian Atom Resonance | ✅ Работает | `d7459fc` |
| B3 | **DAE** — Differentiable Atom Evolution | ✅ Работает | `9985cda` |
| B1.5 | **FHC** — Flat Hierarchical Codebook | ✅ Работает | `7abc392` |
| B2 | **Agent Swarm** — 1024 агента как рой | ✅ Работает | `7989d06` |

### Новые цели (документы)

| Документ | Описание |
|----------|----------|
| `GOALS.md` | Главные цели проекта: 70B+, 1M context, победить Claude/GPT/Gemini |
| `SOUL.md` | Формат ACTUAL агента с Skills Stack + Meta-Skill Creator |
| `skills/` | Структура скиллов (python_expert, meta_skill_creator) |
| `soul/` | Директория для SOUL.md файлов агентов |

### Исправления

| # | Что | Было | Стало | Коммит |
|---|-----|------|-------|--------|
| 1 | commitment_beta | 0.25 (взрывало loss=1127) | 0.0001 (loss=~592) | `0fc0be4` |

---

## Анализ Клода (ключевые выводы)

Клод дал **экспертный анализ 9/10**. Главные выводы:

### Клод правильно понял
1. **TRILIX — символически-нейронный гибрид** (веса = дискретные символы, индексы в кодбуке)
2. **Инновации нужно объединять** — они компоненты одной системы, не изолированные модули
3. **Meta-Reflective Mutation сломана** — конфликт двух градиентных сигналов
4. **Builder Expert — circular dependency** — нужен task_embedding, который получается из сети

### Что отверг Клод
- **UAS объединять НЕ нужно** — Soul и WorldModel уже работают. Добавить Belief Gate в WorldModel.
- **R-MoE как дерево — плохо** — нужно flat hierarchical (один matmul)
- **EDH слишком сложно** — сначала простые инновации

---

## Инновации Клода (приоритеты)

| # | Инновация | Оценка | Приоритет | Статус |
|---|-----------|--------|-----------|--------|
| HAR | **Hebbian Atom Resonance** — обучение без градиентов | ⭐ 10/10 | B4 🔴 | Запланировано |
| SDO | **Symbolic Diff Operations** — аналогии через XOR | ⭐ 9/10 | B5 🔴 | Запланировано |
| FHC | **Flat Hierarchical Codebook** — правильная R-MoE | 9/10 | B1.5 🟡 | Запланировано |
| EDH | **Error-Driven Hypernetwork** — Builder Expert без circular dependency | 7/10 | C1 🟠 | Не сейчас |
| REL | **Reflective Error Loop** — "я сомневаюсь здесь" | 8/10 | C2 🟠 | Не сейчас |
| UAS-Belief | **Belief Gate** — добавить в WorldModel | 8/10 | Дополнить A2 | Запланировано |

---

## Очередь на внедрение (обновлённая)

### Группа A — ✅ Завершено

| # | Инновация | Статус | Коммит |
|---|-----------|--------|--------|
| A1 | **Soul Codebook** | ✅ ЗАВЕРШЕНО | `1bbb49f` |
| A2 | **Latent World Model** | ✅ ЗАВЕРШЕНО | `bc4006b` |
| A3 | **Belief Gate** | ✅ ЗАВЕРШЕНО | — |

---

### Группа B — Внедряем

#### B3: Differentiable Atom Evolution (DAE)
**Источник**: Дипсик  
**Статус**: ✅ ЗАВЕРШЕНО — 2026-04-23

**Суть**: Дифференцируемая эволюция — градиенты управляют мутациями атомов.

**Как работает:**
- HAR: безградиентное, статистическое, "выживают популярные"
- DAE: градиентное, selection pressure через ||d_loss/d_atom||
- Mutation: atom + lr * grad(atom) * fitness
- Crossover: топ-K атомов производят offspring через interpolate
- Selection: лучшие survive, худшие заменяются

**Отличие от HAR:**
- Использует GRADIENT для направления эволюции
- Может создавать НОВЫЕ атомы (не только переиспользовать)
- Эволюция идёт в пространстве параметров

**Интеграция:** `layer.enable_dae()` + `layer.step_dae()` (после backward)

**Файлы**: `layers.py` (DifferentiableAtomEvolver), `train_small_moe.py`

---

#### B4: Hebbian Atom Resonance (HAR) ⭐
**Источник**: Клод (оригинальная)  
**Статус**: ✅ ЗАВЕРШЕНО — 2026-04-23

**Суть**: Принцип Хебба: "Нейроны, которые активируются вместе — соединяются". Работает без градиентов.

**Как работает:**
1. После каждого forward: собирает статистику co-activation и success_score
2. Раз в N шагов:
   - Резонирующие пары (часто вместе + слишком похожи) → flip биты для ортогональности
   - Мёртвые атомы → заменить на мутантов лучших

**Почему для TRILIX:** атомы бинарные → операция "сделать ортогональным" = O(r) переворот битов. В FP16 потребовала бы оптимизации.

**Чем улучшит:**
- Решает мёртвые атомы без градиентов
- Выпрыгивает из локальных минимумов
- HAR + DAE = полная эволюционная система

**Файлы**: `layers.py` (HebbianAtomResonance), `train.py`

---

#### B5: Symbolic Diff Operations (SDO) ⭐
**Источник**: Клод (оригинальная)  
**Статус**: ✅ ЗАВЕРШЕНО (коммит 5421917) — 2026-04-23

**Суть**: Аналогии через XOR. `XOR(a,b) ≈ XOR(c,d)` → модель умеет делать структурные аналогии.

**Суть**: Аналогии через XOR. `XOR(a,b) ≈ XOR(c,d)` → модель умеет делать структурные аналогии.

**Как работает:**
```python
# Семплировать случайные четвёрки (a, b, c, d) из codebook
# XOR в {±1} = поэлементное умножение
diff_ab = a * b  # "разность" a и b
diff_cd = c * d  # "разность" c и d
# Если diff_ab близок к diff_cd → есть аналогия
```

**Почему делает TRILIX умнее:**
- Стандартные LLM учат аналогии косвенно
- TRILIX учит через явные символические операции (XOR)
- Это ближе к логическому мышлению человека
- Должно улучшить ARC-Challenge и WinoGrande

**Два компонента:**
1. `analogy_clarity`: поощряет высокий |similarity| (чёткие аналогии)
2. `analogy_diversity`: поощряет разнообразие (не коллапс к одной)

**Файлы**: `layers.py` (SymbolicDiffLoss), `model.py`, `train.py`

---

#### B1.5: Flat Hierarchical Codebook (FHC)
**Источник**: Клод (улучшение R-MoE Дипсика)  
**Статус**: ✅ ЗАВЕРШЕНО — 2026-04-23

**Суть**: 4 мета-эксперта × 4 базовых = 16 виртуальных специализаций. Один matmul, clear gradient flow.

**vs R-MoE Дипсика:** дерево = непрозрачный gradient flow. FHC = один forward, легко профилировать.

**Архитектура:**
- Meta-experts (4): "стратегии" маршрутизации
- Base experts (4): "базовые паттерны"
- Meta affinity (learned): [meta_k × base_k] — какой meta любит какой base
- Virtual expert = meta[b] ⊙ base[a] (element-wise, не matmul)

---

#### B2: Emergent Agent Swarm
**Источник**: Дипсик  
**Статус**: ✅ ЗАВЕРШЕНО — 2026-04-23

**Суть**: 1024+ агента работают как рой. Агенты НЕ знают друг о друге — специализация emergent.

**Как работает:**
- Все агенты живут в SoulCodebook (1024 вектора)
- Каждый forward: task_embedding = attention query между агентами
- "Лучшие" агенты для задачи получают больше внимания
- Специализация emerges: агент128 → Python, агент256 → математика

**Интеграция**: `agent_swarm = EmergentAgentSwarm(1024, rank, 4 heads)` → forward → specialization_loss

**Файлы**: `layers.py` (EmergentAgentSwarm), `model.py`

---

### Группа C — Продвинутые (не сейчас)

| # | Инновация | Почему не сейчас |
|---|-----------|-----------------|
| **EDH** | Error-Driven Hypernetwork | Слишком сложно, требует отдельного обучения |
| **REL** | Reflective Error Loop | Per-token loss сложно для synthetic training |
| **C1** | Recursive Self-Compilation Loops | Builder Expert — circular dependency |
| **C2** | Dynamic Ephemeral Experts | Inference-time генерация не дифференцируема |
| **C3** | Meta-Reflective Mutation | Конфликт двух градиентных сигналов |

---

## Мои инновации (мои идеи)

| # | Инновация | Приоритет | Статус |
|---|-----------|-----------|--------|
| **M1** | Temporal Atom Memory | 🟡 СРЕДНИЙ | Не реализовано |
| **M2** | Hierarchical Codebook Pruning | 🟡 СРЕДНИЙ | Не реализовано |
| **M3** | Cross-Layer Codebook Sharing | 🟡 СРЕДНИЙ | Не реализовано |
| **M4** | Adversarial Codebook Perturbation | 🟠 ПРОДВИНУТЫЙ | Не реализовано |
| **M5** | Curriculum Atom Learning | 🟡 СРЕДНИЙ | Не реализовано |

---

## Что НЕ реализуем

| Идея | Почему | Когда можно |
|------|--------|-------------|
| **UAS объединять** | Soul и WorldModel уже работают раздельно. Объединение сломает отладку. | Добавить Belief Gate в WorldModelHead |
| **Meta-Reflective Mutation** (Гемини) | Конфликт градиентов. | После REL от Клода |
| **R-MoE как дерево** (Дипсик) | Рекурсия = непрозрачный gradient | Заменить на FHC от Клода |

---

## План реализации (обновлённый)

```
✅ Завершено:
├── commitment_beta fix
├── SAIB + RVQ + SGH + ATC + LCC
├── A1: Soul Codebook
└── A2: Latent World Model

📋 Следующие (B4 → B3):
├── B4: Hebbian Atom Resonance (~2 дня)
│   └── files: layers.py, train.py
│
├── B3: Differentiable Atom Evolution (~3 дня)
│   └── files: evolution.py (новый), train.py
│
├── A3: Belief Gate в WorldModel (~0.5 дня)
│   └── files: layers.py, model.py
│
📋 Среднесрочные:
├── B1.5: Flat Hierarchical Codebook (~4 дня)
├── B2: Emergent Agent Swarm (~2 дня)
│
📋 Продвинутые (когда стабилизируется):
├── C1: EDH
├── C2: REL
└── M1-M5: мои инновации
```

---

## Архитектура TRILIX Neural OS (по Клоду)

```
                    TRILIX NEURAL OS
                    ════════════════
    
    ┌─────────────────────────────────────────────┐
    │           KERNEL (неизменный фундамент)      │
    │  Latent Factorization: W = U·V^T           │
    │  XOR Atoms: C[j] = sign(XOR(atoms))         │
    │  Scale Factors: h, g, l (BF16)               │
    └──────────────┬──────────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────────┐
    │         MEMORY SUBSYSTEM                     │
    │  FHC (Flat Hierarchical Codebook)           │ ← иерархическая память
    │  HAR (Hebbian Atom Resonance)               │ ← самоорганизация памяти
    │  DAE (Differentiable Atom Evolution)       │ ← эволюция памяти
    └──────────────┬──────────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────────┐
    │         AGENT SUBSYSTEM                      │
    │  UAS/Soul (Unified Agent State)            │ ← кто я
    │  WorldModel + Belief Gate                   │ ← что я знаю
    │  EDH (Error-Driven Hypernetwork)            │ ← создаю инструменты
    └──────────────┬──────────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────────┐
    │         REASONING SUBSYSTEM                  │
    │  SDO (Symbolic Diff Operations)             │ ← аналогии через XOR
    │  REL (Reflective Error Loop)                │ ← "я сомневаюсь здесь"
    └─────────────────────────────────────────────┘
```

Это **4-уровневая ОС**: Kernel → Memory → Agency → Reasoning.

---

## Метрики успеха

| Инновация | Метрика | Целевое значение |
|-----------|---------|------------------|
| Все | Loss | < 8.0 (сейчас ~592) |
| Все | BPW | < 0.0048 |
| Все | Скорость | 3-5 сек/шаг |
| Soul Codebook | Количество агентов | 1024+ |
| World Model | World Model Loss | < 0.1 |
| HAR | Dead atoms | < 1% |
| SDO | Analogy clarity | > 0.8 |
| DAE | Улучшение loss после evolution | > 5% |
| FHC | Виртуальных специализаций | ×16 |

---

## Версии файлов

| Файл | Назначение |
|------|------------|
| `trilix/layers.py` | TRILIXLinear + CodebookExpert + MoE + все инновации |
| `trilix/config.py` | TRILIXConfig — конфигурация |
| `trilix/model.py` | TRILIXTransformer + WorldModelHead + SoulCodebook |
| `train_small_moe.py` | Скрипт обучения |

---

## Git история

| Коммит | Что |
|--------|-----|
| `0fc0be4` | FIX: commitment_beta 0.25 → 0.0001 |
| `96b3e92` | FEAT: Properly integrate RVQ + SAIB + SGH |
| `e39abf7` | FEAT: Add all 5 innovations (initial stubs) |
| `ef44225` | FEAT: ATC — Adaptive Temperature Cascade |
| `bc4006b` | FEAT: A2 Latent World Model |
| `1bbb49f` | FEAT: A1 Soul Codebook — 1024 агента |
| `d7459fc` | FEAT: B4 Hebbian Atom Resonance |

---

*Документ обновляется после каждого коммита. Последнее обновление: 2026-04-23*