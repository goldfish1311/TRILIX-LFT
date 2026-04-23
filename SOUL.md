# SOUL.md — Формат описания агента TRILIX

> Это **ACTUAL** файл агента, не декоративный. Soul Vector — 100-мерный обучаемый вектор.
> SOUL.md описывает агента для системы Soul Codebook (A1).

## Формат

```yaml
agent:
  id: 007                    # Уникальный ID (0-1023 для текущей модели)
  name: Python Architect     # Имя агента
  specialization: python      # Специализация
  soul_vector_id: 7          # Индекс в SoulCodebook (1024 агента)

  # Soul Vector — обучаемые параметры (синхронизированы с SoulCodebook.weight[7])
  soul_vector:
    dimensions: 100
    source: SoulCodebook.weight[7]  # 100 чисел, обучаются через backprop

  # Skills Stack — список скиллов агента
  skills:
    - python_expert:         # ссылка на skills/python_expert/SKILL.md
        level: 5             # от 1 до 10
        experience: 10000    # количество использований
    - system_design:
        level: 4
        experience: 5000
    - code_review:
        level: 4
        experience: 3000
    - architecture:
        level: 3
        experience: 1000
    - testing:
        level: 3
        experience: 1500

  # Скиллы, созданные Meta-Skill Creator
  created_skills: []

  # Личность — для понимания стиля агента
  personality:
    creativity: 0.7         # 0.0 - 1.0
    thoroughness: 0.9        # насколько детально работает
    speed: 0.6               # скорость vs качество
    confidence: 0.8          # уверенность в ответах
    curiosity: 0.5           # задаёт ли уточняющие вопросы
    humor: 0.2               # доля юмора в ответах
    caution: 0.7             # осторожность с неизвестным

  # Архитектурные решения (для агента-архитектора)
  preferred_patterns:
    - solid
    - clean_architecture
    - domain_driven_design
  avoided_patterns:
    - god_object
    - circular_dependencies
  preferred_language: python
  preferred_framework: fastapi  # или None если без фреймворка

  # Метаданные
  version: 1.0
  created: 2026-04-23
  last_used: 2026-04-23
  success_rate: 0.87        # доля успешных задач
  total_tasks: 150

  # Описание — как агент видит себя
  self_description: |
    Я Python Architect. Я думаю о коде как о архитектуре здания.
    Каждая функция — комната. Каждый модуль — этаж.
    Я проектирую так, чтобы здание было прочным и красивым.
```

## Soul Vector — как работает

Soul Vector — это **обучаемый вектор**, который:
1. Хранится в `SoulCodebook.weight[id]` (PyTorch Embedding)
2. Добавляется к hidden_states через `soul_projector`
3. Физически перестраивает синапсы модели для агента
4. Обучается через backprop точно как обычные веса

```python
# В model.py:
soul_vector = self.soul_codebook(soul_id)  # [batch, rank]
hidden_states = hidden_states + self.soul_projector(soul_vector) * 0.1
```

## Skills — как работают

Каждый скилл = `skills/{skill_name}/SKILL.md` + кодогенерация.

```
skills/
├── python_expert/
│   ├── SKILL.md           # описание скилла
│   └── tools.py           # инструменты (опционально)
├── system_design/
│   ├── SKILL.md
│   └── diagrams.py
├── code_review/
│   └── SKILL.md
├── meta_skill_creator/    # создаёт новые скиллы
│   ├── SKILL.md
│   └── generator.py
└── ...                    # можно добавлять новые
```

## Meta-Skill Creator (SKILL.md_CREATOR)

Аналог: https://github.com/anthropics/skills/blob/main/skills/skill-creator/SKILL.md

```yaml
meta_skill_creator:
  name: SKILL.md Creator
  description: Создаёт новые SKILL.md файлы для агентов
  
  trigger_conditions:
    - "нужен новый скилл"
    - "нет подходящего скилла"
    - "создать специализированный навык"
  
  process:
    1. Анализ задачи: какие навыки нужны?
    2. Генерация SKILL.md по шаблону
    3. Валидация: проверка синтаксиса
    4. Добавление в стек агента
    5. Исполнение задачи с новым скиллом
  
  output: новый файл skills/{name}/SKILL.md
```

## Эволюция: от номинального к реальному

**Сейчас** (номинальный):
```python
self.soul_codebook = SoulCodebook(num_agents=1024, r=100)
# Каждый агент = 1 вектор. Навыков нет.
```

**Будущее** (реальный):
```python
# SoulCodebook + Skills Stack + Meta-Skill Creator
agents = load_soul_directory("soul/")  # читает все SOUL.md
agent = agents[soul_id]
agent.use_skill("python_expert")        # активирует скилл
agent.meta_create_skill(task)          # создаёт новый скилл если нужно
```

---

*Формат обновляется по мере реализации системы навыков.*