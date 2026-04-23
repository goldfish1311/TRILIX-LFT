# SKILL.md — Meta-Skill Creator

> Аналог: https://github.com/anthropics/skills/blob/main/skills/skill-creator/SKILL.md

```yaml
meta_skill_creator:
  name: SKILL.md Creator
  version: 1.0
  description: Создаёт новые SKILL.md файлы для агентов. Анализирует задачу и генерирует специализированные навыки.

  trigger_conditions:
    - "нужен новый скилл"
    - "нет подходящего скилла"
    - "создать специализированный навык"
    - "такой скилл не существует"
    - "сгенерируй скилл для"

  process:
    - step_1_analysis:
        description: Анализ задачи — какие навыки нужны
        output: requirements_for_skill
    - step_2_generation:
        description: Генерация SKILL.md по шаблону
        template: skills/skill_template/SKILL.md
    - step_3_validation:
        description: Валидация синтаксиса и структуры
        checks:
          - yaml_valid
          - required_fields
          - actions_defined
          - trigger_conditions_defined
    - step_4_creation:
        description: Создание файла skills/{name}/SKILL.md
    - step_5_addition:
        description: Добавление в стек агента (agent.skills.append(new_skill))
    - step_6_execution:
        description: Исполнение задачи с новым скиллом

  output:
    path: "skills/{generated_name}/SKILL.md"
    format: yaml

  quality:
    min_defined_actions: 1
    min_defined_triggers: 1
    requires_description: true
    requires_version: true
```

## Пример: создание скилла "rust_expert"

**Вход**: "Нужен скилл для работы с Rust"

**Анализ (шаг 1)**:
```
required_skills = ["rust_syntax", "cargo", "lifetimes", "error_handling"]
style = "functional + ownership"
patterns = ["match", "Result", "Option", "iterators"]
```

**Генерация (шаг 2)**:
```yaml
name: rust_expert
version: 1.0
description: Эксперт по Rust. Пишет безопасный, эффективный код с учётом ownership.
trigger_conditions:
  - contains: "rust"
  - contains: ".rs"
  - task_type: rust_generation
actions:
  - write_code
  - handle_lifetimes
  - error_handling
tools:
  - rustc
  - cargo
  - clippy
```

**Создание (шаг 4)**: `skills/rust_expert/SKILL.md`

**Добавление (шаг 5)**: `agent.skills.append("rust_expert")`