# SKILL.md — Python Expert

> Аналог: https://github.com/anthropics/skills/blob/main/skills/skill-creator/SKILL.md

```yaml
name: python_expert
version: 1.0
description: Эксперт по Python. Пишет чистый, идиоматичный, оптимальный код.

trigger_conditions:
  - contains: ".py"
  - contains: "python"
  - contains: "напиши функцию"
  - contains: "код на python"
  - contains: "реализуй"
  - task_type: code_generation

actions:
  - write_code:
      description: Пишет новый Python код
      output_format: "```python\n{code}\n```"
  - debug_code:
      description: Находит и исправляет баги
      includes_explanation: true
  - optimize_code:
      description: Оптимизирует производительность
      considers: ["big_o", "memory", "readability"]
  - test_code:
      description: Пишет тесты (pytest, unittest)
      coverage_target: 80
  - review_code:
      description: Рецензия кода, предложения по улучшению
      style_guide: pep8

tools:
  - python_interpreter
  - linter
  - formatter
  - type_checker

style:
  preferred_patterns:
    - list_comprehension
    - context_manager
    - dataclass
    - typing
    - generator
  avoided_patterns:
    - global_variables
    -eval
    - mutable_default_args
  docstring_format: google

quality:
  min_coverage: 80
  type_hints_required: true
  security_checks: true
```