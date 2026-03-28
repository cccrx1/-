# Claude.md

This file documents project context and conventions for AI coding agents.

## Project Purpose

CIFAR-10 backdoor attack and defense experiment suite with reproducible orchestration.

## Canonical Entry Points

- Preferred: `python run_suite.py ...`
- Direct module entry:
  - `python -m core.pipeline.suite_pipeline`
  - `python -m core.pipeline.run_case`
  - `python -m core.pipeline.run_test_suite`

## Parameter Model

Priority order:

1. CLI explicit arguments
2. `cases[case_name]` in `test/test_matrix.json`
3. `defaults` in `test/test_matrix.json`
4. hardcoded defaults in code

## Folder Responsibilities

- `core/`: core implementation and orchestration (`core/pipeline`)
- `test/`: matrix configuration (`test_matrix.json`)
- `experiments/`: generated logs, checkpoints, metrics
- `example_model/`: pretrained checkpoints
- root: dependency and batch entry scripts

## Output Grouping

- `single` mode: outputs under `experiments/single/...`
- `smoke` mode: outputs under `experiments/smoke/...`
- `case` mode: outputs under `experiments/case/<case_name>/...`
- `suite` mode: per-case outputs under `experiments/suite/<case_name>/...`, summaries under `experiments/suite/summary/...`

## Maintenance Rules

- Keep orchestration logic in `core/pipeline`, not in `test/`.
- Keep `test/` as configuration-only (matrix definitions).
- Do not write experiment outputs outside `experiments/`.
- Periodically clean generated cache directories (`__pycache__/`, `.cache/`) from the project tree.

## Validation Checklist

- `python run_suite.py --help`
- `python run_suite.py smoke --dry-run`
- `python -m core.pipeline.run_case --case badnets_refine --run-group case --dry-run`
- `python -m core.pipeline.run_test_suite --list-cases`
- `Get-ChildItem -Path . -Recurse -Force -Directory | Where-Object { $_.Name -in @('__pycache__','.cache','cache') -or $_.FullName -match '\\cache(\\|$)' }`
