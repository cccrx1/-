# Claude.md

This file documents project context and conventions for AI coding agents.

## Project Purpose

CIFAR-10 backdoor attack and defense experiment suite with reproducible orchestration.

## Canonical Entry Points

- Preferred: `python run_suite.py ...`
- Compatibility: `python run.py ...` (forwards to `run_suite.py`)
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
- `test/`: standalone runnable scripts and matrix file (`_bootstrap.py` for shim startup)
- `experiments/`: generated logs, checkpoints, metrics
- `example_model/`: pretrained checkpoints
- root: dependency and batch entry scripts

## Maintenance Rules

- Keep orchestration logic in `core/pipeline`, not in `test/`.
- Keep `test/` scripts as thin wrappers/shims.
- Keep backward compatibility in `run.py` unless explicitly removed.
- Do not write experiment outputs outside `experiments/`.
- Periodically clean generated cache directories (`__pycache__/`, `.cache/`) from the project tree.

## Validation Checklist

- `python run_suite.py --help`
- `python run_suite.py smoke --dry-run`
- `python test/run_case.py --case badnets_refine --dry-run`
- `python test/run_test_suite.py --list-cases`
- `Get-ChildItem -Path . -Recurse -Force -Directory | Where-Object { $_.Name -in @('__pycache__','.cache','cache') -or $_.FullName -match '\\cache(\\|$)' }`
