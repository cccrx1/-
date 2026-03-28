# Backdoor Defense Experiment Suite

This project is organized around six clear responsibilities:

1. `core/` stores all core implementations (attacks, defenses, models, pipeline).
2. `experiments/` stores run outputs, logs, and summaries.
3. `test/` stores matrix configuration.
4. `example_model/` stores pretrained checkpoints.
5. Project root keeps `requirements.txt` and batch entry scripts.
6. Parameter control supports both per-script overrides and unified suite overrides.

## Directory Layout

- `core/`
  - `attacks/`, `defenses/`, `models/`, `utils/`
  - `pipeline/` (suite orchestration and matrix runners)
- `test/`
  - `test_matrix.json`
- `experiments/`
  - generated outputs and summaries
- `example_model/`
  - pretrained model files
- `run_suite.py`
  - main entrypoint

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Single run:

```bash
python run_suite.py single --only-attack badnets --defense-variant refine
```

Smoke run:

```bash
python run_suite.py smoke --dry-run
```

Run one matrix case:

```bash
python run_suite.py case --case badnets_refine --dry-run
```

Run suite with unified overrides:

```bash
python run_suite.py suite --cases badnets_refine,blended_refine --seed 888 --batch-size 64 --dry-run
```

Standalone scripts in `test/`:

```bash
python -m core.pipeline.run_case --case badnets_refine --run-group case --dry-run
python -m core.pipeline.run_test_suite --cases all --dry-run
```

## Parameter Priority

Runtime parameter priority is:

`CLI > case config in test_matrix.json > defaults in test_matrix.json > code defaults`

This applies to both single script execution and batch suite execution.

## Notes

- Use `run_suite.py` as the primary entrypoint.
- Orchestration stays in `core/pipeline`.
- Outputs are written under `experiments/`.

## Output Grouping

Outputs are grouped by run type under `experiments/`:

- `single`: `experiments/single/...`
- `smoke`: `experiments/smoke/...`
- `case`: `experiments/case/<case_name>/...`
- `suite`: `experiments/suite/<case_name>/...`

Batch summary files are written to:

- `experiments/suite/summary/suite_summary_latest.json`
- `experiments/suite/summary/suite_summary_latest.md`

Shared attack cache roots are also grouped by run type, e.g.:

- `experiments/case/shared_attack_cache`
- `experiments/suite/shared_attack_cache`

## Cache Cleanup

This repo may generate temporary cache directories such as `__pycache__/` and `.cache/` during runs.

PowerShell cleanup command:

```powershell
Get-ChildItem -Path . -Recurse -Force -Directory |
  Where-Object { $_.Name -in @('__pycache__','.cache','cache') -or $_.FullName -match '\\cache(\\|$)' } |
  ForEach-Object { Remove-Item -LiteralPath $_.FullName -Recurse -Force }
```
