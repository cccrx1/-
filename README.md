# Backdoor Defense Experiment Suite

This project is organized around six clear responsibilities:

1. `core/` stores all core implementations (attacks, defenses, models, pipeline).
2. `experiments/` stores run outputs, logs, and summaries.
3. `test/` stores standalone runnable scripts.
4. `example_model/` stores pretrained checkpoints.
5. Project root keeps `requirements.txt` and batch entry scripts.
6. Parameter control supports both per-script overrides and unified suite overrides.

## Directory Layout

- `core/`
  - `attacks/`, `defenses/`, `models/`, `utils/`
  - `pipeline/` (suite orchestration and matrix runners)
- `test/`
  - `run_case.py`, `run_test_suite.py`
  - case wrappers such as `run_badnets_refine.py`
  - `_bootstrap.py` (shared standalone script bootstrap)
  - `test_matrix.json`
- `experiments/`
  - generated outputs and summaries
- `example_model/`
  - pretrained model files
- `run_suite.py`
  - main entrypoint
- `run.py`
  - compatibility forwarder to `run_suite.py`

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
python test/run_badnets_refine.py --seed 777 --dry-run
python test/run_test_suite.py --list-cases
```

## Parameter Priority

Runtime parameter priority is:

`CLI > case config in test_matrix.json > defaults in test_matrix.json > code defaults`

This applies to both single script execution and batch suite execution.

## Notes

- Use `run_suite.py` as the primary entrypoint.
- `run.py` is retained for backward compatibility.
- `test/` wrappers are thin shims; orchestration stays in `core/pipeline`.
- Outputs are written under `experiments/`.

## Cache Cleanup

This repo may generate temporary cache directories such as `__pycache__/` and `.cache/` during runs.

PowerShell cleanup command:

```powershell
Get-ChildItem -Path . -Recurse -Force -Directory |
  Where-Object { $_.Name -in @('__pycache__','.cache','cache') -or $_.FullName -match '\\cache(\\|$)' } |
  ForEach-Object { Remove-Item -LiteralPath $_.FullName -Recurse -Force }
```
