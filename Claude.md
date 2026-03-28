# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

CIFAR-10 backdoor attack and defense experiment suite. Implements three attack methods (BadNets, Blended, LabelConsistent) and multiple REFINE defense variants, with reproducible orchestration and matrix-driven batch execution.

## Commands

```bash
# Install
pip install -r requirements.txt

# Dry-run smoke test (quick sanity check)
python run_suite.py smoke --dry-run

# Run a single experiment directly
python run_suite.py single --only-attack badnets --defense-variant refine

# Run one matrix-defined case
python run_suite.py case --case badnets_refine --dry-run

# Run full suite (all or selected cases)
python run_suite.py suite --cases all --dry-run
python run_suite.py suite --cases badnets_refine,blended_refine --seed 888 --batch-size 64

# List available matrix cases
python -m core.pipeline.run_test_suite --list-cases

# Direct module entry (bypasses run_suite.py dispatcher)
python -m core.pipeline.suite_pipeline --only-attack badnets --defense-variant refine
python -m core.pipeline.run_case --case badnets_refine --run-group case --dry-run
```

## Parameter Priority

`CLI args > cases[name] in test/test_matrix.json > defaults in test/test_matrix.json > code defaults in suite_config.py`

## Architecture

### Execution flow

`run_suite.py` is a thin CLI dispatcher. It parses the mode (`single`/`smoke`/`case`/`suite`) and shells out to the appropriate module:

- `single`/`smoke` → `core.pipeline.suite_pipeline` (runs one full pipeline directly)
- `case` → `core.pipeline.run_case` (resolves one case from matrix, then calls suite_pipeline)
- `suite` → `core.pipeline.run_test_suite` (iterates cases, calls run_case for each, aggregates summary)

Each level communicates via subprocess, forwarding CLI args down the chain.

### Pipeline stages (suite_pipeline.py)

A single pipeline run executes these stages sequentially:
1. Train benign CIFAR-10 model (ResNet-18)
2. Train attack model(s) — BadNets / Blended / LabelConsistent (controlled by `--only-attack`)
3. Run REFINE defense on each attacked model and evaluate

Stages are resumable: `StageStatusManager` tracks completion in `stage_status.json`. A `PipelineRunLock` prevents concurrent runs on the same output directory.

### Key modules

- `core/pipeline/suite_config.py` — `RuntimeConfig` dataclass holding all hyperparameters; `parse_suite_args()` builds it from CLI
- `core/pipeline/matrix_utils.py` — loads `test/test_matrix.json`, merges defaults+case config, converts to CLI args
- `core/pipeline/pipeline_state.py` — `StageStatusManager` (resumable stages), `PipelineRunLock`, `StageLogger` (tee to pipeline.log)
- `core/attacks/base.py` — `Base` class for attack training/testing loop
- `core/defenses/base.py` — `Base` class for defense with seed/determinism setup
- `core/defenses/REFINE*.py` — defense variants: REFINE (vanilla), REFINE_CG (confidence gating), REFINE_SSL (self-supervised), REFINE_PDB (proactive defense), REFINE_ADAPTIVE

### Defense variants

Selected via `--defense-variant`: `refine`, `refine_cg`, `refine_ssl`, `refine_pdb`, `refine_pdb_ssl`, `refine_adaptive`. Each variant adds auxiliary loss terms or gating mechanisms on top of the base REFINE purification.

### Matrix configuration

`test/test_matrix.json` defines `defaults` and named `cases`. Each case specifies `only_attack`, `defense_variant`, and optional overrides. The matrix runner resolves `defaults ← case overrides ← CLI overrides`.

## Folder Responsibilities

- `core/` — all core implementation (attacks, defenses, models, utils, pipeline orchestration)
- `test/` — configuration only (`test_matrix.json`); no code logic
- `experiments/` — all generated outputs (logs, checkpoints, metrics, summaries)
- `example_model/` — pretrained checkpoint files

## Output Grouping

Outputs are written under `experiments/<mode>/`:
- `single` → `experiments/single/...`
- `smoke` → `experiments/smoke/...`
- `case` → `experiments/case/<case_name>/...`
- `suite` → `experiments/suite/<case_name>/...` with summaries in `experiments/suite/summary/`

Attack caches are shared per run-group: `experiments/<mode>/shared_attack_cache/`

## Maintenance Rules

- Keep orchestration logic in `core/pipeline/`, not in `test/`.
- Keep `test/` as configuration-only.
- Do not write experiment outputs outside `experiments/`.
- Python 3.10–3.13 supported; torch version is pinned per Python version range in requirements.txt.
