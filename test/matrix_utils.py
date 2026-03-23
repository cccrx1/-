"""Shared helpers for test matrix loading, case resolving, and CLI mapping."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


CASE_TO_CLI_FLAG = {
    "only_attack": "--only-attack",
    "defense_variant": "--defense-variant",
    "seed": "--seed",
    "deterministic": "--deterministic",
    "device_mode": "--device-mode",
    "cuda_selected_devices": "--cuda-selected-devices",
    "batch_size": "--batch-size",
    "num_workers": "--num-workers",
    "benign_epochs": "--benign-epochs",
    "attack_epochs": "--attack-epochs",
    "lc_epochs": "--lc-epochs",
    "refine_epochs": "--refine-epochs",
    "badnets_rate": "--badnets-rate",
    "blended_rate": "--blended-rate",
    "lc_rate": "--lc-rate",
    "lc_eps": "--lc-eps",
    "lc_alpha": "--lc-alpha",
    "lc_steps": "--lc-steps",
    "refine_first_channels": "--refine-first-channels",
    "cg_threshold": "--cg-threshold",
    "cg_temperature": "--cg-temperature",
    "cg_strength": "--cg-strength",
    "ssl_temperature": "--ssl-temperature",
    "ssl_weight": "--ssl-weight",
    "skip_lc": "--skip-lc",
    "force_rebuild": "--force-rebuild",
    "attack_cache_root": "--attack-cache-root",
    "output_root": "--output-root",
    "adv_dataset_root": "--adv-dataset-root",
    "dataset_root": "--dataset-root",
}


def load_matrix(matrix_path: Path) -> Dict:
    with matrix_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid matrix format: {matrix_path}")
    return data


def get_cases(matrix_data: Dict) -> Dict:
    cases = matrix_data.get("cases", {})
    if not isinstance(cases, dict) or not cases:
        raise ValueError("Matrix 'cases' must be a non-empty object")
    return cases


def resolve_cases(cases_text: str, known_cases: Iterable[str]) -> List[str]:
    known = list(known_cases)
    if cases_text == "all":
        return known

    names = [item.strip() for item in cases_text.split(",") if item.strip()]
    invalid = [name for name in names if name not in known]
    if invalid:
        raise ValueError(f"Unknown cases: {invalid}. Supported: {known}")
    return names


def load_case_config(matrix_path: Path, case_name: str) -> Dict:
    matrix = load_matrix(matrix_path)
    cases = get_cases(matrix)
    if case_name not in cases:
        raise KeyError(f"Case '{case_name}' not found in {matrix_path}.")

    defaults = matrix.get("defaults", {})
    if not isinstance(defaults, dict):
        defaults = {}

    cfg = dict(defaults)
    cfg.update(cases[case_name])
    return cfg


def case_cfg_to_cli_args(case_cfg: Dict) -> List[str]:
    args: List[str] = []

    for key, flag in CASE_TO_CLI_FLAG.items():
        if key not in case_cfg:
            continue

        val = case_cfg[key]
        if isinstance(val, bool):
            if val:
                args.append(flag)
            continue

        args.extend([flag, str(val)])

    extra = case_cfg.get("extra_args", [])
    if isinstance(extra, list):
        args.extend([str(item) for item in extra])

    return args


def stage_keys_from_attack(only_attack: str) -> Tuple[str | None, str | None]:
    mapping = {
        "badnets": ("badnets", "refine_badnets"),
        "blended": ("blended", "refine_blended"),
        "label_consistent": ("label_consistent", "refine_label_consistent"),
    }
    return mapping.get(only_attack, (None, None))


def safe_get_top1(stage_dict, metric_key: str):
    if not isinstance(stage_dict, dict):
        return None
    metric = stage_dict.get(metric_key)
    if not isinstance(metric, dict):
        return None
    val = metric.get("top1_acc")
    return float(val) if isinstance(val, (int, float)) else None


def collect_case_metrics(root: Path, case_cfg: Dict) -> Dict:
    output_root = case_cfg.get("output_root")
    if not isinstance(output_root, str):
        return {"error": "missing output_root in matrix"}

    summary_path = (root / output_root / "metrics_summary.json").resolve()
    if not summary_path.exists():
        return {"error": f"metrics not found: {summary_path}"}

    with summary_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    stages = data.get("stages", {}) if isinstance(data, dict) else {}
    attack_key, refine_key = stage_keys_from_attack(str(case_cfg.get("only_attack", "")))

    attack_stage = stages.get(attack_key, {}) if attack_key else {}
    refine_stage = stages.get(refine_key, {}) if refine_key else {}

    return {
        "summary_path": str(summary_path),
        "clean_before": safe_get_top1(attack_stage, "clean"),
        "poisoned_before": safe_get_top1(attack_stage, "poisoned"),
        "clean_after": safe_get_top1(refine_stage, "clean_after_refine"),
        "poisoned_after": safe_get_top1(refine_stage, "poisoned_after_refine"),
    }
