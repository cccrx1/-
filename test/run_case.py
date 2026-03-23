"""Run one matrix-defined case using the unified suite entry."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run one case from test_matrix.json.")
    parser.add_argument("--case", type=str, required=True, help="Case name defined in matrix file.")
    parser.add_argument("--matrix", type=str, default="test/test_matrix.json", help="Matrix json path.")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable.")
    parser.add_argument("--dry-run", action="store_true", help="Print command only.")
    args, passthrough = parser.parse_known_args()
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    return args, passthrough


def load_case(matrix_path: Path, case_name: str):
    with matrix_path.open("r", encoding="utf-8") as f:
        matrix = json.load(f)
    cases = matrix.get("cases", {})
    if case_name not in cases:
        raise KeyError(f"Case '{case_name}' not found in {matrix_path}.")
    defaults = matrix.get("defaults", {})
    if not isinstance(defaults, dict):
        defaults = {}

    case_cfg = dict(defaults)
    case_cfg.update(cases[case_name])
    return case_cfg


def to_cli_args(case_cfg):
    args = []
    mapping = {
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
        "attack_cache_root": "--attack-cache-root",
        "output_root": "--output-root",
        "adv_dataset_root": "--adv-dataset-root",
        "dataset_root": "--dataset-root",
    }
    for key, flag in mapping.items():
        if key in case_cfg:
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


def main():
    args, passthrough = parse_args()
    root = Path(__file__).resolve().parents[1]
    matrix_path = (root / args.matrix).resolve()
    case_cfg = load_case(matrix_path, args.case)

    suite_script = root / "run_cifar10_attack_refine_suite.py"
    cmd = [args.python, str(suite_script), *to_cli_args(case_cfg), *passthrough]

    print("Case:", args.case)
    print("Matrix:", str(matrix_path))
    print("Command:", " ".join(cmd))

    if args.dry_run:
        return

    result = subprocess.run(cmd, cwd=str(root), check=False)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
