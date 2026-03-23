"""Unified project entrypoint.

Design goals:
1) Keep one top-level command for all run modes.
2) Keep experiment orchestration scripts inside test/.
3) Centralize common and method-specific runtime arguments.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def _add_pipeline_args(parser: argparse.ArgumentParser, with_defaults: bool) -> None:
    default = (lambda x: x) if with_defaults else (lambda _x: None)

    parser.add_argument("--dataset-root", type=str, default=default("./datasets"))
    parser.add_argument("--output-root", type=str, default=default("./experiments/runs"))
    parser.add_argument("--adv-dataset-root", type=str, default=default("./experiments/adv_dataset"))
    parser.add_argument("--seed", type=int, default=default(666))
    parser.add_argument("--deterministic", action="store_true", default=default(True))
    parser.add_argument("--y-target", type=int, default=default(0))
    parser.add_argument("--device-mode", type=str, choices=["GPU", "CPU"], default=default("GPU"))
    parser.add_argument("--cuda-selected-devices", type=str, default=default("0"))
    parser.add_argument("--batch-size", type=int, default=default(128))
    parser.add_argument("--num-workers", type=int, default=default(8))
    parser.add_argument("--benign-epochs", type=int, default=default(200))
    parser.add_argument("--attack-epochs", type=int, default=default(200))
    parser.add_argument("--lc-epochs", type=int, default=default(200))
    parser.add_argument("--refine-epochs", type=int, default=default(150))
    parser.add_argument("--badnets-rate", type=float, default=default(0.05))
    parser.add_argument("--blended-rate", type=float, default=default(0.05))
    parser.add_argument("--lc-rate", type=float, default=default(0.25))
    parser.add_argument("--lc-eps", type=float, default=default(8.0))
    parser.add_argument("--lc-alpha", type=float, default=default(1.5))
    parser.add_argument("--lc-steps", type=int, default=default(100))
    parser.add_argument("--refine-first-channels", type=int, default=default(64))

    # Method-specific defense parameters.
    parser.add_argument("--defense-variant", type=str, choices=["refine", "refine_cg", "refine_ssl"], default=default("refine"))
    parser.add_argument("--cg-threshold", type=float, default=default(0.35))
    parser.add_argument("--cg-temperature", type=float, default=default(0.10))
    parser.add_argument("--cg-strength", type=float, default=default(1.0))
    parser.add_argument("--ssl-temperature", type=float, default=default(0.07))
    parser.add_argument("--ssl-weight", type=float, default=default(0.02))

    parser.add_argument("--only-attack", type=str, choices=["all", "badnets", "blended", "label_consistent"], default=default("all"))
    parser.add_argument("--attack-cache-root", type=str, default=default(""))
    parser.add_argument("--skip-lc", action="store_true", default=default(False))
    parser.add_argument("--force-rebuild", action="store_true", default=default(False))


def _append_arg(cmd: List[str], flag: str, value: Optional[object]) -> None:
    if value is None:
        return
    cmd.extend([flag, str(value)])


def _append_bool(cmd: List[str], flag: str, enabled: Optional[bool]) -> None:
    if enabled:
        cmd.append(flag)


def _pipeline_args_to_cmd(args: argparse.Namespace, include_defaults: bool) -> List[str]:
    cmd: List[str] = []

    keys = [
        ("--dataset-root", "dataset_root"),
        ("--output-root", "output_root"),
        ("--adv-dataset-root", "adv_dataset_root"),
        ("--seed", "seed"),
        ("--y-target", "y_target"),
        ("--device-mode", "device_mode"),
        ("--cuda-selected-devices", "cuda_selected_devices"),
        ("--batch-size", "batch_size"),
        ("--num-workers", "num_workers"),
        ("--benign-epochs", "benign_epochs"),
        ("--attack-epochs", "attack_epochs"),
        ("--lc-epochs", "lc_epochs"),
        ("--refine-epochs", "refine_epochs"),
        ("--badnets-rate", "badnets_rate"),
        ("--blended-rate", "blended_rate"),
        ("--lc-rate", "lc_rate"),
        ("--lc-eps", "lc_eps"),
        ("--lc-alpha", "lc_alpha"),
        ("--lc-steps", "lc_steps"),
        ("--refine-first-channels", "refine_first_channels"),
        ("--defense-variant", "defense_variant"),
        ("--cg-threshold", "cg_threshold"),
        ("--cg-temperature", "cg_temperature"),
        ("--cg-strength", "cg_strength"),
        ("--ssl-temperature", "ssl_temperature"),
        ("--ssl-weight", "ssl_weight"),
        ("--only-attack", "only_attack"),
        ("--attack-cache-root", "attack_cache_root"),
    ]

    for flag, attr in keys:
        val = getattr(args, attr, None)
        if include_defaults or val is not None:
            _append_arg(cmd, flag, val)

    if include_defaults or getattr(args, "deterministic", False):
        _append_bool(cmd, "--deterministic", getattr(args, "deterministic", False))
    if include_defaults or getattr(args, "skip_lc", False):
        _append_bool(cmd, "--skip-lc", getattr(args, "skip_lc", False))
    if include_defaults or getattr(args, "force_rebuild", False):
        _append_bool(cmd, "--force-rebuild", getattr(args, "force_rebuild", False))

    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified entry for single run, matrix case, and suite execution.")
    sub = parser.add_subparsers(dest="mode", required=True)

    single = sub.add_parser("single", help="Run one direct pipeline job with explicit parameters.")
    _add_pipeline_args(single, with_defaults=True)

    case = sub.add_parser("case", help="Run one matrix-defined case from test/test_matrix.json.")
    case.add_argument("--case", type=str, required=True)
    case.add_argument("--matrix", type=str, default="test/test_matrix.json")
    case.add_argument("--dry-run", action="store_true", default=False)
    _add_pipeline_args(case, with_defaults=False)

    suite = sub.add_parser("suite", help="Run one or more matrix-defined cases.")
    suite.add_argument("--cases", type=str, default="all")
    suite.add_argument("--matrix", type=str, default="test/test_matrix.json")
    suite.add_argument("--continue-on-error", action="store_true", default=False)
    suite.add_argument("--list-cases", action="store_true", default=False)
    suite.add_argument("--dry-run", action="store_true", default=False)
    suite.add_argument("--summary-dir", type=str, default="./experiments/test/summary")
    _add_pipeline_args(suite, with_defaults=False)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent

    if args.mode == "single":
        target = root / "runner" / "suite_pipeline.py"
        cmd = [sys.executable, str(target), *_pipeline_args_to_cmd(args, include_defaults=True)]
    elif args.mode == "case":
        target = root / "test" / "run_case.py"
        cmd = [
            sys.executable,
            str(target),
            "--case",
            args.case,
            "--matrix",
            args.matrix,
        ]
        if args.dry_run:
            cmd.append("--dry-run")
        cmd.extend(_pipeline_args_to_cmd(args, include_defaults=False))
    else:
        target = root / "test" / "run_test_suite.py"
        cmd = [
            sys.executable,
            str(target),
            "--cases",
            args.cases,
            "--matrix",
            args.matrix,
            "--summary-dir",
            args.summary_dir,
        ]
        if args.continue_on_error:
            cmd.append("--continue-on-error")
        if args.list_cases:
            cmd.append("--list-cases")
        if args.dry_run:
            cmd.append("--dry-run")
        cmd.extend(_pipeline_args_to_cmd(args, include_defaults=False))

    print("Command:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(root), check=False)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
