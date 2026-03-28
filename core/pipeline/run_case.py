"""Run one matrix-defined case using the unified suite entry."""

import argparse
import subprocess
import sys
from pathlib import Path

from core.pipeline.matrix_utils import case_cfg_to_cli_args, load_case_config


def parse_args():
    parser = argparse.ArgumentParser(description="Run one case from test_matrix.json.")
    parser.add_argument("--case", type=str, required=True, help="Case name defined in matrix file.")
    parser.add_argument("--matrix", type=str, default="test/test_matrix.json", help="Matrix json path.")
    parser.add_argument("--run-group", type=str, choices=["case", "suite"], default="case", help="Output grouping under experiments/.")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable.")
    parser.add_argument("--dry-run", action="store_true", help="Print command only.")
    args, passthrough = parser.parse_known_args()
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    return args, passthrough


def main():
    args, passthrough = parse_args()
    root = Path(__file__).resolve().parents[2]
    matrix_path = (root / args.matrix).resolve()
    case_cfg = load_case_config(matrix_path, args.case)

    grouped_root = Path("./experiments") / args.run_group / args.case
    output_root = f"./{(grouped_root / 'runs').as_posix()}"
    adv_dataset_root = f"./{(grouped_root / 'adv_dataset').as_posix()}"
    attack_cache_root = f"./{(Path('experiments') / args.run_group / 'shared_attack_cache').as_posix()}"

    # Enforce grouped output roots without duplicating output flags from matrix config.
    for key in ("output_root", "adv_dataset_root", "attack_cache_root"):
        case_cfg.pop(key, None)

    cmd = [
        args.python,
        "-m",
        "core.pipeline.suite_pipeline",
        *case_cfg_to_cli_args(case_cfg),
        *passthrough,
        "--output-root",
        output_root,
        "--adv-dataset-root",
        adv_dataset_root,
        "--attack-cache-root",
        attack_cache_root,
    ]

    print("Case:", args.case)
    print("Matrix:", str(matrix_path))
    print("Command:", " ".join(cmd))

    if args.dry_run:
        return

    result = subprocess.run(cmd, cwd=str(root), check=False)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
