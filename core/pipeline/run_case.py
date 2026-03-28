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

    cmd = [args.python, "-m", "core.pipeline.suite_pipeline", *case_cfg_to_cli_args(case_cfg), *passthrough]

    print("Case:", args.case)
    print("Matrix:", str(matrix_path))
    print("Command:", " ".join(cmd))

    if args.dry_run:
        return

    result = subprocess.run(cmd, cwd=str(root), check=False)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
