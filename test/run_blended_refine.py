"""Compatibility wrapper for blended_refine case."""

import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    run_case = root / "test" / "run_case.py"
    cmd = [sys.executable, str(run_case), "--case", "blended_refine", *sys.argv[1:]]
    raise SystemExit(subprocess.run(cmd, cwd=str(root), check=False).returncode)
