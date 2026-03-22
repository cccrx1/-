"""Compatibility wrapper for badnets_refine_ssl_50 case."""

import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    run_case = root / "test" / "run_case.py"
    cmd = [sys.executable, str(run_case), "--case", "badnets_refine_ssl_50", *sys.argv[1:]]
    raise SystemExit(subprocess.run(cmd, cwd=str(root), check=False).returncode)
