"""Backward-compatible wrapper for the relocated suite runner.

Primary suite implementation now lives in test/run_test_suite.py.
"""

import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    target = root / "test" / "run_test_suite.py"
    cmd = [sys.executable, str(target), *sys.argv[1:]]
    result = subprocess.run(cmd, cwd=str(root), check=False)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
