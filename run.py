"""Backward-compatible entrypoint that forwards to run_suite.py."""

import subprocess
import sys
from pathlib import Path
def main() -> None:
    root = Path(__file__).resolve().parent
    cmd = [sys.executable, str(root / "run_suite.py"), *sys.argv[1:]]
    print("Forward Command:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(root), check=False)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
