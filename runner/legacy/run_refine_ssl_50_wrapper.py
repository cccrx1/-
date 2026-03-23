"""Backward-compatible shortcut for REFINE_SSL (50 epochs).

Preferred command is now:
python run.py single --defense-variant refine_ssl --refine-epochs 50
"""

import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    cmd = [
        sys.executable,
        str(root / "run.py"),
        "single",
        "--defense-variant",
        "refine_ssl",
        "--refine-epochs",
        "50",
        *sys.argv[1:],
    ]
    result = subprocess.run(cmd, cwd=str(root), check=False)
    raise SystemExit(result.returncode)
