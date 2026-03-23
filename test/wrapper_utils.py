"""Shared compatibility wrapper launcher for named matrix cases."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence


def run_named_case(case_name: str, passthrough: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    run_case = root / "test" / "run_case.py"
    cmd = [sys.executable, str(run_case), "--case", case_name]
    if passthrough:
        cmd.extend(list(passthrough))
    return subprocess.run(cmd, cwd=str(root), check=False).returncode
