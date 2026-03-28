"""Shared compatibility wrapper launcher for named matrix cases."""

from __future__ import annotations

import subprocess
import sys
from typing import Sequence

try:
    from test._bootstrap import ensure_project_root_on_path
except ModuleNotFoundError:
    from _bootstrap import ensure_project_root_on_path


def run_named_case(case_name: str, passthrough: Sequence[str] | None = None) -> int:
    root = ensure_project_root_on_path()
    cmd = [sys.executable, "-m", "core.pipeline.run_case", "--case", case_name]
    if passthrough:
        cmd.extend(list(passthrough))
    return subprocess.run(cmd, cwd=str(root), check=False).returncode
