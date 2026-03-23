"""Pipeline runtime state helpers: lock, stage status, and log teeing."""

from __future__ import annotations

import atexit
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional


class StageStatusManager:
    """Track and manage pipeline stage completion for resumable runs."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.status_file = self.output_dir / "stage_status.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if self.status_file.exists():
            try:
                with self.status_file.open("r", encoding="utf-8") as f:
                    self.status = json.load(f)
            except (OSError, json.JSONDecodeError):
                self.status = {}
        else:
            self.status = {}

    def _save(self) -> None:
        with self.status_file.open("w", encoding="utf-8") as f:
            json.dump(self.status, f, indent=2)

    def is_completed(self, stage_name: str, force_rebuild: bool = False) -> bool:
        if force_rebuild:
            return False
        return self.status.get(stage_name) == "completed"

    def mark_completed(self, stage_name: str, metadata: Optional[dict] = None) -> None:
        self.status[stage_name] = "completed"
        if metadata:
            self.status[f"{stage_name}_meta"] = metadata
        self._save()

    def mark_failed(self, stage_name: str, error_msg: str = "") -> None:
        self.status[stage_name] = "failed"
        if error_msg:
            self.status[f"{stage_name}_error"] = error_msg
        self._save()

    def get_status(self, stage_name: str) -> str:
        return self.status.get(stage_name, "not-started")

    def reset(self) -> None:
        self.status = {}
        self._save()


class _TeeStream:
    """Forward stdout writes to terminal and pipeline.log at the same time."""

    def __init__(self, original, log_path: Path) -> None:
        self._original = original
        self._log_path = log_path

    def write(self, data: str) -> int:
        self._original.write(data)
        self._original.flush()
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(data)
        return len(data)

    def flush(self) -> None:
        self._original.flush()

    def fileno(self):
        return self._original.fileno()

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self._original, name)


class StageLogger:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / "pipeline.log"
        if not isinstance(sys.stdout, _TeeStream):
            sys.stdout = _TeeStream(sys.stdout, self.log_file)

    def log(self, msg: str) -> None:
        ts = time.strftime("[%Y-%m-%d %H:%M:%S] ", time.localtime())
        line = f"{ts}{msg}"
        print(line, flush=True)


def _format_log_line(msg: str) -> str:
    ts = time.strftime("[%Y-%m-%d %H:%M:%S] ", time.localtime())
    return f"{ts}{msg}"


def append_pipeline_log(output_dir: Path, msg: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "pipeline.log").open("a", encoding="utf-8") as f:
        f.write(_format_log_line(msg) + "\n")


class PipelineRunLock:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.lock_path = self.output_dir / "pipeline.lock.json"
        self.active = False

    @staticmethod
    def _is_pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _read_lock(self) -> Optional[Dict[str, object]]:
        if not self.lock_path.exists():
            return None
        try:
            with self.lock_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def acquire(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        lock_info = self._read_lock()

        if lock_info is not None:
            owner_pid = int(lock_info.get("pid", -1))
            if owner_pid > 0 and self._is_pid_alive(owner_pid):
                started_at = lock_info.get("started_at", "unknown")
                command = lock_info.get("command", "unknown")
                append_pipeline_log(
                    self.output_dir,
                    "Pipeline launch rejected: another active run already holds the lock. "
                    f"pid={owner_pid}, started_at={started_at}, command={command}",
                )
                raise RuntimeError(
                    "Another pipeline run is already active for this output directory. "
                    f"pid={owner_pid}, started_at={started_at}."
                )

            append_pipeline_log(
                self.output_dir,
                f"Detected stale pipeline lock, removing it: {self.lock_path}",
            )
            self.lock_path.unlink(missing_ok=True)

        lock_info = {
            "pid": os.getpid(),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "command": " ".join(sys.argv),
            "cwd": os.getcwd(),
        }

        tmp_path = self.lock_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(lock_info, f, indent=2)
        tmp_path.replace(self.lock_path)
        self.active = True
        atexit.register(self.release)

    def release(self) -> None:
        if not self.active:
            return

        lock_info = self._read_lock()
        if lock_info is not None and int(lock_info.get("pid", -1)) == os.getpid():
            self.lock_path.unlink(missing_ok=True)
        self.active = False
