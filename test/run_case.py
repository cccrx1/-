"""Compatibility shim for case runner; delegates to core.pipeline.run_case."""

try:
    from test._bootstrap import ensure_project_root_on_path
except ModuleNotFoundError:
    from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from core.pipeline.run_case import main


if __name__ == "__main__":
    main()
