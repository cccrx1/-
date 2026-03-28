"""Compatibility shim for suite runner; delegates to core.pipeline.run_test_suite."""

try:
    from test._bootstrap import ensure_project_root_on_path
except ModuleNotFoundError:
    from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from core.pipeline.run_test_suite import main


if __name__ == "__main__":
    main()
