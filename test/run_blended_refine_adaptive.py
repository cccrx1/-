"""Compatibility wrapper for blended_refine_adaptive case."""

import sys

try:
    from test._bootstrap import ensure_project_root_on_path
except ModuleNotFoundError:
    from _bootstrap import ensure_project_root_on_path

try:
    from test.wrapper_utils import run_named_case
except ModuleNotFoundError:
    from wrapper_utils import run_named_case


if __name__ == "__main__":
    ensure_project_root_on_path()
    raise SystemExit(run_named_case("blended_refine_adaptive", sys.argv[1:]))
