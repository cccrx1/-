"""Compatibility wrapper for badnets_refine_ssl_50 case."""

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
    raise SystemExit(run_named_case("badnets_refine_ssl_50", sys.argv[1:]))
