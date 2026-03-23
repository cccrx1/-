"""Compatibility wrapper for blended_refine case."""

import sys

from test.wrapper_utils import run_named_case


if __name__ == "__main__":
    raise SystemExit(run_named_case("blended_refine", sys.argv[1:]))
