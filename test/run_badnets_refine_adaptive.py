"""Compatibility wrapper for badnets_refine_adaptive case."""

import sys

from test.wrapper_utils import run_named_case


if __name__ == "__main__":
    raise SystemExit(run_named_case("badnets_refine_adaptive", sys.argv[1:]))
