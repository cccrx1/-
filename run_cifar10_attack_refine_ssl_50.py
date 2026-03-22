"""
Run CIFAR-10 attack + REFINE suite with self-supervised REFINE loss for 50 epochs.

This wrapper keeps all original defaults/hyperparameters from
`run_cifar10_attack_refine_suite.py` and only changes:
1) --defense-variant refine_ssl
2) --refine-epochs 50
3) default outputs under ./experiments
"""

import sys

from run_cifar10_attack_refine_suite import main


def _inject_default_arg(flag: str, value: str) -> None:
    if flag not in sys.argv:
        sys.argv.extend([flag, value])


if __name__ == "__main__":
    _inject_default_arg("--defense-variant", "refine_ssl")
    _inject_default_arg("--refine-epochs", "50")
    _inject_default_arg("--output-root", "./experiments/runs")
    _inject_default_arg("--adv-dataset-root", "./experiments/adv_dataset")
    main()
