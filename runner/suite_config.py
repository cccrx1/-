"""CLI parsing and runtime config model for CIFAR-10 suite pipeline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class RuntimeConfig:
    dataset_root: str = "./datasets"
    output_root: str = "./experiments/runs"
    adv_dataset_root: str = "./experiments/adv_dataset"
    seed: int = 666
    deterministic: bool = True
    y_target: int = 0
    device_mode: str = "GPU"
    cuda_selected_devices: str = "0"
    batch_size: int = 128
    num_workers: int = 8
    benign_epochs: int = 200
    attack_epochs: int = 200
    lc_epochs: int = 200
    refine_epochs: int = 150
    attack_poisoned_rate_badnets: float = 0.05
    attack_poisoned_rate_blended: float = 0.05
    attack_poisoned_rate_lc: float = 0.25
    lc_eps: float = 8.0
    lc_alpha: float = 1.5
    lc_steps: int = 100
    refine_first_channels: int = 64
    skip_lc: bool = False
    defense_variant: str = "refine"
    cg_threshold: float = 0.35
    cg_temperature: float = 0.10
    cg_strength: float = 1.0
    ssl_temperature: float = 0.07
    ssl_weight: float = 0.02
    only_attack: str = "all"
    attack_cache_root: str = ""
    pretrained_benign_model_path: str = ""
    pretrained_attack_model_path: str = ""
    force_rebuild: bool = False


def parse_suite_args() -> RuntimeConfig:
    parser = argparse.ArgumentParser(description="CIFAR-10 BadNets/Blended/LC + REFINE unified experiments")
    parser.add_argument("--dataset-root", type=str, default="./datasets")
    parser.add_argument("--output-root", type=str, default="./experiments/runs")
    parser.add_argument("--adv-dataset-root", type=str, default="./experiments/adv_dataset")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--y-target", type=int, default=0)
    parser.add_argument("--device-mode", type=str, default="GPU", choices=["GPU", "CPU"])
    parser.add_argument("--cuda-selected-devices", type=str, default="0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--benign-epochs", type=int, default=200)
    parser.add_argument("--attack-epochs", type=int, default=200)
    parser.add_argument("--lc-epochs", type=int, default=None,
                        help="LabelConsistent training epochs; defaults to --attack-epochs if not set")
    parser.add_argument("--refine-epochs", type=int, default=150)
    parser.add_argument("--badnets-rate", type=float, default=0.05)
    parser.add_argument("--blended-rate", type=float, default=0.05)
    parser.add_argument("--lc-rate", type=float, default=0.25)
    parser.add_argument("--lc-eps", type=float, default=8.0)
    parser.add_argument("--lc-alpha", type=float, default=1.5)
    parser.add_argument("--lc-steps", type=int, default=100)
    parser.add_argument("--refine-first-channels", type=int, default=64)
    parser.add_argument("--skip-lc", action="store_true",
                        help="Skip LabelConsistent attack and REFINE on LabelConsistent in this run")
    parser.add_argument("--defense-variant", type=str, default="refine", choices=["refine", "refine_cg", "refine_ssl"],
                        help="Defense backbone variant used in REFINE stage")
    parser.add_argument("--cg-threshold", type=float, default=0.35,
                        help="Suspicion threshold for REFINE_CG gate")
    parser.add_argument("--cg-temperature", type=float, default=0.10,
                        help="Temperature for REFINE_CG gate sharpness")
    parser.add_argument("--cg-strength", type=float, default=1.0,
                        help="Gate strength multiplier for REFINE_CG")
    parser.add_argument("--ssl-temperature", type=float, default=0.07,
                        help="Temperature for REFINE_SSL contrastive logits")
    parser.add_argument("--ssl-weight", type=float, default=0.02,
                        help="Loss weight of self-supervised term in REFINE_SSL")
    parser.add_argument("--only-attack", type=str, default="all",
                        choices=["all", "badnets", "blended", "label_consistent"],
                        help="Only run the specified attack + corresponding REFINE stage")
    parser.add_argument("--attack-cache-root", type=str, default="",
                        help="Optional shared cache root for attack-stage models to enable cross-case reuse")
    parser.add_argument("--pretrained-benign-model-path", type=str, default="",
                        help="Optional path to pretrained benign ResNet-18 model checkpoint")
    parser.add_argument("--pretrained-attack-model-path", type=str, default="",
                        help="Optional path to pretrained attacked ResNet-18 model checkpoint (effective when only one attack stage is run)")
    parser.add_argument("--force-rebuild", action="store_true", default=False,
                        help="Force rebuild all stages, ignore cached results")

    args = parser.parse_args()

    return RuntimeConfig(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        adv_dataset_root=args.adv_dataset_root,
        seed=args.seed,
        deterministic=args.deterministic,
        y_target=args.y_target,
        device_mode=args.device_mode,
        cuda_selected_devices=args.cuda_selected_devices,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        benign_epochs=args.benign_epochs,
        attack_epochs=args.attack_epochs,
        lc_epochs=args.lc_epochs if args.lc_epochs is not None else args.attack_epochs,
        refine_epochs=args.refine_epochs,
        attack_poisoned_rate_badnets=args.badnets_rate,
        attack_poisoned_rate_blended=args.blended_rate,
        attack_poisoned_rate_lc=args.lc_rate,
        lc_eps=args.lc_eps,
        lc_alpha=args.lc_alpha,
        lc_steps=args.lc_steps,
        refine_first_channels=args.refine_first_channels,
        skip_lc=args.skip_lc,
        defense_variant=args.defense_variant,
        cg_threshold=args.cg_threshold,
        cg_temperature=args.cg_temperature,
        cg_strength=args.cg_strength,
        ssl_temperature=args.ssl_temperature,
        ssl_weight=args.ssl_weight,
        only_attack=args.only_attack,
        attack_cache_root=args.attack_cache_root,
        pretrained_benign_model_path=args.pretrained_benign_model_path,
        pretrained_attack_model_path=args.pretrained_attack_model_path,
        force_rebuild=args.force_rebuild,
    )
