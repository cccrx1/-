"""
Unified CIFAR-10 experiment pipeline:
1) train benign model
2) train three attacks (BadNets / Blended / Label-Consistent)
3) train and evaluate REFINE defense on each attacked model

This script only prepares the full workflow. It does not auto-run unless invoked.
"""

import copy
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ConvertImageDtype, PILToTensor, RandomHorizontalFlip

import core
from core.attacks.base import accuracy as attack_accuracy
from runner.pipeline_state import PipelineRunLock, StageLogger, StageStatusManager
from runner.suite_config import RuntimeConfig, parse_suite_args


def build_tensor_transform() -> Compose:
    """Convert PIL image to float tensor in [0, 1] without numpy dependency."""
    return Compose([
        PILToTensor(),
        ConvertImageDtype(torch.float32),
    ])


def set_global_seed(seed: int, deterministic: bool) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def ensure_cifar10_downloaded(dataset_root: str, logger: StageLogger) -> Tuple[torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR10]:
    root = Path(dataset_root)
    cifar_flag = root / "cifar-10-batches-py" / "data_batch_1"
    need_download = not cifar_flag.exists()

    if need_download:
        logger.log("CIFAR-10 not found locally, download=True.")
    else:
        logger.log("CIFAR-10 detected locally, download=False.")

    train_transform = Compose([
        RandomHorizontalFlip(),
        PILToTensor(),
        ConvertImageDtype(torch.float32),
    ])
    test_transform = build_tensor_transform()

    trainset = torchvision.datasets.CIFAR10(
        root=str(root),
        train=True,
        transform=train_transform,
        download=need_download,
    )
    testset = torchvision.datasets.CIFAR10(
        root=str(root),
        train=False,
        transform=test_transform,
        download=need_download,
    )
    return trainset, testset


def build_cifar10_model() -> nn.Module:
    return core.models.ResNet(18)


def make_badnets_trigger() -> Tuple[torch.Tensor, torch.Tensor]:
    pattern = torch.zeros((32, 32), dtype=torch.uint8)
    pattern[-3:, -3:] = 255
    weight = torch.zeros((32, 32), dtype=torch.float32)
    weight[-3:, -3:] = 1.0
    return pattern, weight


def make_blended_trigger() -> Tuple[torch.Tensor, torch.Tensor]:
    pattern = torch.zeros((32, 32), dtype=torch.uint8)
    pattern[-3:, -3:] = 255
    weight = torch.zeros((32, 32), dtype=torch.float32)
    weight[-3:, -3:] = 0.2
    return pattern, weight


def make_lc_trigger() -> Tuple[torch.Tensor, torch.Tensor]:
    pattern = torch.zeros((32, 32), dtype=torch.uint8)
    pattern[-1, -1] = 255
    pattern[-1, -3] = 255
    pattern[-3, -1] = 255
    pattern[-2, -2] = 255
    pattern[0, -1] = 255
    pattern[1, -2] = 255
    pattern[2, -3] = 255
    pattern[2, -1] = 255
    pattern[0, 0] = 255
    pattern[1, 1] = 255
    pattern[2, 2] = 255
    pattern[2, 0] = 255
    pattern[-1, 0] = 255
    pattern[-1, 2] = 255
    pattern[-2, 1] = 255
    pattern[-3, 0] = 255

    weight = torch.zeros((32, 32), dtype=torch.float32)
    weight[:3, :3] = 1.0
    weight[:3, -3:] = 1.0
    weight[-3:, :3] = 1.0
    weight[-3:, -3:] = 1.0
    return pattern, weight


def make_attack_schedule(cfg: RuntimeConfig, experiment_name: str, benign_training: bool, epochs: int) -> Dict:
    schedule = {
        "device": cfg.device_mode,
        "CUDA_SELECTED_DEVICES": cfg.cuda_selected_devices,
        "GPU_num": 1,
        "benign_training": benign_training,
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "gamma": 0.1,
        "schedule": [150, 180],
        "epochs": epochs,
        "log_iteration_interval": 100,
        "test_epoch_interval": 10,
        "save_epoch_interval": 10,
        "save_dir": cfg.output_root,
        "experiment_name": experiment_name,
    }
    return schedule


def make_refine_schedule(cfg: RuntimeConfig, experiment_name: str, epochs: int) -> Dict:
    if epochs < 4:
        refine_milestones: List[int] = []
    else:
        m1 = max(1, int(0.6 * epochs))
        m2 = max(m1 + 1, int(0.8 * epochs))
        if m2 >= epochs:
            m2 = epochs - 1
        refine_milestones = [m1] if m1 >= m2 else [m1, m2]

    schedule = {
        "device": cfg.device_mode,
        "CUDA_VISIBLE_DEVICES": cfg.cuda_selected_devices,
        "GPU_num": 1,
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "lr": 0.01,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.0,
        "amsgrad": False,
        "schedule": refine_milestones,
        "gamma": 0.1,
        "epochs": epochs,
        "log_iteration_interval": 100,
        "test_epoch_interval": 10,
        "save_epoch_interval": 10,
        "save_dir": cfg.output_root,
        "experiment_name": experiment_name,
    }
    return schedule


def pick_eval_device(cfg: RuntimeConfig) -> torch.device:
    if cfg.device_mode == "GPU" and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def evaluate_attack_base(base_obj, dataset, device: torch.device, batch_size: int, num_workers: int) -> Dict[str, float]:
    predict_digits, labels, mean_loss = base_obj._test(
        dataset=dataset,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    total_num = labels.size(0)
    prec1, prec5 = attack_accuracy(predict_digits, labels, topk=(1, 5))
    return {
        "top1_acc": float(prec1.item() / 100.0),
        "top5_acc": float(prec5.item() / 100.0),
        "total_num": int(total_num),
        "mean_loss": float(mean_loss),
    }


def evaluate_refine(defense, dataset, device: torch.device, batch_size: int, num_workers: int) -> Dict[str, float]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )
    defense.unet = defense.unet.to(device)
    defense.unet.eval()
    defense.model = defense.model.to(device)
    defense.model.eval()

    if device.type != "cuda":
        def _label_shuffle_cpu(label):
            label_new = torch.zeros_like(label)
            index = torch.from_numpy(defense.arr_shuffle).repeat(label.shape[0], 1).to(label.device)
            return label_new.scatter(1, index, label)

        defense.label_shuffle = _label_shuffle_cpu

    logits_all: List[torch.Tensor] = []
    labels_all: List[torch.Tensor] = []

    with torch.no_grad():
        for batch_img, batch_label in loader:
            batch_img = batch_img.to(device)
            batch_logits = defense.forward(batch_img)
            logits_all.append(batch_logits.cpu())
            labels_all.append(batch_label.cpu())

    predict_digits = torch.cat(logits_all, dim=0)
    labels = torch.cat(labels_all, dim=0)
    total_num = labels.size(0)
    prec1, prec5 = attack_accuracy(predict_digits, labels, topk=(1, 5))

    return {
        "top1_acc": float(prec1.item() / 100.0),
        "top5_acc": float(prec5.item() / 100.0),
        "total_num": int(total_num),
    }


def build_badnets_attack(cfg: RuntimeConfig, trainset, testset):
    pattern, weight = make_badnets_trigger()
    return core.BadNets(
        train_dataset=trainset,
        test_dataset=testset,
        model=build_cifar10_model(),
        loss=nn.CrossEntropyLoss(),
        y_target=cfg.y_target,
        poisoned_rate=cfg.attack_poisoned_rate_badnets,
        pattern=pattern,
        weight=weight,
        seed=cfg.seed,
        deterministic=cfg.deterministic,
    )


def build_blended_attack(cfg: RuntimeConfig, trainset, testset):
    pattern, weight = make_blended_trigger()
    return core.Blended(
        train_dataset=trainset,
        test_dataset=testset,
        model=build_cifar10_model(),
        loss=nn.CrossEntropyLoss(),
        y_target=cfg.y_target,
        poisoned_rate=cfg.attack_poisoned_rate_blended,
        pattern=pattern,
        weight=weight,
        seed=cfg.seed,
        deterministic=cfg.deterministic,
    )


def build_lc_adv_dir(cfg: RuntimeConfig) -> Path:
    return Path(cfg.adv_dataset_root) / (
        f"CIFAR10_eps{cfg.lc_eps}_alpha{cfg.lc_alpha}_steps{cfg.lc_steps}_"
        f"poisoned_rate{cfg.attack_poisoned_rate_lc}_seed{cfg.seed}"
    )


def build_label_consistent_attack(cfg: RuntimeConfig, trainset, testset, benign_model: nn.Module):
    pattern, weight = make_lc_trigger()
    adv_dir = build_lc_adv_dir(cfg)
    attack = core.LabelConsistent(
        train_dataset=trainset,
        test_dataset=testset,
        model=build_cifar10_model(),
        adv_model=copy.deepcopy(benign_model),
        adv_dataset_dir=str(adv_dir),
        loss=nn.CrossEntropyLoss(),
        y_target=cfg.y_target,
        poisoned_rate=cfg.attack_poisoned_rate_lc,
        adv_transform=build_tensor_transform(),
        pattern=pattern,
        weight=weight,
        eps=cfg.lc_eps,
        alpha=cfg.lc_alpha,
        steps=cfg.lc_steps,
        max_pixel=255,
        poisoned_transform_train_index=0,
        poisoned_transform_test_index=0,
        poisoned_target_transform_index=0,
        schedule=make_attack_schedule(cfg, "CIFAR10_ResNet18_LabelConsistent", benign_training=False, epochs=cfg.lc_epochs),
        seed=cfg.seed,
        deterministic=cfg.deterministic,
    )
    return attack, adv_dir


def build_attack_metrics(cfg: RuntimeConfig, attack_obj, attack_name: str, adv_dataset_dir: Optional[Path] = None) -> Dict[str, object]:
    eval_device = pick_eval_device(cfg)
    metrics: Dict[str, object] = {
        "attack": attack_name,
        "clean": evaluate_attack_base(attack_obj, attack_obj.test_dataset, eval_device, cfg.batch_size, cfg.num_workers),
        "poisoned": evaluate_attack_base(attack_obj, attack_obj.poisoned_test_dataset, eval_device, cfg.batch_size, cfg.num_workers),
    }
    if adv_dataset_dir is not None:
        metrics["adv_dataset_dir"] = str(adv_dataset_dir)
    return metrics


def run_benign_training(
    cfg: RuntimeConfig,
    trainset,
    testset,
    logger: StageLogger,
) -> nn.Module:
    logger.log("Stage[Benign]: start training benign ResNet-18 on CIFAR-10.")

    benign = core.BadNets(
        train_dataset=trainset,
        test_dataset=testset,
        model=build_cifar10_model(),
        loss=nn.CrossEntropyLoss(),
        y_target=cfg.y_target,
        poisoned_rate=0.0,
        pattern=None,
        weight=None,
        seed=cfg.seed,
        deterministic=cfg.deterministic,
    )

    schedule = make_attack_schedule(cfg, "CIFAR10_ResNet18_Benign", benign_training=True, epochs=cfg.benign_epochs)
    benign.train(schedule)

    eval_device = pick_eval_device(cfg)
    clean_metrics = evaluate_attack_base(benign, testset, eval_device, cfg.batch_size, cfg.num_workers)
    logger.log(
        "Stage[Benign]: done. "
        f"BA(top1)={clean_metrics['top1_acc']:.4f}, "
        f"loss={clean_metrics['mean_loss']:.6f}."
    )

    benign_model = copy.deepcopy(benign.get_model())
    benign_model.eval()
    return benign_model


def run_badnets(
    cfg: RuntimeConfig,
    trainset,
    testset,
    logger: StageLogger,
) -> Tuple[object, Dict]:
    logger.log("Stage[BadNets]: start.")
    attack = build_badnets_attack(cfg, trainset, testset)

    schedule = make_attack_schedule(cfg, "CIFAR10_ResNet18_BadNets", benign_training=False, epochs=cfg.attack_epochs)
    attack.train(schedule)

    metrics = build_attack_metrics(cfg, attack, "BadNets")
    logger.log(
        "Stage[BadNets]: done. "
        f"BA(top1)={metrics['clean']['top1_acc']:.4f}, "
        f"ASR(top1)={metrics['poisoned']['top1_acc']:.4f}."
    )
    return attack, metrics


def run_blended(
    cfg: RuntimeConfig,
    trainset,
    testset,
    logger: StageLogger,
) -> Tuple[object, Dict]:
    logger.log("Stage[Blended]: start.")
    attack = build_blended_attack(cfg, trainset, testset)

    schedule = make_attack_schedule(cfg, "CIFAR10_ResNet18_Blended", benign_training=False, epochs=cfg.attack_epochs)
    attack.train(schedule)

    metrics = build_attack_metrics(cfg, attack, "Blended")
    logger.log(
        "Stage[Blended]: done. "
        f"BA(top1)={metrics['clean']['top1_acc']:.4f}, "
        f"ASR(top1)={metrics['poisoned']['top1_acc']:.4f}."
    )
    return attack, metrics


def run_label_consistent(
    cfg: RuntimeConfig,
    trainset,
    testset,
    benign_model: nn.Module,
    logger: StageLogger,
) -> Tuple[object, Dict]:
    logger.log("Stage[LabelConsistent]: start.")
    attack, adv_dir = build_label_consistent_attack(cfg, trainset, testset, benign_model)

    attack.train()

    metrics = build_attack_metrics(cfg, attack, "LabelConsistent", adv_dataset_dir=adv_dir)
    logger.log(
        "Stage[LabelConsistent]: done. "
        f"BA(top1)={metrics['clean']['top1_acc']:.4f}, "
        f"ASR(top1)={metrics['poisoned']['top1_acc']:.4f}."
    )
    return attack, metrics


def run_refine_for_attack(
    cfg: RuntimeConfig,
    attack_name: str,
    attacked_model: nn.Module,
    trainset,
    clean_testset,
    poisoned_testset,
    logger: StageLogger,
) -> Dict:
    logger.log(f"Stage[REFINE/{attack_name}]: start training defense.")

    defense_kwargs = {
        "unet": core.models.UNetLittle(args=None, n_channels=3, n_classes=3, first_channels=cfg.refine_first_channels),
        "model": copy.deepcopy(attacked_model),
        "num_classes": 10,
        "seed": cfg.seed,
        "deterministic": cfg.deterministic,
    }

    if cfg.defense_variant == "refine_cg":
        defense = core.REFINE_CG(
            **defense_kwargs,
            cg_threshold=cfg.cg_threshold,
            cg_temperature=cfg.cg_temperature,
            cg_strength=cfg.cg_strength,
            cg_enable=True,
        )
    elif cfg.defense_variant == "refine_ssl":
        defense = core.REFINE_SSL(
            **defense_kwargs,
            temperature=cfg.ssl_temperature,
            selfsup_weight=cfg.ssl_weight,
        )
    else:
        defense = core.REFINE(**defense_kwargs)

    train_schedule = make_refine_schedule(cfg, f"CIFAR10_REFINE_train_{attack_name}", epochs=cfg.refine_epochs)
    defense.train_unet(trainset, clean_testset, train_schedule)

    eval_device = pick_eval_device(cfg)
    clean_metrics = evaluate_refine(defense, clean_testset, eval_device, cfg.batch_size, cfg.num_workers)
    poisoned_metrics = evaluate_refine(defense, poisoned_testset, eval_device, cfg.batch_size, cfg.num_workers)

    result = {
        "attack": attack_name,
        "defense_variant": cfg.defense_variant,
        "clean_after_refine": clean_metrics,
        "poisoned_after_refine": poisoned_metrics,
    }

    logger.log(
        f"Stage[REFINE/{attack_name}]: done. "
        f"BA(top1)={clean_metrics['top1_acc']:.4f}, "
        f"ASR(top1)={poisoned_metrics['top1_acc']:.4f}."
    )
    return result


def to_builtin(obj):
    if isinstance(obj, dict):
        return {k: to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def get_cache_dir(output_dir: Path) -> Path:
    """Get the cache directory for storing intermediate results."""
    cache_dir = output_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def make_attack_cache_key(attack_name: str, cfg: RuntimeConfig) -> str:
    payload = {
        "attack": attack_name,
        "seed": cfg.seed,
        "deterministic": cfg.deterministic,
        "y_target": cfg.y_target,
        "attack_epochs": cfg.attack_epochs,
        "lc_epochs": cfg.lc_epochs,
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "badnets_rate": cfg.attack_poisoned_rate_badnets,
        "blended_rate": cfg.attack_poisoned_rate_blended,
        "lc_rate": cfg.attack_poisoned_rate_lc,
        "lc_eps": cfg.lc_eps,
        "lc_alpha": cfg.lc_alpha,
        "lc_steps": cfg.lc_steps,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


def build_run_signature(cfg: RuntimeConfig) -> Dict[str, object]:
    return {
        "seed": cfg.seed,
        "deterministic": cfg.deterministic,
        "y_target": cfg.y_target,
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "benign_epochs": cfg.benign_epochs,
        "attack_epochs": cfg.attack_epochs,
        "lc_epochs": cfg.lc_epochs,
        "refine_epochs": cfg.refine_epochs,
        "defense_variant": cfg.defense_variant,
        "badnets_rate": cfg.attack_poisoned_rate_badnets,
        "blended_rate": cfg.attack_poisoned_rate_blended,
        "lc_rate": cfg.attack_poisoned_rate_lc,
        "lc_eps": cfg.lc_eps,
        "lc_alpha": cfg.lc_alpha,
        "lc_steps": cfg.lc_steps,
        "refine_first_channels": cfg.refine_first_channels,
        "cg_threshold": cfg.cg_threshold,
        "cg_temperature": cfg.cg_temperature,
        "cg_strength": cfg.cg_strength,
        "ssl_temperature": cfg.ssl_temperature,
        "ssl_weight": cfg.ssl_weight,
        "only_attack": cfg.only_attack,
        "skip_lc": cfg.skip_lc,
        "pretrained_benign_model_path": cfg.pretrained_benign_model_path,
        "pretrained_attack_model_path": cfg.pretrained_attack_model_path,
    }


def resolve_attack_cache_root(cfg: RuntimeConfig, output_dir: Path) -> Path:
    if cfg.attack_cache_root:
        return Path(cfg.attack_cache_root)
    return output_dir


def save_model_cache(model: nn.Module, name: str, output_dir: Path, cache_key: str = "") -> None:
    """Save model to cache."""
    cache_dir = get_cache_dir(output_dir)
    suffix = f"_{cache_key}" if cache_key else ""
    cache_path = cache_dir / f"{name}{suffix}_model.pth"
    torch.save(model.state_dict(), cache_path)


def load_model_cache(model: nn.Module, name: str, output_dir: Path, cache_key: str = "") -> Optional[nn.Module]:
    """Load model from cache, return None if not found."""
    cache_dir = get_cache_dir(output_dir)
    suffix = f"_{cache_key}" if cache_key else ""
    cache_path = cache_dir / f"{name}{suffix}_model.pth"
    if not cache_path.exists():
        return None
    try:
        model.load_state_dict(torch.load(cache_path, map_location='cpu'))
        return model
    except Exception:
        return None


def save_refine_cache(results: Dict, name: str, output_dir: Path) -> None:
    """Save REFINE results to cache."""
    cache_dir = get_cache_dir(output_dir)
    cache_path = cache_dir / f"{name}_refine_results.json"
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(to_builtin(results), f, indent=2)


def load_refine_cache(name: str, output_dir: Path) -> Optional[Dict]:
    """Load REFINE results from cache, return None if not found."""
    cache_dir = get_cache_dir(output_dir)
    cache_path = cache_dir / f"{name}_refine_results.json"
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_pretrained_cifar10_model(model_path: str, logger: StageLogger, stage_label: str) -> Optional[nn.Module]:
    if not model_path:
        return None

    path = Path(model_path)
    if not path.exists():
        logger.log(f"{stage_label}: pretrained model path does not exist: {path}")
        return None

    try:
        state_obj = torch.load(path, map_location="cpu")
        if isinstance(state_obj, dict) and "state_dict" in state_obj and isinstance(state_obj["state_dict"], dict):
            state_obj = state_obj["state_dict"]

        if isinstance(state_obj, dict):
            cleaned_state = {}
            for key, value in state_obj.items():
                if key.startswith("module."):
                    cleaned_state[key[len("module."):]] = value
                else:
                    cleaned_state[key] = value
            state_obj = cleaned_state

        model = build_cifar10_model()
        model.load_state_dict(state_obj, strict=True)
        model.eval()
        logger.log(f"{stage_label}: loaded pretrained model from {path}.")
        return model
    except Exception as exc:
        logger.log(f"{stage_label}: failed to load pretrained model from {path}, fallback to default flow. reason={exc}")
        return None


def run_or_load_refine_stage(
    cfg: RuntimeConfig,
    stage_status: StageStatusManager,
    output_dir: Path,
    stage_elapsed_seconds: Dict[str, float],
    all_metrics: Dict[str, object],
    stage_key: str,
    cache_name: str,
    attack_name: str,
    attacked_model: nn.Module,
    trainset,
    testset,
    poisoned_testset,
    logger: StageLogger,
) -> None:
    stage_label = f"Stage[REFINE/{attack_name}]"
    if stage_status.is_completed(stage_key, cfg.force_rebuild):
        logger.log(f"{stage_label}: loading from cache.")
        refine_result = load_refine_cache(cache_name, output_dir)
        if refine_result is not None:
            logger.log(f"{stage_label}: loaded from cache successfully.")
            stage_elapsed_seconds[stage_key] = 0.0
            all_metrics["stages"][stage_key] = refine_result
            return

        logger.log(f"{stage_label}: cache load failed, rebuilding.")
        stage_status.mark_failed(stage_key, "cache load failed")

    stage_t0 = time.time()
    refine_result = run_refine_for_attack(
        cfg,
        attack_name,
        attacked_model,
        trainset,
        testset,
        poisoned_testset,
        logger,
    )
    save_refine_cache(refine_result, cache_name, output_dir)
    stage_elapsed_seconds[stage_key] = time.time() - stage_t0
    stage_status.mark_completed(stage_key)
    all_metrics["stages"][stage_key] = refine_result


def run_or_load_attack_stage(
    cfg: RuntimeConfig,
    stage_status: StageStatusManager,
    attack_cache_root: Path,
    stage_elapsed_seconds: Dict[str, float],
    all_metrics: Dict[str, object],
    stage_key: str,
    cache_name: str,
    cache_key: str,
    stage_display_name: str,
    attack_metric_name: str,
    pretrained_model_path: str,
    build_attack_fn: Callable[[], Tuple[object, Optional[Path]]],
    run_attack_fn: Callable[[], Tuple[object, Dict]],
    logger: StageLogger,
) -> object:
    stage_label = f"Stage[{stage_display_name}]"
    attack_obj, adv_dir = build_attack_fn()

    pretrained_model = load_pretrained_cifar10_model(pretrained_model_path, logger, stage_label)
    if pretrained_model is not None:
        attack_obj.model = pretrained_model
        metrics = build_attack_metrics(cfg, attack_obj, attack_metric_name, adv_dataset_dir=adv_dir)
        logger.log(f"{stage_label}: using --pretrained-attack-model-path.")
        save_model_cache(attack_obj.get_model(), cache_name, attack_cache_root, cache_key)
        stage_elapsed_seconds[stage_key] = 0.0
        stage_status.mark_completed(stage_key, metrics)
        all_metrics["stages"][stage_key] = metrics
        return attack_obj

    if stage_status.is_completed(stage_key, cfg.force_rebuild):
        logger.log(f"{stage_label}: loading from cache.")
        cached_model = load_model_cache(build_cifar10_model(), cache_name, attack_cache_root, cache_key)
        if cached_model is not None:
            attack_obj.model = cached_model
            metrics = build_attack_metrics(cfg, attack_obj, attack_metric_name, adv_dataset_dir=adv_dir)
            logger.log(f"{stage_label}: loaded from cache successfully.")
            stage_elapsed_seconds[stage_key] = 0.0
            all_metrics["stages"][stage_key] = metrics
            return attack_obj

        logger.log(f"{stage_label}: cache load failed, rebuilding.")
        stage_status.mark_failed(stage_key, "cache load failed")
        stage_t0 = time.time()
        attack_obj, metrics = run_attack_fn()
        save_model_cache(attack_obj.get_model(), cache_name, attack_cache_root, cache_key)
        stage_elapsed_seconds[stage_key] = time.time() - stage_t0
        stage_status.mark_completed(stage_key, metrics)
        all_metrics["stages"][stage_key] = metrics
        return attack_obj

    cached_model = None
    if not cfg.force_rebuild:
        cached_model = load_model_cache(build_cifar10_model(), cache_name, attack_cache_root, cache_key)

    if cached_model is not None:
        attack_obj.model = cached_model
        metrics = build_attack_metrics(cfg, attack_obj, attack_metric_name, adv_dataset_dir=adv_dir)
        logger.log(f"{stage_label}: reused shared attack cache.")
        stage_elapsed_seconds[stage_key] = 0.0
        stage_status.mark_completed(stage_key, metrics)
        all_metrics["stages"][stage_key] = metrics
        return attack_obj

    stage_t0 = time.time()
    attack_obj, metrics = run_attack_fn()
    save_model_cache(attack_obj.get_model(), cache_name, attack_cache_root, cache_key)
    stage_elapsed_seconds[stage_key] = time.time() - stage_t0
    stage_status.mark_completed(stage_key, metrics)
    all_metrics["stages"][stage_key] = metrics
    return attack_obj


def write_pipeline_outputs(output_dir: Path, all_metrics: Dict[str, object]) -> Tuple[Path, Path]:
    serializable_metrics = to_builtin(all_metrics)

    summary_path = output_dir / "metrics_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(serializable_metrics, f, indent=2)

    report_path = output_dir / "comparison_report.md"
    report_path.write_text(build_comparison_markdown(serializable_metrics), encoding="utf-8")
    return summary_path, report_path


def build_comparison_markdown(all_metrics: Dict[str, object]) -> str:
    stages = all_metrics.get("stages", {}) if isinstance(all_metrics, dict) else {}
    timing = all_metrics.get("timing", {}) if isinstance(all_metrics, dict) else {}

    lines: List[str] = []
    lines.append("# Experiment Comparison Report")
    lines.append("")
    total_seconds = timing.get("total_elapsed_seconds")
    if isinstance(total_seconds, (int, float)):
        lines.append(f"- Total elapsed: {format_duration(float(total_seconds))} ({float(total_seconds):.1f}s)")
    lines.append("")
    lines.append("## Attack / Defense Metrics")
    lines.append("")
    lines.append("| Stage | Clean Top1 | Poisoned Top1 | Top1 Delta (Poisoned-Clean) |")
    lines.append("|---|---:|---:|---:|")

    def _append_row(stage_key: str, stage_name: str, clean_key: str, poison_key: str) -> None:
        stage = stages.get(stage_key, {}) if isinstance(stages, dict) else {}
        clean = stage.get(clean_key, {}) if isinstance(stage, dict) else {}
        poison = stage.get(poison_key, {}) if isinstance(stage, dict) else {}
        c = clean.get("top1_acc") if isinstance(clean, dict) else None
        p = poison.get("top1_acc") if isinstance(poison, dict) else None
        if isinstance(c, (int, float)) and isinstance(p, (int, float)):
            delta = float(p) - float(c)
            lines.append(f"| {stage_name} | {float(c):.4f} | {float(p):.4f} | {delta:+.4f} |")

    _append_row("badnets", "BadNets (Attack)", "clean", "poisoned")
    _append_row("blended", "Blended (Attack)", "clean", "poisoned")
    _append_row("label_consistent", "LabelConsistent (Attack)", "clean", "poisoned")
    _append_row("refine_badnets", "REFINE on BadNets", "clean_after_refine", "poisoned_after_refine")
    _append_row("refine_blended", "REFINE on Blended", "clean_after_refine", "poisoned_after_refine")
    _append_row("refine_label_consistent", "REFINE on LabelConsistent", "clean_after_refine", "poisoned_after_refine")

    lines.append("")
    lines.append("## Stage Timing")
    lines.append("")
    lines.append("| Stage | Elapsed (HH:MM:SS) | Seconds |")
    lines.append("|---|---:|---:|")
    stage_seconds = timing.get("stage_elapsed_seconds") if isinstance(timing, dict) else None
    if isinstance(stage_seconds, dict):
        for key, value in stage_seconds.items():
            if isinstance(value, (int, float)):
                lines.append(f"| {key} | {format_duration(float(value))} | {float(value):.1f} |")

    lines.append("")
    return "\n".join(lines)


def parse_args() -> RuntimeConfig:
    """Backward-compatible wrapper around centralized suite config parser."""
    return parse_suite_args()


def log_runtime_environment(cfg: RuntimeConfig, logger: StageLogger) -> None:
    py_ver = sys.version.split()[0]
    torch_ver = getattr(torch, "__version__", "unknown")
    torchvision_ver = getattr(torchvision, "__version__", "unknown")
    numpy_ver = getattr(np, "__version__", "unknown")
    cuda_available = torch.cuda.is_available()
    cuda_runtime = torch.version.cuda if getattr(torch, "version", None) is not None else None

    logger.log(
        "Environment: "
        f"python={py_ver}, "
        f"torch={torch_ver}, "
        f"torchvision={torchvision_ver}, "
        f"numpy={numpy_ver}, "
        f"cuda_available={cuda_available}, "
        f"torch_cuda={cuda_runtime}"
    )

    if cfg.device_mode == "GPU" and not cuda_available:
        logger.log("[WARN] device_mode=GPU but CUDA is unavailable. Consider switching to --device-mode CPU.")

    if numpy_ver.startswith("2.") and (
        torchvision_ver.startswith("0.17")
        or torchvision_ver.startswith("0.18")
        or torchvision_ver.startswith("0.19")
    ):
        logger.log("[WARN] numpy>=2 with torchvision<0.20 may cause ToTensor compatibility issues in some environments.")


def main() -> None:
    cfg = parse_args()

    if cfg.only_attack == "label_consistent" and cfg.skip_lc:
        cfg.skip_lc = False

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_selected_devices

    output_dir = Path(cfg.output_root)
    attack_cache_root = resolve_attack_cache_root(cfg, output_dir)
    attack_cache_root.mkdir(parents=True, exist_ok=True)
    run_lock = PipelineRunLock(output_dir)
    run_lock.acquire()
    logger = StageLogger(output_dir)
    stage_status = StageStatusManager(output_dir)
    pipeline_start = time.time()
    stage_elapsed_seconds: Dict[str, float] = {}

    try:
        logger.log("Pipeline init: CIFAR-10 benign + BadNets/Blended/LabelConsistent + REFINE.")
        logger.log(f"Run metadata: pid={os.getpid()}, cwd={os.getcwd()}, command={' '.join(sys.argv)}")
        logger.log(f"Runtime config: {asdict(cfg)}")
        log_runtime_environment(cfg, logger)

        run_signature = build_run_signature(cfg)
        previous_signature = stage_status.status.get("_run_signature")
        if previous_signature is not None and previous_signature != run_signature and not cfg.force_rebuild:
            logger.log("Run signature changed since previous run. Resetting stage status to avoid stale-cache comparability drift.")
            stage_status.reset()
        stage_status.status["_run_signature"] = run_signature
        stage_status._save()

        set_global_seed(cfg.seed, cfg.deterministic)

        trainset, testset = ensure_cifar10_downloaded(cfg.dataset_root, logger)

        all_metrics: Dict[str, object] = {
            "runtime_config": asdict(cfg),
            "stages": {},
        }

        run_badnets_flag = cfg.only_attack in ("all", "badnets")
        run_blended_flag = cfg.only_attack in ("all", "blended")
        run_lc_flag = cfg.only_attack in ("all", "label_consistent") and (not cfg.skip_lc)
        if cfg.pretrained_attack_model_path and cfg.only_attack == "all":
            logger.log("[WARN] --pretrained-attack-model-path is ignored when --only-attack=all. Please specify one attack stage.")

        benign_model = None
        if run_lc_flag:
            benign_model = load_pretrained_cifar10_model(
                cfg.pretrained_benign_model_path,
                logger,
                "Stage[Benign]",
            )
            if benign_model is not None:
                stage_elapsed_seconds["benign"] = 0.0
                stage_status.mark_completed("benign")
                all_metrics["stages"]["benign"] = {
                    "model": "ResNet18",
                    "dataset": "CIFAR10",
                    "source": "pretrained",
                    "path": cfg.pretrained_benign_model_path,
                }
            elif stage_status.is_completed("benign", cfg.force_rebuild):
                logger.log("Stage[Benign]: loading from cache.")
                benign_model = load_model_cache(build_cifar10_model(), "benign", output_dir)
                if benign_model is None:
                    logger.log("Stage[Benign]: cache load failed, rebuilding.")
                    stage_status.mark_failed("benign", "cache load failed")
                    stage_t0 = time.time()
                    benign_model = run_benign_training(cfg, trainset, testset, logger)
                    save_model_cache(benign_model, "benign", output_dir)
                    stage_elapsed_seconds["benign"] = time.time() - stage_t0
                    stage_status.mark_completed("benign")
                else:
                    logger.log("Stage[Benign]: loaded from cache successfully.")
                    stage_elapsed_seconds["benign"] = 0.0
            else:
                stage_t0 = time.time()
                benign_model = run_benign_training(cfg, trainset, testset, logger)
                save_model_cache(benign_model, "benign", output_dir)
                stage_elapsed_seconds["benign"] = time.time() - stage_t0
                stage_status.mark_completed("benign")
            if "benign" not in all_metrics["stages"]:
                all_metrics["stages"]["benign"] = {
                    "model": "ResNet18",
                    "dataset": "CIFAR10",
                }
        else:
            logger.log("Stage[Benign]: skipped (only required for LabelConsistent pipeline).")

        badnets_attack = None
        if run_badnets_flag:
            badnets_cache_key = make_attack_cache_key("badnets", cfg)
            badnets_attack = run_or_load_attack_stage(
                cfg=cfg,
                stage_status=stage_status,
                attack_cache_root=attack_cache_root,
                stage_elapsed_seconds=stage_elapsed_seconds,
                all_metrics=all_metrics,
                stage_key="badnets",
                cache_name="badnets",
                cache_key=badnets_cache_key,
                stage_display_name="BadNets",
                attack_metric_name="BadNets",
                pretrained_model_path=cfg.pretrained_attack_model_path if cfg.only_attack == "badnets" else "",
                build_attack_fn=lambda: (build_badnets_attack(cfg, trainset, testset), None),
                run_attack_fn=lambda: run_badnets(cfg, trainset, testset, logger),
                logger=logger,
            )
        else:
            logger.log("Stage[BadNets]: skipped by --only-attack.")

        blended_attack = None
        if run_blended_flag:
            blended_cache_key = make_attack_cache_key("blended", cfg)
            blended_attack = run_or_load_attack_stage(
                cfg=cfg,
                stage_status=stage_status,
                attack_cache_root=attack_cache_root,
                stage_elapsed_seconds=stage_elapsed_seconds,
                all_metrics=all_metrics,
                stage_key="blended",
                cache_name="blended",
                cache_key=blended_cache_key,
                stage_display_name="Blended",
                attack_metric_name="Blended",
                pretrained_model_path=cfg.pretrained_attack_model_path if cfg.only_attack == "blended" else "",
                build_attack_fn=lambda: (build_blended_attack(cfg, trainset, testset), None),
                run_attack_fn=lambda: run_blended(cfg, trainset, testset, logger),
                logger=logger,
            )
        else:
            logger.log("Stage[Blended]: skipped by --only-attack.")

        lc_attack = None
        if run_lc_flag:
            lc_cache_key = make_attack_cache_key("label_consistent", cfg)
            lc_attack = run_or_load_attack_stage(
                cfg=cfg,
                stage_status=stage_status,
                attack_cache_root=attack_cache_root,
                stage_elapsed_seconds=stage_elapsed_seconds,
                all_metrics=all_metrics,
                stage_key="label_consistent",
                cache_name="label_consistent",
                cache_key=lc_cache_key,
                stage_display_name="LabelConsistent",
                attack_metric_name="LabelConsistent",
                pretrained_model_path=cfg.pretrained_attack_model_path if cfg.only_attack == "label_consistent" else "",
                build_attack_fn=lambda: build_label_consistent_attack(cfg, trainset, testset, benign_model),
                run_attack_fn=lambda: run_label_consistent(cfg, trainset, testset, benign_model, logger),
                logger=logger,
            )
        else:
            if cfg.skip_lc:
                logger.log("Stage[LabelConsistent]: skipped by --skip-lc.")
            else:
                logger.log("Stage[LabelConsistent]: skipped by --only-attack.")

        if badnets_attack is not None:
            run_or_load_refine_stage(
                cfg=cfg,
                stage_status=stage_status,
                output_dir=output_dir,
                stage_elapsed_seconds=stage_elapsed_seconds,
                all_metrics=all_metrics,
                stage_key="refine_badnets",
                cache_name="badnets",
                attack_name="BadNets",
                attacked_model=badnets_attack.get_model(),
                trainset=trainset,
                testset=testset,
                poisoned_testset=badnets_attack.poisoned_test_dataset,
                logger=logger,
            )
        else:
            logger.log("Stage[REFINE/BadNets]: skipped.")

        if blended_attack is not None:
            run_or_load_refine_stage(
                cfg=cfg,
                stage_status=stage_status,
                output_dir=output_dir,
                stage_elapsed_seconds=stage_elapsed_seconds,
                all_metrics=all_metrics,
                stage_key="refine_blended",
                cache_name="blended",
                attack_name="Blended",
                attacked_model=blended_attack.get_model(),
                trainset=trainset,
                testset=testset,
                poisoned_testset=blended_attack.poisoned_test_dataset,
                logger=logger,
            )
        else:
            logger.log("Stage[REFINE/Blended]: skipped.")

        if lc_attack is not None:
            run_or_load_refine_stage(
                cfg=cfg,
                stage_status=stage_status,
                output_dir=output_dir,
                stage_elapsed_seconds=stage_elapsed_seconds,
                all_metrics=all_metrics,
                stage_key="refine_label_consistent",
                cache_name="label_consistent",
                attack_name="LabelConsistent",
                attacked_model=lc_attack.get_model(),
                trainset=trainset,
                testset=testset,
                poisoned_testset=lc_attack.poisoned_test_dataset,
                logger=logger,
            )
        else:
            logger.log("Stage[REFINE/LabelConsistent]: skipped.")

        total_elapsed = time.time() - pipeline_start
        all_metrics["timing"] = {
            "total_elapsed_seconds": total_elapsed,
            "total_elapsed_hms": format_duration(total_elapsed),
            "stage_elapsed_seconds": stage_elapsed_seconds,
            "stage_elapsed_hms": {k: format_duration(v) for k, v in stage_elapsed_seconds.items()},
        }

        summary_path, report_path = write_pipeline_outputs(output_dir, all_metrics)

        logger.log(
            f"Pipeline completed in {format_duration(total_elapsed)}. "
            f"Summary saved to: {summary_path}; report saved to: {report_path}"
        )
    finally:
        run_lock.release()


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
