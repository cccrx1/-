"""
Unified CIFAR-10 experiment pipeline:
1) train benign model
2) train three attacks (BadNets / Blended / Label-Consistent)
3) train and evaluate REFINE defense on each attacked model

This script only prepares the full workflow. It does not auto-run unless invoked.
"""

import argparse
import atexit
import copy
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomHorizontalFlip, ToTensor

import core
from core.attacks.base import accuracy as attack_accuracy


@dataclass
class RuntimeConfig:
    dataset_root: str = "./datasets"
    output_root: str = "./projects/cifar10_attack_refine_suite/runs"
    adv_dataset_root: str = "./projects/cifar10_attack_refine_suite/adv_dataset"
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


class _TeeStream:
    """将写入操作同时转发到原始 stdout 和 pipeline.log 文件。"""
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
        # 将 stdout 接管为 Tee，base.py 的训练日志也同步写入 pipeline.log
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
        ToTensor(),
    ])
    test_transform = Compose([
        ToTensor(),
    ])

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
        "schedule": [100, 130],
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
    pattern, weight = make_badnets_trigger()

    attack = core.BadNets(
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

    schedule = make_attack_schedule(cfg, "CIFAR10_ResNet18_BadNets", benign_training=False, epochs=cfg.attack_epochs)
    attack.train(schedule)

    eval_device = pick_eval_device(cfg)
    clean_metrics = evaluate_attack_base(attack, attack.test_dataset, eval_device, cfg.batch_size, cfg.num_workers)
    asr_metrics = evaluate_attack_base(attack, attack.poisoned_test_dataset, eval_device, cfg.batch_size, cfg.num_workers)

    metrics = {
        "attack": "BadNets",
        "clean": clean_metrics,
        "poisoned": asr_metrics,
    }
    logger.log(
        "Stage[BadNets]: done. "
        f"BA(top1)={clean_metrics['top1_acc']:.4f}, "
        f"ASR(top1)={asr_metrics['top1_acc']:.4f}."
    )
    return attack, metrics


def run_blended(
    cfg: RuntimeConfig,
    trainset,
    testset,
    logger: StageLogger,
) -> Tuple[object, Dict]:
    logger.log("Stage[Blended]: start.")
    pattern, weight = make_blended_trigger()

    attack = core.Blended(
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

    schedule = make_attack_schedule(cfg, "CIFAR10_ResNet18_Blended", benign_training=False, epochs=cfg.attack_epochs)
    attack.train(schedule)

    eval_device = pick_eval_device(cfg)
    clean_metrics = evaluate_attack_base(attack, attack.test_dataset, eval_device, cfg.batch_size, cfg.num_workers)
    asr_metrics = evaluate_attack_base(attack, attack.poisoned_test_dataset, eval_device, cfg.batch_size, cfg.num_workers)

    metrics = {
        "attack": "Blended",
        "clean": clean_metrics,
        "poisoned": asr_metrics,
    }
    logger.log(
        "Stage[Blended]: done. "
        f"BA(top1)={clean_metrics['top1_acc']:.4f}, "
        f"ASR(top1)={asr_metrics['top1_acc']:.4f}."
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
    pattern, weight = make_lc_trigger()

    adv_dir = Path(cfg.adv_dataset_root) / (
        f"CIFAR10_eps{cfg.lc_eps}_alpha{cfg.lc_alpha}_steps{cfg.lc_steps}_"
        f"poisoned_rate{cfg.attack_poisoned_rate_lc}_seed{cfg.seed}"
    )

    attack = core.LabelConsistent(
        train_dataset=trainset,
        test_dataset=testset,
        model=build_cifar10_model(),
        adv_model=copy.deepcopy(benign_model),
        adv_dataset_dir=str(adv_dir),
        loss=nn.CrossEntropyLoss(),
        y_target=cfg.y_target,
        poisoned_rate=cfg.attack_poisoned_rate_lc,
        adv_transform=Compose([ToTensor()]),
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

    attack.train()

    eval_device = pick_eval_device(cfg)
    clean_metrics = evaluate_attack_base(attack, attack.test_dataset, eval_device, cfg.batch_size, cfg.num_workers)
    asr_metrics = evaluate_attack_base(attack, attack.poisoned_test_dataset, eval_device, cfg.batch_size, cfg.num_workers)

    metrics = {
        "attack": "LabelConsistent",
        "clean": clean_metrics,
        "poisoned": asr_metrics,
        "adv_dataset_dir": str(adv_dir),
    }
    logger.log(
        "Stage[LabelConsistent]: done. "
        f"BA(top1)={clean_metrics['top1_acc']:.4f}, "
        f"ASR(top1)={asr_metrics['top1_acc']:.4f}."
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
        defense = core.REFINE_SSL(**defense_kwargs)
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
    parser = argparse.ArgumentParser(description="CIFAR-10 BadNets/Blended/LC + REFINE unified experiments")
    parser.add_argument("--dataset-root", type=str, default="./datasets")
    parser.add_argument("--output-root", type=str, default="./projects/cifar10_attack_refine_suite/runs")
    parser.add_argument("--adv-dataset-root", type=str, default="./projects/cifar10_attack_refine_suite/adv_dataset")
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
    )


def main() -> None:
    cfg = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_selected_devices

    output_dir = Path(cfg.output_root)
    run_lock = PipelineRunLock(output_dir)
    run_lock.acquire()
    logger = StageLogger(output_dir)
    pipeline_start = time.time()
    stage_elapsed_seconds: Dict[str, float] = {}

    try:
        logger.log("Pipeline init: CIFAR-10 benign + BadNets/Blended/LabelConsistent + REFINE.")
        logger.log(f"Run metadata: pid={os.getpid()}, cwd={os.getcwd()}, command={' '.join(sys.argv)}")
        logger.log(f"Runtime config: {asdict(cfg)}")

        set_global_seed(cfg.seed, cfg.deterministic)

        trainset, testset = ensure_cifar10_downloaded(cfg.dataset_root, logger)

        all_metrics: Dict[str, object] = {
            "runtime_config": asdict(cfg),
            "stages": {},
        }

        stage_t0 = time.time()
        benign_model = run_benign_training(cfg, trainset, testset, logger)
        stage_elapsed_seconds["benign"] = time.time() - stage_t0
        all_metrics["stages"]["benign"] = {
            "model": "ResNet18",
            "dataset": "CIFAR10",
        }

        stage_t0 = time.time()
        badnets_attack, badnets_metrics = run_badnets(cfg, trainset, testset, logger)
        stage_elapsed_seconds["badnets"] = time.time() - stage_t0
        all_metrics["stages"]["badnets"] = badnets_metrics

        stage_t0 = time.time()
        blended_attack, blended_metrics = run_blended(cfg, trainset, testset, logger)
        stage_elapsed_seconds["blended"] = time.time() - stage_t0
        all_metrics["stages"]["blended"] = blended_metrics

        lc_attack = None
        if cfg.skip_lc:
            logger.log("Stage[LabelConsistent]: skipped by --skip-lc.")
        else:
            stage_t0 = time.time()
            lc_attack, lc_metrics = run_label_consistent(cfg, trainset, testset, benign_model, logger)
            stage_elapsed_seconds["label_consistent"] = time.time() - stage_t0
            all_metrics["stages"]["label_consistent"] = lc_metrics

        stage_t0 = time.time()
        refine_badnets = run_refine_for_attack(
            cfg,
            "BadNets",
            badnets_attack.get_model(),
            trainset,
            testset,
            badnets_attack.poisoned_test_dataset,
            logger,
        )
        stage_elapsed_seconds["refine_badnets"] = time.time() - stage_t0
        all_metrics["stages"]["refine_badnets"] = refine_badnets

        stage_t0 = time.time()
        refine_blended = run_refine_for_attack(
            cfg,
            "Blended",
            blended_attack.get_model(),
            trainset,
            testset,
            blended_attack.poisoned_test_dataset,
            logger,
        )
        stage_elapsed_seconds["refine_blended"] = time.time() - stage_t0
        all_metrics["stages"]["refine_blended"] = refine_blended

        if cfg.skip_lc:
            logger.log("Stage[REFINE/LabelConsistent]: skipped because LabelConsistent is skipped.")
        else:
            stage_t0 = time.time()
            refine_lc = run_refine_for_attack(
                cfg,
                "LabelConsistent",
                lc_attack.get_model(),
                trainset,
                testset,
                lc_attack.poisoned_test_dataset,
                logger,
            )
            stage_elapsed_seconds["refine_label_consistent"] = time.time() - stage_t0
            all_metrics["stages"]["refine_label_consistent"] = refine_lc

        total_elapsed = time.time() - pipeline_start
        all_metrics["timing"] = {
            "total_elapsed_seconds": total_elapsed,
            "total_elapsed_hms": format_duration(total_elapsed),
            "stage_elapsed_seconds": stage_elapsed_seconds,
            "stage_elapsed_hms": {k: format_duration(v) for k, v in stage_elapsed_seconds.items()},
        }

        summary_path = output_dir / "metrics_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(to_builtin(all_metrics), f, indent=2)

        report_path = output_dir / "comparison_report.md"
        report_path.write_text(build_comparison_markdown(to_builtin(all_metrics)), encoding="utf-8")

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
