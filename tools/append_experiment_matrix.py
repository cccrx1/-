"""Incrementally append experiment summaries into a long-lived comparison matrix.

Usage:
  python tools/append_experiment_matrix.py
  python tools/append_experiment_matrix.py --rebuild
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


COLUMNS = [
    "created_at_utc",
    "run_path",
    "metrics_json",
    "output_root",
    "seed",
    "method_tag",
    "defense_variant",
    "only_attack",
    "refine_epochs",
    "ssl_weight",
    "pdb_weight",
    "pdb_batch_ratio",
    "pdb_apply_inference_trigger",
    "pdb_warmup_ratio",
    "ssl_warmup_ratio",
    "aux_loss_cap_ratio",
    "pretrained_attack_model_path",
    "attack_ba",
    "attack_asr",
    "refine_ba",
    "refine_asr",
    "ba_drop",
    "asr_drop",
    "total_time_hms",
    "refine_time_hms",
    "json_mtime_utc",
]


METHOD_TAG_MAP = {
    "": "U",
    "refine": "R",
    "refine_ssl": "RS",
    "refine_pdb": "RB",
    "refine_pdb_ssl": "RSB",
    "refine_cg": "RCG",
}


ATTACK_LABEL_MAP = {
    "badnets": "BadNets",
    "blended": "Blended",
    "label_consistent": "LabelConsistent",
    "": "Unknown",
}


def _to_float(value) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _fmt_float(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def _method_tag(defense_variant: str) -> str:
    dv = (defense_variant or "").strip()
    if dv in METHOD_TAG_MAP:
        return METHOD_TAG_MAP[dv]
    return dv.upper() if dv else "U"


def _is_collapse(refine_ba_text: str, collapse_ba_threshold: float) -> bool:
    refine_ba = _to_float(refine_ba_text)
    if refine_ba is None:
        return False
    return refine_ba < collapse_ba_threshold


def _pick_attack_stage(stages: Dict, only_attack: str) -> Tuple[Optional[str], Optional[str]]:
    preferred = {
        "badnets": "badnets",
        "blended": "blended",
        "label_consistent": "label_consistent",
    }.get(only_attack, "")

    if preferred and preferred in stages:
        refine_key = f"refine_{preferred}"
        return preferred, refine_key if refine_key in stages else None

    attack_keys = [k for k in stages.keys() if not k.startswith("refine_")]
    if not attack_keys:
        return None, None

    attack_key = sorted(attack_keys)[0]
    refine_key = f"refine_{attack_key}"
    return attack_key, refine_key if refine_key in stages else None


def _extract_row(repo_root: Path, metrics_path: Path) -> Optional[Dict[str, str]]:
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    runtime = payload.get("runtime_config", {}) or {}
    stages = payload.get("stages", {}) or {}
    timing = payload.get("timing", {}) or {}

    only_attack = str(runtime.get("only_attack", ""))
    attack_key, refine_key = _pick_attack_stage(stages, only_attack)

    attack_ba = None
    attack_asr = None
    refine_ba = None
    refine_asr = None

    if attack_key and isinstance(stages.get(attack_key), dict):
        attack_ba = _to_float(stages[attack_key].get("clean", {}).get("top1_acc"))
        attack_asr = _to_float(stages[attack_key].get("poisoned", {}).get("top1_acc"))

    if refine_key and isinstance(stages.get(refine_key), dict):
        refine_ba = _to_float(stages[refine_key].get("clean_after_refine", {}).get("top1_acc"))
        refine_asr = _to_float(stages[refine_key].get("poisoned_after_refine", {}).get("top1_acc"))

    ba_drop = attack_ba - refine_ba if attack_ba is not None and refine_ba is not None else None
    asr_drop = attack_asr - refine_asr if attack_asr is not None and refine_asr is not None else None

    stage_hms = timing.get("stage_elapsed_hms", {}) or {}
    refine_time_hms = ""
    if refine_key and isinstance(stage_hms, dict):
        refine_time_hms = str(stage_hms.get(refine_key, ""))

    rel_metrics = metrics_path.relative_to(repo_root).as_posix()
    run_path = metrics_path.parent.relative_to(repo_root).as_posix()
    mtime = datetime.fromtimestamp(metrics_path.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    row = {
        "created_at_utc": now_utc,
        "run_path": run_path,
        "metrics_json": rel_metrics,
        "output_root": str(runtime.get("output_root", "")),
        "seed": str(runtime.get("seed", "")),
        "defense_variant": str(runtime.get("defense_variant", "")),
        "method_tag": _method_tag(str(runtime.get("defense_variant", ""))),
        "only_attack": only_attack,
        "refine_epochs": str(runtime.get("refine_epochs", "")),
        "ssl_weight": str(runtime.get("ssl_weight", "")),
        "pdb_weight": str(runtime.get("pdb_weight", "")),
        "pdb_batch_ratio": str(runtime.get("pdb_batch_ratio", "")),
        "pdb_apply_inference_trigger": str(runtime.get("pdb_apply_inference_trigger", "")),
        "pdb_warmup_ratio": str(runtime.get("pdb_warmup_ratio", "")),
        "ssl_warmup_ratio": str(runtime.get("ssl_warmup_ratio", "")),
        "aux_loss_cap_ratio": str(runtime.get("aux_loss_cap_ratio", "")),
        "pretrained_attack_model_path": str(runtime.get("pretrained_attack_model_path", "")),
        "attack_ba": _fmt_float(attack_ba),
        "attack_asr": _fmt_float(attack_asr),
        "refine_ba": _fmt_float(refine_ba),
        "refine_asr": _fmt_float(refine_asr),
        "ba_drop": _fmt_float(ba_drop),
        "asr_drop": _fmt_float(asr_drop),
        "total_time_hms": str(timing.get("total_elapsed_hms", "")),
        "refine_time_hms": refine_time_hms,
        "json_mtime_utc": mtime,
    }
    return row


def _read_existing_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _normalize_existing_row(row: Dict[str, str]) -> Dict[str, str]:
    normalized = {k: row.get(k, "") for k in COLUMNS}
    normalized["method_tag"] = row.get("method_tag", "") or _method_tag(row.get("defense_variant", ""))
    return normalized


def _write_rows(csv_path: Path, rows: List[Dict[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in COLUMNS})


def _append_rows(csv_path: Path, new_rows: List[Dict[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if not file_exists:
            writer.writeheader()
        for row in new_rows:
            writer.writerow({k: row.get(k, "") for k in COLUMNS})


def _build_markdown(rows: List[Dict[str, str]]) -> str:
    lines = []
    lines.append("# Experiment Matrix")
    lines.append("")
    lines.append(f"- Rows: {len(rows)}")
    lines.append(f"- Updated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    lines.append("")
    lines.append("| Method | Attack | Seed | Epochs | SSL | PDB | Ratio | BA(before) | ASR(before) | BA(after) | ASR(after) | BA drop | ASR drop | Run |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")

    sorted_rows = sorted(
        rows,
        key=lambda r: (
            r.get("method_tag", ""),
            r.get("only_attack", ""),
            r.get("seed", ""),
            r.get("refine_epochs", ""),
            r.get("run_path", ""),
        ),
    )
    for r in sorted_rows:
        attack_key = (r.get("only_attack", "") or "").strip()
        attack_label = ATTACK_LABEL_MAP.get(attack_key, attack_key or "Unknown")
        lines.append(
            "| {method} | {attack} | {seed} | {epochs} | {ssl} | {pdb} | {ratio} | {aba} | {aasr} | {rba} | {rasr} | {bdrop} | {adrop} | {run} |".format(
                method=r.get("method_tag", ""),
                attack=attack_label,
                seed=r.get("seed", ""),
                epochs=r.get("refine_epochs", ""),
                ssl=r.get("ssl_weight", ""),
                pdb=r.get("pdb_weight", ""),
                ratio=r.get("pdb_batch_ratio", ""),
                aba=r.get("attack_ba", ""),
                aasr=r.get("attack_asr", ""),
                rba=r.get("refine_ba", ""),
                rasr=r.get("refine_asr", ""),
                bdrop=r.get("ba_drop", ""),
                adrop=r.get("asr_drop", ""),
                run=r.get("run_path", ""),
            )
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Append experiment metrics into a persistent matrix CSV.")
    parser.add_argument("--repo-root", type=str, default=".")
    parser.add_argument("--experiments-dir", type=str, default="experiments")
    parser.add_argument("--out-csv", type=str, default="experiment_matrix.csv")
    parser.add_argument("--out-md", type=str, default="experiment_matrix.md")
    parser.add_argument("--collapse-ba-threshold", type=float, default=0.30,
                        help="Exclude collapsed runs when refine BA is below this value")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild matrix from scratch instead of append-only")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    experiments_dir = (repo_root / args.experiments_dir).resolve()
    out_csv = (repo_root / args.out_csv).resolve()
    out_md = (repo_root / args.out_md).resolve()

    metrics_files = sorted(experiments_dir.rglob("metrics_summary.json"))

    scanned_total = 0
    collapsed_filtered = 0
    scanned_rows = []
    for p in metrics_files:
        row = _extract_row(repo_root, p)
        if row is None:
            continue
        scanned_total += 1
        if _is_collapse(row.get("refine_ba", ""), args.collapse_ba_threshold):
            collapsed_filtered += 1
            continue
        scanned_rows.append(row)

    if args.rebuild:
        final_rows = sorted(scanned_rows, key=lambda r: r.get("run_path", ""))
        _write_rows(out_csv, final_rows)
        appended = len(final_rows)
    else:
        existing_rows = [_normalize_existing_row(r) for r in _read_existing_rows(out_csv)]
        existing_rows = [
            r for r in existing_rows
            if not _is_collapse(r.get("refine_ba", ""), args.collapse_ba_threshold)
        ]

        existing_keys = {r.get("run_path", "") for r in existing_rows}
        row_by_key = {r.get("run_path", ""): r for r in existing_rows if r.get("run_path", "")}
        for r in scanned_rows:
            key = r.get("run_path", "")
            if key:
                row_by_key[key] = r

        final_rows = sorted(row_by_key.values(), key=lambda r: r.get("run_path", ""))
        appended = sum(1 for r in scanned_rows if r.get("run_path", "") not in existing_keys)
        _write_rows(out_csv, final_rows)

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_build_markdown(final_rows), encoding="utf-8")

    print(f"scanned_total={scanned_total}")
    print(f"collapsed_filtered={collapsed_filtered}")
    print(f"scanned_kept={len(scanned_rows)}")
    print(f"appended={appended}")
    print(f"csv={out_csv}")
    print(f"md={out_md}")


if __name__ == "__main__":
    main()
