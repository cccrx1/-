"""Unified runner for matrix-driven cases in test/test_matrix.json.

Examples:
python run_test_suite.py
python run_test_suite.py --list-cases
python run_test_suite.py --cases badnets_refine,blended_refine_ssl_50
python run_test_suite.py --dry-run -- --batch-size 256
python run_test_suite.py -- --cuda-selected-devices 0
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run one or more matrix-defined test cases.")
    parser.add_argument("--cases", type=str, default="all", help="Comma-separated case names or 'all'.")
    parser.add_argument("--matrix", type=str, default="test/test_matrix.json", help="Matrix json path.")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable used to launch case runner.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue running remaining cases if one fails.")
    parser.add_argument("--list-cases", action="store_true", help="List all available cases and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only, do not execute.")
    parser.add_argument("--summary-dir", type=str, default="./experiments/test/summary", help="Directory to store aggregated summaries.")
    args, passthrough = parser.parse_known_args()
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    return args, passthrough


def load_matrix(matrix_path: Path):
    with matrix_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    cases = data.get("cases", {})
    if not isinstance(cases, dict) or not cases:
        raise ValueError(f"Invalid or empty cases in matrix: {matrix_path}")
    return cases


def resolve_cases(cases_text: str, known_cases):
    if cases_text == "all":
        return list(known_cases)

    names = [item.strip() for item in cases_text.split(",") if item.strip()]
    invalid = [name for name in names if name not in known_cases]
    if invalid:
        raise ValueError(f"Unknown cases: {invalid}. Supported: {list(known_cases)}")
    return names


def stage_keys_from_attack(only_attack: str):
    mapping = {
        "badnets": ("badnets", "refine_badnets"),
        "blended": ("blended", "refine_blended"),
        "label_consistent": ("label_consistent", "refine_label_consistent"),
    }
    return mapping.get(only_attack, (None, None))


def safe_get_top1(stage_dict, metric_key):
    if not isinstance(stage_dict, dict):
        return None
    metric = stage_dict.get(metric_key)
    if not isinstance(metric, dict):
        return None
    val = metric.get("top1_acc")
    return float(val) if isinstance(val, (int, float)) else None


def collect_case_metrics(root: Path, case_cfg):
    output_root = case_cfg.get("output_root")
    if not isinstance(output_root, str):
        return {"error": "missing output_root in matrix"}

    summary_path = (root / output_root / "metrics_summary.json").resolve()
    if not summary_path.exists():
        return {"error": f"metrics not found: {summary_path}"}

    with summary_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    stages = data.get("stages", {}) if isinstance(data, dict) else {}
    attack_key, refine_key = stage_keys_from_attack(str(case_cfg.get("only_attack", "")))

    attack_stage = stages.get(attack_key, {}) if attack_key else {}
    refine_stage = stages.get(refine_key, {}) if refine_key else {}

    clean_before = safe_get_top1(attack_stage, "clean")
    poisoned_before = safe_get_top1(attack_stage, "poisoned")
    clean_after = safe_get_top1(refine_stage, "clean_after_refine")
    poisoned_after = safe_get_top1(refine_stage, "poisoned_after_refine")

    return {
        "summary_path": str(summary_path),
        "clean_before": clean_before,
        "poisoned_before": poisoned_before,
        "clean_after": clean_after,
        "poisoned_after": poisoned_after,
    }


def write_aggregate_summary(summary_dir: Path, run_report):
    summary_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    json_path = summary_dir / f"suite_summary_{ts}.json"
    latest_json = summary_dir / "suite_summary_latest.json"
    md_path = summary_dir / f"suite_summary_{ts}.md"
    latest_md = summary_dir / "suite_summary_latest.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(run_report, f, indent=2)
    with latest_json.open("w", encoding="utf-8") as f:
        json.dump(run_report, f, indent=2)

    lines = []
    lines.append("# Test Suite Summary")
    lines.append("")
    lines.append(f"- started_at: {run_report.get('started_at')}")
    lines.append(f"- total_cases: {run_report.get('total_cases')}")
    lines.append(f"- failed_cases: {len(run_report.get('failed', []))}")
    lines.append("")
    lines.append("| case | status | clean_before | poisoned_before | clean_after | poisoned_after |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    for item in run_report.get("results", []):
        metrics = item.get("metrics", {}) if isinstance(item, dict) else {}
        lines.append(
            "| {case} | {status} | {cb} | {pb} | {ca} | {pa} |".format(
                case=item.get("case"),
                status=item.get("status"),
                cb=("{:.4f}".format(metrics["clean_before"]) if isinstance(metrics.get("clean_before"), float) else "-"),
                pb=("{:.4f}".format(metrics["poisoned_before"]) if isinstance(metrics.get("poisoned_before"), float) else "-"),
                ca=("{:.4f}".format(metrics["clean_after"]) if isinstance(metrics.get("clean_after"), float) else "-"),
                pa=("{:.4f}".format(metrics["poisoned_after"]) if isinstance(metrics.get("poisoned_after"), float) else "-"),
            )
        )

    md_text = "\n".join(lines) + "\n"
    md_path.write_text(md_text, encoding="utf-8")
    latest_md.write_text(md_text, encoding="utf-8")

    return json_path, md_path


def main():
    args, passthrough = parse_args()
    root = Path(__file__).resolve().parent
    matrix_path = (root / args.matrix).resolve()
    matrix_cases = load_matrix(matrix_path)

    if args.list_cases:
        print("Available cases:")
        for name in matrix_cases.keys():
            print("-", name)
        return

    cases = resolve_cases(args.cases, matrix_cases.keys())
    print(f"Selected cases: {cases}")
    print(f"Matrix file: {matrix_path}")
    if passthrough:
        print(f"Forward args: {passthrough}")

    run_report = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "matrix": str(matrix_path),
        "cases": cases,
        "total_cases": len(cases),
        "results": [],
        "failed": [],
        "dry_run": bool(args.dry_run),
    }

    case_runner = root / "test" / "run_case.py"

    for case in cases:
        cmd = [
            args.python,
            str(case_runner),
            "--case",
            case,
            "--matrix",
            str(Path(args.matrix).as_posix()),
        ]
        if args.dry_run:
            cmd.append("--dry-run")
        cmd.extend(passthrough)

        print("=" * 80)
        print(f"Running case: {case}")
        print("Command:", " ".join(cmd))

        t0 = time.time()
        result = subprocess.run(cmd, cwd=str(root), check=False)
        elapsed = time.time() - t0

        status = "ok" if result.returncode == 0 else "failed"
        metrics = {}
        if status == "ok" and not args.dry_run:
            metrics = collect_case_metrics(root, matrix_cases[case])

        run_report["results"].append(
            {
                "case": case,
                "status": status,
                "exit_code": int(result.returncode),
                "elapsed_seconds": elapsed,
                "metrics": metrics,
            }
        )

        if result.returncode != 0:
            run_report["failed"].append({"case": case, "exit_code": int(result.returncode)})
            print(f"Case failed: {case}, exit_code={result.returncode}")
            if not args.continue_on_error:
                break
        else:
            print(f"Case finished: {case}, elapsed={elapsed:.1f}s")

    summary_dir = (root / args.summary_dir).resolve()
    summary_json_path, summary_md_path = write_aggregate_summary(summary_dir, run_report)

    print("=" * 80)
    print(f"Suite summary json: {summary_json_path}")
    print(f"Suite summary md: {summary_md_path}")

    if run_report["failed"]:
        print(f"Completed with failures: {run_report['failed']}")
        raise SystemExit(1)

    print("All selected cases finished successfully.")


if __name__ == "__main__":
    main()
