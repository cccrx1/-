"""Matrix-driven suite runner.

Responsibilities:
1) Resolve cases from test/test_matrix.json
2) Launch each case through core.pipeline.run_case
3) Aggregate stage metrics into summary json/md
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from core.pipeline.matrix_utils import collect_case_metrics, get_cases, load_matrix, resolve_cases


def parse_args():
    parser = argparse.ArgumentParser(description="Run one or more matrix-defined test cases.")
    parser.add_argument("--cases", type=str, default="all", help="Comma-separated case names or 'all'.")
    parser.add_argument("--matrix", type=str, default="test/test_matrix.json", help="Matrix json path.")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable used to launch case runner.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue running remaining cases if one fails.")
    parser.add_argument("--list-cases", action="store_true", help="List all available cases and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only, do not execute.")
    parser.add_argument("--summary-dir", type=str, default="./experiments/suite/summary", help="Directory to store aggregated summaries.")
    args, passthrough = parser.parse_known_args()
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    return args, passthrough


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
    root = Path(__file__).resolve().parents[2]
    matrix_path = (root / args.matrix).resolve()
    matrix = load_matrix(matrix_path)
    matrix_cases = get_cases(matrix)

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

    for case in cases:
        cmd = [
            args.python,
            "-m",
            "core.pipeline.run_case",
            "--case",
            case,
            "--matrix",
            str(Path(args.matrix).as_posix()),
            "--run-group",
            "suite",
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
            effective_case_cfg = dict(matrix_cases[case])
            effective_case_cfg["output_root"] = f"./experiments/suite/{case}/runs"
            metrics = collect_case_metrics(root, effective_case_cfg)

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
