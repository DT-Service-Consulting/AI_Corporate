"""Offline parser-corrected re-scoring for completed FC-LAMMR runs."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any

from fc_lammr.run_fc_lammr_hybrid_test import _reasoning_score


DEFAULT_INPUT = Path("outputs/fc_lammr/fc_lammr_hybrid_results_all.json")
DEFAULT_OUTPUT = Path("outputs/fc_lammr/fc_lammr_hybrid_results_all_parser_corrected.json")
DEFAULT_SUMMARY_JSON = Path("results/fc_lammr/fc_lammr_parser_corrected_summary.json")
DEFAULT_SUMMARY_MD = Path("results/fc_lammr/fc_lammr_parser_corrected_summary.md")
DEFAULT_BASELINE_METHOD_SUMMARY = Path("outputs/research_eval_openai_compare/method_summary.json")
DEFAULT_BASELINE_TASK_BREAKDOWN = Path("outputs/research_eval_openai_compare/task_breakdown.csv")
DEFAULT_BASELINE_METHOD = "hybrid_rule_plus_learning"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline parser-corrected FC-LAMMR re-scoring.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY_JSON))
    parser.add_argument("--summary-md", default=str(DEFAULT_SUMMARY_MD))
    parser.add_argument("--baseline-method-summary", default=str(DEFAULT_BASELINE_METHOD_SUMMARY))
    parser.add_argument("--baseline-task-breakdown", default=str(DEFAULT_BASELINE_TASK_BREAKDOWN))
    parser.add_argument("--baseline-method", default=DEFAULT_BASELINE_METHOD)
    return parser.parse_args(argv)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def summarise_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored_rows = [row for row in rows if row.get("score") is not None]
    extraction_scores = [float(row["score"]) for row in scored_rows if row.get("task_type") == "extraction"]
    reasoning_scores = [float(row["score"]) for row in scored_rows if row.get("task_type") == "reasoning"]
    overall_avg = _mean([float(row["score"]) for row in scored_rows])
    extraction_avg = _mean(extraction_scores)
    reasoning_avg = _mean(reasoning_scores)

    macro_values = [value for value in [extraction_avg, reasoning_avg] if value is not None]
    return {
        "total_tasks": len(rows),
        "scored_tasks": len(scored_rows),
        "skipped_tasks": len(rows) - len(scored_rows),
        "overall_avg": overall_avg,
        "extraction_avg": extraction_avg,
        "reasoning_avg": reasoning_avg,
        "macro_avg": _mean(macro_values),
    }


def rescore_reasoning_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    corrected_rows = copy.deepcopy(rows)
    changed_reasoning_rows = 0
    upgraded_accuracy_0_to_1 = 0

    for row in corrected_rows:
        if row.get("score") is None or row.get("task_type") != "reasoning":
            continue

        old_score = float(row.get("score", 0.0))
        old_metrics = row.get("metrics") or {}
        old_accuracy = float(old_metrics.get("accuracy", 0.0))
        new_score, new_metrics = _reasoning_score(str(row.get("output", "")), str(row.get("ground_truth", "")))

        if abs(new_score - old_score) > 1e-9 or float(new_metrics.get("accuracy", 0.0)) != old_accuracy:
            changed_reasoning_rows += 1
        if old_accuracy == 0.0 and float(new_metrics.get("accuracy", 0.0)) == 1.0:
            upgraded_accuracy_0_to_1 += 1

        row["score"] = round(float(new_score), 6)
        row["metrics"] = new_metrics

    return corrected_rows, {
        "changed_reasoning_rows": changed_reasoning_rows,
        "upgraded_accuracy_0_to_1": upgraded_accuracy_0_to_1,
    }


def load_baseline_comparator(
    method_summary_path: Path,
    task_breakdown_path: Path,
    *,
    method_name: str,
) -> dict[str, Any]:
    comparator = {
        "method_name": method_name,
        "method_summary_path": str(method_summary_path),
        "task_breakdown_path": str(task_breakdown_path),
        "available": False,
    }
    if not method_summary_path.exists() or not task_breakdown_path.exists():
        return comparator

    method_rows = json.loads(method_summary_path.read_text(encoding="utf-8"))
    row = next((item for item in method_rows if item.get("method") == method_name), None)
    if row is None:
        return comparator

    task_rows: list[dict[str, str]]
    with task_breakdown_path.open("r", encoding="utf-8", newline="") as handle:
        task_rows = [item for item in csv.DictReader(handle) if item.get("method") == method_name]

    extraction_row = next((item for item in task_rows if item.get("task_type") == "extraction"), None)
    reasoning_row = next((item for item in task_rows if item.get("task_type") == "reasoning"), None)

    comparator.update(
        {
            "available": True,
            "overall_avg": float(row.get("avg_score_mean", 0.0)),
            "macro_avg": float(row.get("macro_task_score_mean", 0.0)),
            "extraction_avg": float(extraction_row["avg_score_mean"]) if extraction_row else None,
            "reasoning_avg": float(reasoning_row["avg_score_mean"]) if reasoning_row else None,
        }
    )
    return comparator


def build_parser_corrected_summary(
    official_rows: list[dict[str, Any]],
    corrected_rows: list[dict[str, Any]],
    *,
    input_path: Path,
    output_path: Path,
    baseline_method_summary_path: Path,
    baseline_task_breakdown_path: Path,
    baseline_method_name: str,
    change_counts: dict[str, int],
) -> dict[str, Any]:
    official_summary = summarise_results(official_rows)
    corrected_summary = summarise_results(corrected_rows)
    baseline = load_baseline_comparator(
        baseline_method_summary_path,
        baseline_task_breakdown_path,
        method_name=baseline_method_name,
    )

    return {
        "metadata": {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "official_file_preserved": True,
            "raw_outputs_unchanged": True,
            "change_scope": "Post-hoc reasoning answer-format parsing correction only.",
        },
        "official": official_summary,
        "parser_corrected": corrected_summary,
        "delta": {
            "overall_avg": (
                corrected_summary["overall_avg"] - official_summary["overall_avg"]
                if official_summary["overall_avg"] is not None and corrected_summary["overall_avg"] is not None
                else None
            ),
            "reasoning_avg": (
                corrected_summary["reasoning_avg"] - official_summary["reasoning_avg"]
                if official_summary["reasoning_avg"] is not None and corrected_summary["reasoning_avg"] is not None
                else None
            ),
            "macro_avg": (
                corrected_summary["macro_avg"] - official_summary["macro_avg"]
                if official_summary["macro_avg"] is not None and corrected_summary["macro_avg"] is not None
                else None
            ),
        },
        "reasoning_rows_changed": change_counts["changed_reasoning_rows"],
        "accuracy_upgrades_0_to_1": change_counts["upgraded_accuracy_0_to_1"],
        "baseline_comparator": baseline,
        "notes": [
            "This is a sensitivity analysis. The official FC-LAMMR final run remains the canonical raw result.",
            "Extraction metrics were left unchanged.",
            "Comparison against hybrid_rule_plus_learning uses saved summary statistics, not paired raw baseline rows.",
        ],
    }


def render_parser_corrected_markdown(summary: dict[str, Any]) -> str:
    official = summary["official"]
    corrected = summary["parser_corrected"]
    baseline = summary["baseline_comparator"]
    lines = [
        "# FC-LAMMR Parser-Corrected Sensitivity Analysis",
        "",
        "## Official Run Result",
        "",
        f"- Source file: `{summary['metadata']['input_path']}`",
        f"- Overall average: `{official['overall_avg']:.6f}`" if official["overall_avg"] is not None else "- Overall average: `None`",
        f"- Extraction average: `{official['extraction_avg']:.6f}`" if official["extraction_avg"] is not None else "- Extraction average: `None`",
        f"- Reasoning average: `{official['reasoning_avg']:.6f}`" if official["reasoning_avg"] is not None else "- Reasoning average: `None`",
        f"- Macro average: `{official['macro_avg']:.6f}`" if official["macro_avg"] is not None else "- Macro average: `None`",
        f"- Scored tasks: `{official['scored_tasks']}`",
        f"- Skipped tasks: `{official['skipped_tasks']}`",
        "",
        "## Parser-Corrected Sensitivity Analysis",
        "",
        f"- Corrected file: `{summary['metadata']['output_path']}`",
        f"- Overall average: `{corrected['overall_avg']:.6f}`" if corrected["overall_avg"] is not None else "- Overall average: `None`",
        f"- Extraction average: `{corrected['extraction_avg']:.6f}`" if corrected["extraction_avg"] is not None else "- Extraction average: `None`",
        f"- Reasoning average: `{corrected['reasoning_avg']:.6f}`" if corrected["reasoning_avg"] is not None else "- Reasoning average: `None`",
        f"- Macro average: `{corrected['macro_avg']:.6f}`" if corrected["macro_avg"] is not None else "- Macro average: `None`",
        f"- Reasoning rows changed: `{summary['reasoning_rows_changed']}`",
        f"- Rows upgraded from accuracy 0 to 1: `{summary['accuracy_upgrades_0_to_1']}`",
        "",
        "## Methodological Note",
        "",
        "- Raw model outputs were unchanged.",
        "- Only post-hoc answer-format parsing was corrected.",
        "- This does not resolve the extraction regression.",
    ]

    if baseline.get("available"):
        lines.extend(
            [
                "",
                "## Reporting Comparator",
                "",
                f"- Comparator: `{baseline['method_name']}`",
                f"- Overall average: `{baseline['overall_avg']:.6f}`",
                f"- Extraction average: `{baseline['extraction_avg']:.6f}`" if baseline["extraction_avg"] is not None else "- Extraction average: `None`",
                f"- Reasoning average: `{baseline['reasoning_avg']:.6f}`" if baseline["reasoning_avg"] is not None else "- Reasoning average: `None`",
                f"- Macro average: `{baseline['macro_avg']:.6f}`",
                "- Comparison uses saved summary statistics rather than paired raw baseline rows, so paired statistical tests are not available.",
            ]
        )

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_json_path = Path(args.summary_json)
    summary_md_path = Path(args.summary_md)
    baseline_method_summary_path = Path(args.baseline_method_summary)
    baseline_task_breakdown_path = Path(args.baseline_task_breakdown)

    official_rows = json.loads(input_path.read_text(encoding="utf-8"))
    corrected_rows, change_counts = rescore_reasoning_rows(official_rows)
    summary = build_parser_corrected_summary(
        official_rows,
        corrected_rows,
        input_path=input_path,
        output_path=output_path,
        baseline_method_summary_path=baseline_method_summary_path,
        baseline_task_breakdown_path=baseline_task_breakdown_path,
        baseline_method_name=args.baseline_method,
        change_counts=change_counts,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    summary_md_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(json.dumps(corrected_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_md_path.write_text(render_parser_corrected_markdown(summary), encoding="utf-8")

    print(f"Saved corrected results: {output_path}")
    print(f"Saved summary JSON: {summary_json_path}")
    print(f"Saved summary Markdown: {summary_md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
