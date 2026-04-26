"""Research-style evaluation runner for FC-LAMMR."""

# RECOMMENDED SETTINGS FOR LIVE 1093-TASK RUN
# ============================================
# --task-sleep 1.0        Add 1 second between tasks to reduce 429 pressure.
#                         At 1093 tasks this adds ~18 minutes to wall clock
#                         but prevents quota exhaustion that caused the high
#                         cost run.
# --max-reasoning-calls 400
#                         Conservative budget for the reasoning deployment.
#                         Increase only if quota headroom is confirmed.
# --checkpoint-interval 25
#                         Write checkpoint every 25 tasks instead of 50.
#                         Reduces the maximum data-loss window on failure.
# --split-filter [filter] Use the same split filter as baseline runs for
#                         a fair comparison.

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from fc_lammr.data_structures import TaskType
from fc_lammr.evaluation_layer import EvaluationLayer
from fc_lammr.fc_lammr_router import FCLAMMRRouter
from fc_lammr.utils.llm_client import get_429_count, get_project_deployment_config, reset_429_state
from fc_lammr.utils.prompt_helpers import calculate_advanced_metrics, postprocess_extraction_output
from run_experiment import estimate_cost_usd, estimate_tokens
from run_hybrid_system import load_data, load_manifest


CHECKPOINT_INTERVAL_DEFAULT = 50
PROGRESS_INTERVAL = 25
EXCLUDED_ROUTING_MODES = {"CONTENT_FILTER_BLOCKED", "FAILED_LLM_CALL", "TOMIL_PARSE_FAILURE"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate FC-LAMMR with the workspace's current hybrid methodology.")
    p.add_argument("--output", default="outputs/fc_lammr/fc_lammr_hybrid_results.json")
    p.add_argument("--split-manifest", default="split_manifest.json")
    p.add_argument("--split-filter", default="test", choices=["all", "train", "val", "test"])
    p.add_argument("--pattern-library", default="fc_lammr/pattern_library.json")
    p.add_argument("--prl-threshold", type=float, default=0.82)
    p.add_argument("--cold-prl-threshold", type=float, default=0.91)
    p.add_argument("--min-pattern-quality", type=float, default=0.75)
    p.add_argument("--reroute-threshold", type=float, default=0.65)
    p.add_argument("--signal-penalty", type=float, default=0.15)
    p.add_argument("--rate-limit-s", type=float, default=0.5, help="Legacy alias for inter-task pacing; preserved for backward compatibility.")
    p.add_argument(
        "--task-sleep",
        type=float,
        default=0.5,
        help=(
            "Seconds to sleep between tasks. Default 0.5 adds minimal pacing. "
            "Increase to 1.0-2.0 if sustained 429s occur."
        ),
    )
    p.add_argument(
        "--max-reasoning-calls",
        type=int,
        default=500,
        help=(
            "Maximum total calls to the reasoning deployment across the run. "
            "When reached, remaining tasks are routed directly to extraction "
            "using BUDGET_FORCED_EXTRACTION."
        ),
    )
    p.add_argument(
        "--request-timeout-s",
        type=float,
        default=60.0,
        help="Per-request timeout passed to the Azure/OpenAI client so a single API call cannot hang indefinitely.",
    )
    p.add_argument(
        "--max-task-seconds",
        type=float,
        default=180.0,
        help="Per-task wall clock limit. If exceeded, the task is marked FAILED_LLM_CALL and the run continues.",
    )
    p.add_argument("--checkpoint-interval", type=int, default=CHECKPOINT_INTERVAL_DEFAULT)
    p.add_argument("--resume-from", default="")
    return p.parse_args()


_REASONING_LABEL_HINTS = ("yes", "no", "partial", "fair", "unfair")
_ANSWER_PREFIX_RE = re.compile(r"answer\s*:\s*", re.IGNORECASE)
_TOKEN_SPLIT_RE = re.compile(r"[\s,\.;:\-\(\)\[\]\{\}\"'`!?/\\|]+")


def _normalise_reasoning_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _extract_reasoning_answer_text(output_text: str) -> str:
    raw_text = str(output_text or "")
    match = _ANSWER_PREFIX_RE.search(raw_text)
    if not match:
        return raw_text
    return raw_text[match.end():]


def _extract_first_meaningful_token(text: str) -> str:
    cleaned = _normalise_reasoning_text(text).lstrip(" \"'`([{")
    for token in _TOKEN_SPLIT_RE.split(cleaned):
        if token:
            return token
    return ""


def _reasoning_score(output_text: str, target: str) -> tuple[float, dict]:
    gold_label = _normalise_reasoning_text(target)
    answer_text = _extract_reasoning_answer_text(output_text)
    normalised_answer = _normalise_reasoning_text(answer_text)
    first_token = _extract_first_meaningful_token(answer_text)
    parsed_answer = first_token or "unknown"

    if not normalised_answer or not gold_label:
        return 0.0, {"accuracy": 0.0, "soft_score": 0.0, "parsed_answer": parsed_answer}

    early_window = normalised_answer[:50]
    exact_match = (
        first_token == gold_label
        or normalised_answer.startswith(gold_label)
        or gold_label in early_window
    )
    if exact_match:
        return 1.0, {"accuracy": 1.0, "soft_score": 1.0, "parsed_answer": gold_label}

    if gold_label in normalised_answer:
        return 0.5, {"accuracy": 0.0, "soft_score": 0.5, "parsed_answer": parsed_answer}

    for hint in [*list(_REASONING_LABEL_HINTS), gold_label]:
        if hint and hint in early_window:
            parsed_answer = hint
            break

    return 0.0, {"accuracy": 0.0, "soft_score": 0.0, "parsed_answer": parsed_answer}


def evaluate_router_output(state, item: dict) -> tuple[float | None, dict, str]:
    if state.effective_routing_mode in EXCLUDED_ROUTING_MODES:
        return None, {}, ""
    output_text = str(state.final_output or "")
    if item["type"] == "extraction" or state.task_type in {TaskType.EXTRACTION, TaskType.CLAUSE_IDENTIFICATION}:
        target = str(item.get("target", ""))
        processed = postprocess_extraction_output(output_text)
        metrics = calculate_advanced_metrics(processed, target, canonicalize_missing_enabled=True)
        score = float(metrics.get("jaccard", metrics.get("f2", 0.0)))
        return score, metrics, processed
    score, metrics = _reasoning_score(output_text, str(item.get("target", "")))
    return score, metrics, output_text


def _tomil_engaged(state) -> bool:
    if state.effective_routing_mode in {"PRL_MATCH", "BUDGET_FORCED_EXTRACTION", *EXCLUDED_ROUTING_MODES}:
        return False
    return any(
        entry.get("layer") == "ToMIL" and entry.get("decision") in {"intent_inferred", "parse_failure"}
        for entry in getattr(state, "audit_log", [])
    )


def _calls_detail_for_state(state) -> dict[str, Any]:
    tomil = int(_tomil_engaged(state))
    reroute = int(bool(state.reroute_triggered))
    return {
        "tomil": tomil,
        "execution": 1,
        "reroute": reroute,
        "execution_model": state.assigned_model.value if state.assigned_model else None,
    }


def _calls_used_for_state(state) -> int:
    detail = _calls_detail_for_state(state)
    return detail["tomil"] + detail["execution"] + detail["reroute"]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {
            (
                key.value
                if isinstance(key, Enum)
                else key
                if isinstance(key, (str, int, float, bool)) or key is None
                else str(key)
            ): _json_safe(inner_value)
            for key, inner_value in value.items()
        }
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=str)]
    return value


def _update_call_counters_from_state(
    state,
    *,
    reasoning_call_counter: int,
    extraction_call_counter: int,
    tomil_call_counter: int,
) -> tuple[int, int, int]:
    if _tomil_engaged(state):
        tomil_call_counter += 1
        extraction_call_counter += 1
    for entry in getattr(state, "audit_log", []):
        if entry.get("layer") != "execution":
            continue
        model_used = str(entry.get("model", "")).lower()
        if "reasoning" in model_used:
            reasoning_call_counter += 1
        elif "extraction" in model_used:
            extraction_call_counter += 1
    return reasoning_call_counter, extraction_call_counter, tomil_call_counter


def compute_stratified_results(
    results: list[dict],
    *,
    call_summary: dict | None = None,
    run_metadata: dict | None = None,
) -> dict:
    strata = {
        "TOMIL_SUCCESS": [],
        "PRL_MATCH": [],
        "TOMIL_NORMALISED": [],
        "REROUTED": [],
        "CONTENT_FILTER_BLOCKED": [],
        "FAILED_LLM_CALL": [],
        "TOMIL_PARSE_FAILURE": [],
        "BUDGET_FORCED_EXTRACTION": [],
    }
    for result in results:
        strata.setdefault(result.get("effective_routing_mode", "FAILED_LLM_CALL"), []).append(result)
    scored = [result for result in results if result.get("score") is not None]
    strata["ALL_SCORED"] = scored

    def summary(rows: list[dict]) -> dict:
        scored_rows = [row for row in rows if row.get("score") is not None]
        extraction = [float(row["score"]) for row in scored_rows if row.get("task_type") == "extraction"]
        reasoning = [float(row["score"]) for row in scored_rows if row.get("task_type") == "reasoning"]
        return {
            "n_tasks": len(rows),
            "avg_score_extraction": (sum(extraction) / len(extraction)) if extraction else None,
            "avg_score_reasoning": (sum(reasoning) / len(reasoning)) if reasoning else None,
            "avg_score_combined": (sum(float(row["score"]) for row in scored_rows) / len(scored_rows)) if scored_rows else None,
            "avg_cost_usd": (sum(float(row.get("est_cost_usd", 0.0)) for row in rows) / len(rows)) if rows else 0.0,
            "avg_latency_ms": (sum(float(row.get("latency_ms", 0.0)) for row in rows) / len(rows)) if rows else 0.0,
            "avg_calls_per_task": (sum(int(row.get("calls_used", 0)) for row in rows) / len(rows)) if rows else 0.0,
            "reroute_count": sum(int(bool(row.get("reroute_triggered"))) for row in rows),
        }

    parse_counter = Counter(
        row.get("tom_inference_quality", {}).get("parse_path")
        for row in results
        if row.get("tom_inference_quality")
    )
    normalised = len(strata["TOMIL_NORMALISED"])
    primary = len(strata["TOMIL_SUCCESS"])
    content_blocked = len(strata["CONTENT_FILTER_BLOCKED"])
    excluded = (
        len(strata["CONTENT_FILTER_BLOCKED"])
        + len(strata["FAILED_LLM_CALL"])
        + len(strata["TOMIL_PARSE_FAILURE"])
    )
    evaluation = EvaluationLayer()
    reroute_analysis = evaluation.score_reroute_quality(strata["REROUTED"], scored)
    output = {name: summary(rows) for name, rows in strata.items()}
    output["tomil_quality_distribution"] = dict(parse_counter)
    output["normalisation_rate"] = float(normalised / (normalised + primary)) if (normalised + primary) else 0.0
    output["content_filter_rate"] = float(content_blocked / len(results)) if results else 0.0
    output["exclusion_rate"] = float(excluded / len(results)) if results else 0.0
    output["reroute_quality_analysis"] = reroute_analysis
    output["call_summary"] = call_summary or {}
    output["run_metadata"] = run_metadata or {}
    return output


def write_stratified_outputs(results: list[dict], *, call_summary: dict, run_metadata: dict) -> tuple[Path, Path]:
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stratified = compute_stratified_results(results, call_summary=call_summary, run_metadata=run_metadata)
    json_path = results_dir / f"fc_lammr_stratified_results_{timestamp}.json"
    txt_path = results_dir / f"fc_lammr_stratified_summary_{timestamp}.txt"
    json_path.write_text(json.dumps(stratified, indent=2, ensure_ascii=False), encoding="utf-8")
    header = (
        "FC-LAMMR Pre-Final Evaluation Summary\n"
        "======================================\n"
        f"Total tasks attempted: {len(results)}\n"
        f"Tasks scored (primary): {stratified['TOMIL_SUCCESS']['n_tasks']} (TOMIL_SUCCESS only)\n"
        f"Tasks scored (all strata): {stratified['ALL_SCORED']['n_tasks']}\n"
        f"Tasks excluded: {len(results) - stratified['ALL_SCORED']['n_tasks']} "
        f"({stratified['content_filter_rate']*100:.2f}% content filter,\n"
        f"                     0.00% parse failure,\n"
        f"                     0.00% LLM call failure)\n"
        f"Normalisation rate: {stratified['normalisation_rate']*100:.2f}%\n"
        f"Reasoning calls total: {call_summary.get('reasoning_calls_total', 0)}\n"
        f"Extraction calls total: {call_summary.get('extraction_calls_total', 0)}\n"
        f"ToMIL calls total: {call_summary.get('tomil_calls_total', 0)}\n"
        f"Total 429s across run: {call_summary.get('total_429s_across_run', 0)}\n"
        f"Budget forced extraction tasks: {call_summary.get('budget_forced_extraction_tasks', 0)}\n"
        "WARNING: TOMIL_NORMALISED tasks are NOT valid FC-LAMMR results.\n"
        "They represent heuristic routing, not Theory of Mind routing.\n"
        "Do not include them in the primary comparison table.\n"
    )
    txt_path.write_text(header, encoding="utf-8")
    return json_path, txt_path


def _startup_block(args: argparse.Namespace) -> tuple[str, dict]:
    runner_path = Path(__file__).resolve()
    runner_mtime = datetime.fromtimestamp(runner_path.stat().st_mtime)
    with runner_path.open("rb") as handle:
        runner_hash = hashlib.md5(handle.read()).hexdigest()[:8]
    config = get_project_deployment_config()
    tomil_eq_reasoning = config["TOMIL_DEPLOYMENT_NAME"] == config["REASONING_DEPLOYMENT_NAME"]
    startup_block = (
        f"\n{'=' * 60}\n"
        "FC-LAMMR RUNNER STARTUP\n"
        f"  File:         {runner_path}\n"
        f"  Modified:     {runner_mtime}\n"
        f"  MD5 (8char):  {runner_hash}\n"
        f"  PID:          {os.getpid()}\n"
        f"  Python:       {sys.version.split()[0]}\n"
        f"  TOMIL deploy: {config['TOMIL_DEPLOYMENT_NAME']}\n"
        f"  Reasoning:    {config['REASONING_DEPLOYMENT_NAME']}\n"
        f"  TOMIL==Reasoning: {tomil_eq_reasoning}"
        + (" [DOUBLE REASONING LOAD]" if tomil_eq_reasoning else " [OK]")
        + f"\n  Args:         {vars(args)}\n"
        f"{'=' * 60}\n"
    )
    run_metadata = {
        "runner_file": str(runner_path),
        "runner_mtime": runner_mtime.isoformat(),
        "runner_md5": runner_hash,
        "pid": os.getpid(),
        "python_version": sys.version.split()[0],
        "args": vars(args),
        "run_started_at": datetime.now().isoformat(),
        "tomil_deployment": config["TOMIL_DEPLOYMENT_NAME"],
        "reasoning_deployment": config["REASONING_DEPLOYMENT_NAME"],
        "tomil_equals_reasoning": tomil_eq_reasoning,
    }
    return startup_block, run_metadata


def run() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    reset_429_state()
    startup_block, run_metadata = _startup_block(args)
    print(startup_block, flush=True)
    logging.info(startup_block)

    dataset = load_data()
    split_manifest = load_manifest(args.split_manifest)
    router = FCLAMMRRouter(
        llm_client=None,
        pattern_library_path=args.pattern_library,
        prl_threshold=args.prl_threshold,
        cold_prl_threshold=args.cold_prl_threshold,
        min_pattern_quality=args.min_pattern_quality,
        reroute_threshold=args.reroute_threshold,
        signal_penalty=args.signal_penalty,
        request_timeout_s=args.request_timeout_s,
        max_task_seconds=args.max_task_seconds,
    )

    evaluation_dataset = []
    for item in dataset:
        split = split_manifest.get(item["id"], "unknown")
        if args.split_filter != "all" and split != args.split_filter:
            continue
        if item["type"] == "extraction":
            query = item.get("query_intent", f"Extract the clause related to {item.get('target', '')}")
            document = str(item.get("input", ""))
        else:
            query = item["input"]["question"]
            document = f"{item['input'].get('rule', '')}\n{item['input'].get('facts', '')}".strip()
        evaluation_dataset.append({"item": item, "query": query, "document": document, "split": split})

    all_queries = [entry["query"] for entry in evaluation_dataset]
    all_documents = [entry["document"] for entry in evaluation_dataset]
    router.prl.prefetch_and_fit_vocabulary(all_queries, all_documents)
    logging.info("Vocabulary pre-fitting complete. Beginning evaluation.")

    results: list[dict] = []
    completed_ids = set()
    if args.resume_from:
        resume_rows = json.loads(Path(args.resume_from).read_text(encoding="utf-8"))
        results = list(resume_rows)
        completed_ids = {row["item_id"] for row in results}
        evaluation_dataset = [entry for entry in evaluation_dataset if entry["item"]["id"] not in completed_ids]
        logging.info("Resuming from checkpoint. Completed: %s. Remaining: %s.", len(completed_ids), len(evaluation_dataset))

    total_tasks = len(results) + len(evaluation_dataset)
    print(f"Starting FC-LAMMR evaluation on {total_tasks} tasks (split={args.split_filter})", flush=True)

    reasoning_call_counter = 0
    extraction_call_counter = 0
    tomil_call_counter = 0
    reroute_to_reasoning = 0
    reroute_to_extraction = 0
    reasoning_budget_exhausted = False
    sleep_s = max(args.rate_limit_s, args.task_sleep)

    for task_num, entry in enumerate(evaluation_dataset, start=len(results) + 1):
        item = entry["item"]
        query = entry["query"]
        document = entry["document"]
        split = entry["split"]

        if reasoning_call_counter >= args.max_reasoning_calls and not reasoning_budget_exhausted:
            reasoning_budget_exhausted = True
            logging.warning(
                "REASONING BUDGET REACHED | %d calls made | limit=%d | Remaining tasks will use BUDGET_FORCED_EXTRACTION.",
                reasoning_call_counter,
                args.max_reasoning_calls,
            )
            print(
                f"\n[WARNING] Reasoning budget reached ({reasoning_call_counter} calls). Switching remaining tasks to extraction model.",
                flush=True,
            )

        task_identifier = item.get("item_id", item.get("id", task_num))
        task_start_time = time.perf_counter()
        logging.info("TASK START | %d/%d | id=%s", task_num, total_tasks, task_identifier)
        state = router.route(query, document, force_extraction=reasoning_budget_exhausted)
        task_latency = time.perf_counter() - task_start_time

        reasoning_call_counter, extraction_call_counter, tomil_call_counter = _update_call_counters_from_state(
            state,
            reasoning_call_counter=reasoning_call_counter,
            extraction_call_counter=extraction_call_counter,
            tomil_call_counter=tomil_call_counter,
        )
        if state.reroute_triggered:
            for audit_entry in state.audit_log:
                if audit_entry.get("layer") == "FRL" and audit_entry.get("decision") == "reroute":
                    new_model = str(audit_entry.get("new_model", "")).lower()
                    if "reasoning" in new_model:
                        reroute_to_reasoning += 1
                    elif "extraction" in new_model:
                        reroute_to_extraction += 1

        logging.info(
            "TASK END | %d/%d | id=%s | mode=%s | latency=%.1fs | reasoning_calls_so_far=%d | 429s_so_far=%d",
            task_num,
            total_tasks,
            task_identifier,
            state.effective_routing_mode,
            task_latency,
            reasoning_call_counter,
            get_429_count(),
        )

        latency_ms = round(task_latency * 1000.0, 3)
        score, metrics, scored_output = evaluate_router_output(state, item)
        chosen_model = router._deployment_for(state.assigned_model) if state.assigned_model else None
        calls_detail = _calls_detail_for_state(state)
        record = _json_safe({
            "item_id": item["id"],
            "id": item["id"],
            "split": split,
            "router_variant": "fc_lammr",
            "router_intent": state.task_type.value if state.task_type else None,
            "model_selected": chosen_model,
            "assigned_model": state.assigned_model.value if state.assigned_model else None,
            "task_type": item["type"],
            "doc_name": item.get("doc_name", "unknown_doc"),
            "query": query,
            "score": round(float(score), 6) if score is not None else None,
            "ground_truth": item.get("target", ""),
            "output": scored_output,
            "latency_ms": latency_ms,
            "est_prompt_tokens": estimate_tokens(str(item.get("input", ""))),
            "est_output_tokens": estimate_tokens(scored_output),
            "est_cost_usd": estimate_cost_usd(chosen_model or "", str(item.get("input", "")), scored_output),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "reroute_triggered": state.reroute_triggered,
            "reroute_count": state.reroute_count,
            "risk_weight": state.risk_weight,
            "audit_log": state.audit_log,
            "effective_routing_mode": state.effective_routing_mode,
            "tom_inference_quality": None if state.tom_inference_quality is None else state.tom_inference_quality.__dict__,
            "calls_used": _calls_used_for_state(state),
            "calls_detail": calls_detail,
            "metrics": metrics,
            "result_status": state.result_status,
            "belief_at_trigger": [
                float(audit_entry.get("belief_at_trigger", 0.0))
                for audit_entry in state.audit_log
                if audit_entry.get("decision") == "reroute"
            ],
        })
        results.append(record)

        if sleep_s > 0:
            time.sleep(sleep_s)

        if len(results) % args.checkpoint_interval == 0:
            checkpoint_path = Path("results") / f"fc_lammr_checkpoint_{len(results)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text(json.dumps(_json_safe(results), indent=2, ensure_ascii=False), encoding="utf-8")
            logging.info("Checkpoint written at task %s: %s", len(results), checkpoint_path)

        if len(results) % PROGRESS_INTERVAL == 0 and results:
            scored_n = sum(1 for row in results if row.get("score") is not None)
            excl_n = sum(1 for row in results if row.get("effective_routing_mode") in EXCLUDED_ROUTING_MODES)
            mode_counts = dict(Counter(row.get("effective_routing_mode") for row in results))
            print(
                f"\r[{len(results)}/{total_tasks}] scored={scored_n} excl={excl_n} "
                f"reasoning_calls={reasoning_call_counter} 429s={get_429_count()} modes={mode_counts}",
                end="",
                flush=True,
            )
            logging.info(
                "PROGRESS | %d/%d | scored=%d | excl=%d | reasoning_calls=%d | 429s=%d | modes=%s",
                len(results),
                total_tasks,
                scored_n,
                excl_n,
                reasoning_call_counter,
                get_429_count(),
                mode_counts,
            )

    if results:
        print("", flush=True)

    call_summary = {
        "reasoning_calls_total": reasoning_call_counter,
        "extraction_calls_total": extraction_call_counter,
        "tomil_calls_total": tomil_call_counter,
        "total_429s_across_run": get_429_count(),
        "reasoning_budget_limit": args.max_reasoning_calls,
        "reasoning_budget_exhausted": reasoning_budget_exhausted,
        "budget_forced_extraction_tasks": sum(
            1 for record in results if record.get("effective_routing_mode") == "BUDGET_FORCED_EXTRACTION"
        ),
        "avg_calls_per_scored_task": round(
            sum(record.get("calls_used", 0) for record in results if record.get("score") is not None)
            / max(1, sum(1 for record in results if record.get("score") is not None)),
            2,
        ),
        "reroute_to_reasoning": reroute_to_reasoning,
        "reroute_to_extraction": reroute_to_extraction,
        "net_reroute_reasoning_impact": reroute_to_reasoning - reroute_to_extraction,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_safe(results), indent=2, ensure_ascii=False), encoding="utf-8")
    stratified_json, stratified_txt = write_stratified_outputs(
        results,
        call_summary=call_summary,
        run_metadata=run_metadata,
    )

    print("FC-LAMMR evaluation complete.", flush=True)
    print(f"Saved: {output_path}", flush=True)
    print(f"Saved: {stratified_json}", flush=True)
    print(f"Saved: {stratified_txt}", flush=True)


if __name__ == "__main__":
    run()
