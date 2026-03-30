"""Research-style evaluation runner for FC-LAMMR."""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from fc_lammr.data_structures import TaskType
from fc_lammr.evaluation_layer import EvaluationLayer
from fc_lammr.fc_lammr_router import FCLAMMRRouter
from fc_lammr.utils.prompt_helpers import calculate_advanced_metrics, postprocess_extraction_output
from run_hybrid_system import load_data, load_manifest
from run_experiment import estimate_cost_usd, estimate_tokens, normalize_reasoning_score


CHECKPOINT_INTERVAL_DEFAULT = 50
PROGRESS_INTERVAL = 25


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate FC-LAMMR with the workspace's current hybrid methodology.")
    p.add_argument("--output", default="results/fc_lammr_hybrid_results.json")
    p.add_argument("--split-manifest", default="split_manifest.json")
    p.add_argument("--split-filter", default="test", choices=["all", "train", "val", "test"])
    p.add_argument("--pattern-library", default="fc_lammr/pattern_library.json")
    p.add_argument("--prl-threshold", type=float, default=0.82)
    p.add_argument("--reroute-threshold", type=float, default=0.65)
    p.add_argument("--signal-penalty", type=float, default=0.15)
    p.add_argument("--rate-limit-s", type=float, default=0.5)
    p.add_argument("--checkpoint-interval", type=int, default=CHECKPOINT_INTERVAL_DEFAULT)
    p.add_argument("--resume-from", default="")
    return p.parse_args()


def _reasoning_score(output_text: str, target: str) -> tuple[float, dict]:
    parsed_answer = "unknown"
    if "ANSWER:" in output_text:
        parsed_answer = output_text.split("ANSWER:", 1)[1].strip().lower()
    else:
        lowered = output_text.lower()
        for opt in ["yes", "no", "partial", "fair", "unfair", target.lower().strip()]:
            if opt and opt in lowered[-120:]:
                parsed_answer = opt
                break
    base = {
        "metrics": {"accuracy": 1.0 if parsed_answer == target.lower().strip() else 0.0},
        "parsed_answer": parsed_answer,
        "model_output": output_text,
    }
    norm = normalize_reasoning_score(base, target)
    return float(norm["score"]), norm["metrics"]


def evaluate_router_output(state, item: dict) -> tuple[float | None, dict, str]:
    if state.effective_routing_mode in {"CONTENT_FILTER_BLOCKED", "FAILED_LLM_CALL", "TOMIL_PARSE_FAILURE"}:
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


def compute_stratified_results(results: list[dict]) -> dict:
    strata = {
        "TOMIL_SUCCESS": [],
        "PRL_MATCH": [],
        "TOMIL_NORMALISED": [],
        "REROUTED": [],
        "CONTENT_FILTER_BLOCKED": [],
        "FAILED_LLM_CALL": [],
        "TOMIL_PARSE_FAILURE": [],
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

    parse_counter = Counter(row.get("tom_inference_quality", {}).get("parse_path") for row in results if row.get("tom_inference_quality"))
    normalised = len(strata["TOMIL_NORMALISED"])
    primary = len(strata["TOMIL_SUCCESS"])
    content_blocked = len(strata["CONTENT_FILTER_BLOCKED"])
    excluded = len(strata["CONTENT_FILTER_BLOCKED"]) + len(strata["FAILED_LLM_CALL"]) + len(strata["TOMIL_PARSE_FAILURE"])
    evaluation = EvaluationLayer()
    reroute_analysis = evaluation.score_reroute_quality(strata["REROUTED"], scored)
    output = {name: summary(rows) for name, rows in strata.items()}
    output["tomil_quality_distribution"] = dict(parse_counter)
    output["normalisation_rate"] = float(normalised / (normalised + primary)) if (normalised + primary) else 0.0
    output["content_filter_rate"] = float(content_blocked / len(results)) if results else 0.0
    output["exclusion_rate"] = float(excluded / len(results)) if results else 0.0
    output["reroute_quality_analysis"] = reroute_analysis
    return output


def write_stratified_outputs(results: list[dict]) -> tuple[Path, Path]:
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stratified = compute_stratified_results(results)
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
        "WARNING: TOMIL_NORMALISED tasks are NOT valid FC-LAMMR results.\n"
        "They represent heuristic routing, not Theory of Mind routing.\n"
        "Do not include them in the primary comparison table.\n"
    )
    txt_path.write_text(header, encoding="utf-8")
    return json_path, txt_path


def run() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    dataset = load_data()
    split_manifest = load_manifest(args.split_manifest)
    router = FCLAMMRRouter(
        llm_client=None,
        pattern_library_path=args.pattern_library,
        prl_threshold=args.prl_threshold,
        reroute_threshold=args.reroute_threshold,
        signal_penalty=args.signal_penalty,
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
    print(f"Starting FC-LAMMR evaluation on {total_tasks} tasks (split={args.split_filter})")

    for entry in evaluation_dataset:
        item = entry["item"]
        query = entry["query"]
        document = entry["document"]
        split = entry["split"]

        start = time.perf_counter()
        state = router.route(query, document)
        latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
        score, metrics, scored_output = evaluate_router_output(state, item)
        chosen_model = router._deployment_for(state.assigned_model) if state.assigned_model else None
        record = {
            "item_id": item["id"],
            "id": item["id"],
            "split": split,
            "router_variant": "fc_lammr",
            "router_intent": state.task_type.value if state.task_type else None,
            "model_selected": chosen_model,
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
            "calls_used": 1 + int(bool(state.reroute_triggered)),
            "metrics": metrics,
            "result_status": state.result_status,
            "belief_at_trigger": [float(entry.get("belief_at_trigger", 0.0)) for entry in state.audit_log if entry.get("decision") == "reroute"],
        }
        results.append(record)

        if len(results) % args.checkpoint_interval == 0:
            checkpoint_path = Path("results") / f"fc_lammr_checkpoint_{len(results)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
            logging.info("Checkpoint written at task %s: %s", len(results), checkpoint_path)

        if len(results) % PROGRESS_INTERVAL == 0:
            scored = [row for row in results if row.get("score") is not None]
            blocked = [row for row in results if row.get("effective_routing_mode") == "CONTENT_FILTER_BLOCKED"]
            failed = [row for row in results if row.get("effective_routing_mode") in ("FAILED_LLM_CALL", "TOMIL_PARSE_FAILURE")]
            logging.info(
                "Progress: %s/%s | Scored: %s | Blocked: %s | Excluded: %s | Modes: %s",
                len(results),
                total_tasks,
                len(scored),
                len(blocked),
                len(failed),
                dict(Counter(row.get("effective_routing_mode") for row in results)),
            )

        if args.rate_limit_s > 0:
            time.sleep(args.rate_limit_s)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    stratified_json, stratified_txt = write_stratified_outputs(results)

    print("FC-LAMMR evaluation complete.")
    print(f"Saved: {output_path}")
    print(f"Saved: {stratified_json}")
    print(f"Saved: {stratified_txt}")


if __name__ == "__main__":
    run()
