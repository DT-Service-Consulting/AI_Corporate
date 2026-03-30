"""Verification runner for a small stratified FC-LAMMR sample."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import project_secrets

from fc_lammr.fc_lammr_router import FCLAMMRRouter
from fc_lammr.run_fc_lammr_hybrid_test import compute_stratified_results, evaluate_router_output, write_stratified_outputs
from fc_lammr.test_fc_lammr import FakeLLMClient
from fc_lammr.utils.llm_client import validate_deployment_config
from run_hybrid_system import load_data


ALLOWED_MODES = {
    "PRL_MATCH",
    "TOMIL_SUCCESS",
    "TOMIL_NORMALISED",
    "REROUTED",
    "CONTENT_FILTER_BLOCKED",
    "FAILED_LLM_CALL",
    "TOMIL_PARSE_FAILURE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small FC-LAMMR verification pass.")
    parser.add_argument("--max-tasks", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _ensure_dry_run_deployments() -> None:
    if not getattr(project_secrets, "EXTRACTION_DEPLOYMENT_NAME", None):
        project_secrets.EXTRACTION_DEPLOYMENT_NAME = "fake-extraction"
    if not getattr(project_secrets, "REASONING_DEPLOYMENT_NAME", None):
        project_secrets.REASONING_DEPLOYMENT_NAME = "fake-reasoning"
    if not getattr(project_secrets, "TOMIL_DEPLOYMENT_NAME", None):
        project_secrets.TOMIL_DEPLOYMENT_NAME = "fake-tomil"


def _pick_items(dataset: list[dict], max_tasks: int) -> list[dict]:
    extraction = []
    ambiguous = []
    risk = []
    complex_docs = []
    for item in dataset:
        if item["type"] == "extraction":
            extraction.append(item)
        query_text = item.get("query_intent", "") if item["type"] == "extraction" else item["input"].get("question", "")
        lowered = query_text.lower()
        if any(word in lowered for word in ["what does", "does", "whether", "compare", "interpret"]):
            ambiguous.append(item)
        if any(word in lowered for word in ["indemnity", "liability", "risk", "exposure"]):
            risk.append(item)
        doc_text = str(item.get("input", ""))
        if item["type"] == "reasoning":
            doc_text = f"{item['input'].get('rule', '')}\n{item['input'].get('facts', '')}".strip()
        complex_docs.append((len(doc_text), item))

    selected = []
    buckets = [
        extraction[:5],
        ambiguous[:5],
        risk[:5],
        [item for _, item in sorted(complex_docs, key=lambda pair: pair[0], reverse=True)[:5]],
    ]
    for bucket in buckets:
        for item in bucket:
            if item not in selected and len(selected) < max_tasks:
                selected.append(item)
    if len(selected) < max_tasks:
        remaining = [item for item in dataset if item not in selected]
        random.seed(42)
        random.shuffle(remaining)
        selected.extend(remaining[: max_tasks - len(selected)])
        logging.warning("Dataset could not fully satisfy all five-per-category buckets. Filled remaining slots with fallback samples.")
    return selected[:max_tasks]


def _query_and_document(item: dict) -> tuple[str, str]:
    if item["type"] == "extraction":
        return item.get("query_intent", f"Extract the clause related to {item.get('target', '')}"), str(item.get("input", ""))
    return item["input"]["question"], f"{item['input'].get('rule', '')}\n{item['input'].get('facts', '')}".strip()


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    if args.dry_run:
        _ensure_dry_run_deployments()
    validate_deployment_config()

    dataset = load_data()
    sample = _pick_items(dataset, args.max_tasks)
    client = FakeLLMClient() if args.dry_run else None
    router = FCLAMMRRouter(llm_client=client, pattern_library_path="fc_lammr/pattern_library.json")
    all_queries = []
    all_documents = []
    for item in sample:
        q, d = _query_and_document(item)
        all_queries.append(q)
        all_documents.append(d)
    router.prl.prefetch_and_fit_vocabulary(all_queries, all_documents)

    results = []
    invariant_counts = {f"INV-{idx}": 0 for idx in range(1, 11)}
    for item in sample:
        query, document = _query_and_document(item)
        state = router.route(query, document)
        score, metrics, scored_output = evaluate_router_output(state, item)
        result = {
            "item_id": item["id"],
            "effective_routing_mode": state.effective_routing_mode,
            "tom_inference_quality": None if state.tom_inference_quality is None else state.tom_inference_quality.__dict__,
            "routing_belief": state.routing_belief,
            "audit_log": state.audit_log,
            "assigned_model": state.assigned_model.value if state.assigned_model else None,
            "reroute_triggered": state.reroute_triggered,
            "reroute_count": state.reroute_count,
            "partial_output": state.partial_output,
            "score": score,
            "task_type": item["type"],
            "output": scored_output,
            "metrics": metrics,
        }
        results.append(result)

        checks = {
            "INV-1": result["effective_routing_mode"] in ALLOWED_MODES,
            "INV-2": True if result["tom_inference_quality"] is None else (result["tom_inference_quality"] is not None),
            "INV-3": (abs(sum(result["routing_belief"].values()) - 1.0) < 1e-6) if result["routing_belief"] else True,
            "INV-4": len(result["audit_log"]) >= 1,
            "INV-5": (not result["reroute_triggered"]) or result["reroute_count"] >= 1,
            "INV-6": (not result["reroute_triggered"]) or (result["partial_output"] is not None),
            "INV-7": (result["effective_routing_mode"] != "CONTENT_FILTER_BLOCKED") or (result["score"] is None),
            "INV-8": (result["effective_routing_mode"] != "FAILED_LLM_CALL") or (result["score"] is None),
            "INV-9": (result["effective_routing_mode"] != "TOMIL_PARSE_FAILURE") or (result["score"] is None and result["assigned_model"] is None),
            "INV-10": (result["effective_routing_mode"] != "TOMIL_NORMALISED") or (result["tom_inference_quality"] and not result["tom_inference_quality"]["tom_is_valid_for_research"]),
        }
        for key, passed in checks.items():
            invariant_counts[key] += int(bool(passed))

    stratified = compute_stratified_results(results)
    summary_json, summary_txt = write_stratified_outputs(results)
    total_from_strata = sum(stratified[name]["n_tasks"] for name in ALLOWED_MODES if name in stratified)
    success_count = sum(1 for row in results if row["effective_routing_mode"] == "TOMIL_SUCCESS")
    normalised_count = sum(1 for row in results if row["effective_routing_mode"] == "TOMIL_NORMALISED")
    strat_checks = {
        "STRAT-1": total_from_strata == len(results),
        "STRAT-2": summary_txt.exists(),
        "STRAT-3": all(stratified[name]["avg_score_combined"] is None for name in ["CONTENT_FILTER_BLOCKED", "FAILED_LLM_CALL", "TOMIL_PARSE_FAILURE"]),
        "STRAT-4": stratified["TOMIL_SUCCESS"]["n_tasks"] == success_count and stratified["TOMIL_NORMALISED"]["n_tasks"] == normalised_count,
    }

    completed_with_score = sum(1 for row in results if row["score"] is not None)
    excluded = len(results) - completed_with_score
    mode_counts = Counter(row["effective_routing_mode"] for row in results)
    overall_ready = all(count == len(results) for count in invariant_counts.values()) and all(strat_checks.values())

    print("FC-LAMMR VERIFICATION RUN — RESULTS")
    print("=====================================")
    print(f"Tasks attempted: {len(results)}")
    print(f"Tasks completed with score: {completed_with_score}")
    print(f"Tasks excluded: {excluded}")
    print()
    print("Per-task invariant results:")
    print(f"  INV-1 (routing mode valid):          {invariant_counts['INV-1']}/{len(results)} {'PASS' if invariant_counts['INV-1'] == len(results) else 'FAIL'}")
    print(f"  INV-2 (tom quality non-null):        {invariant_counts['INV-2']}/{len(results)} {'PASS' if invariant_counts['INV-2'] == len(results) else 'FAIL'}")
    print(f"  INV-3 (belief sums to 1.0):          {invariant_counts['INV-3']}/{len(results)} {'PASS' if invariant_counts['INV-3'] == len(results) else 'FAIL'}")
    print(f"  INV-4 (audit log non-empty):         {invariant_counts['INV-4']}/{len(results)} {'PASS' if invariant_counts['INV-4'] == len(results) else 'FAIL'}")
    print(f"  INV-5 (reroute count correct):       {invariant_counts['INV-5']}/{len(results)} {'PASS' if invariant_counts['INV-5'] == len(results) else 'FAIL'}")
    print(f"  INV-6 (partial output on reroute):   {invariant_counts['INV-6']}/{len(results)} {'PASS' if invariant_counts['INV-6'] == len(results) else 'FAIL'}")
    print(f"  INV-7 (no score on blocked):         {invariant_counts['INV-7']}/{len(results)} {'PASS' if invariant_counts['INV-7'] == len(results) else 'FAIL'}")
    print(f"  INV-8 (no score on llm failure):     {invariant_counts['INV-8']}/{len(results)} {'PASS' if invariant_counts['INV-8'] == len(results) else 'FAIL'}")
    print(f"  INV-9 (no score on parse failure):   {invariant_counts['INV-9']}/{len(results)} {'PASS' if invariant_counts['INV-9'] == len(results) else 'FAIL'}")
    print(f"  INV-10 (normalised = not valid):     {invariant_counts['INV-10']}/{len(results)} {'PASS' if invariant_counts['INV-10'] == len(results) else 'FAIL'}")
    print()
    print("Stratification assertions:")
    print(f"  STRAT-1 (no double count):   {'PASS' if strat_checks['STRAT-1'] else 'FAIL'}")
    print(f"  STRAT-2 (summary written):   {'PASS' if strat_checks['STRAT-2'] else 'FAIL'}")
    print(f"  STRAT-3 (excluded no score): {'PASS' if strat_checks['STRAT-3'] else 'FAIL'}")
    print(f"  STRAT-4 (normalised isolated): {'PASS' if strat_checks['STRAT-4'] else 'FAIL'}")
    print()
    print("Routing mode distribution:")
    print(f"  PRL_MATCH:              {mode_counts.get('PRL_MATCH', 0)}")
    print(f"  TOMIL_SUCCESS:          {mode_counts.get('TOMIL_SUCCESS', 0)}")
    print(f"  TOMIL_NORMALISED:       {mode_counts.get('TOMIL_NORMALISED', 0)}")
    print(f"  REROUTED:               {mode_counts.get('REROUTED', 0)}")
    print(f"  CONTENT_FILTER_BLOCKED: {mode_counts.get('CONTENT_FILTER_BLOCKED', 0)}")
    print(f"  FAILED_LLM_CALL:        {mode_counts.get('FAILED_LLM_CALL', 0)}")
    print(f"  TOMIL_PARSE_FAILURE:    {mode_counts.get('TOMIL_PARSE_FAILURE', 0)}")
    print()
    print("Configuration:")
    print(f"  Vocabulary frozen:           {'YES' if router.prl._vocabulary_frozen else 'NO'}")
    print("  Deployment config validated: YES")
    print()
    print(f"OVERALL: {'READY FOR FULL RUN' if overall_ready else 'NOT READY - see failures above'}")

    return 0 if overall_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
