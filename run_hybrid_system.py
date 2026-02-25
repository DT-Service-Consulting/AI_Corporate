import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from typing import Dict, List

from tqdm import tqdm

from core_logic import evaluate_reasoning, evaluate_single_extraction
from intelligent_router import IntelligentRouter


CONTROL_GROUP = [
    {
        "id": "control_extract_1",
        "type": "extraction",
        "doc_name": "Control_Consulting_Agreement.txt",
        "input": "The Consultant shall receive a retainer of $10,000 per month.",
        "target": "retainer of $10,000 per month",
        "is_discovery": False,
    },
    {
        "id": "control_reason_1",
        "type": "reasoning",
        "doc_name": "Control_Hearsay_Test",
        "input": {
            "rule": "Hearsay is an out-of-court statement offered to prove the truth of the matter asserted.",
            "facts": "Alice testifies that she saw the light turn red.",
            "question": "Is this hearsay?",
            "options": ["Yes", "No"],
        },
        "target": "No",
        "is_discovery": False,
    },
]


MODEL_COST_PER_1K_TOKENS = {
    "Llama-4-Maverick-17B-128E-Instruct-FP8": 0.20,
    "Llama-3.3-70B-Instruct": 0.80,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hybrid router evaluator with ablation modes.")
    p.add_argument("--output", default="hybrid_system_results.json")
    p.add_argument("--split-manifest", default="split_manifest.json")
    p.add_argument("--split-filter", default="test", choices=["all", "train", "val", "test"])
    p.add_argument("--router-variant", default="full", choices=["full", "no_keyword", "no_length", "no_reasoning_override"])
    p.add_argument("--rate-limit-s", type=float, default=0.5)
    return p.parse_args()


def load_json_list(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return []
        data = json.loads(content)
    return data if isinstance(data, list) else []


def load_data() -> List[Dict]:
    data = CONTROL_GROUP + load_json_list("data/real_tenders.json") + load_json_list("data/legalbench_data.json")
    for idx, item in enumerate(data):
        if not item.get("id"):
            raw = json.dumps(
                {
                    "type": item.get("type"),
                    "doc_name": item.get("doc_name", ""),
                    "target": item.get("target", ""),
                    "input": item.get("input", ""),
                },
                sort_keys=True,
                ensure_ascii=True,
            )
            item["id"] = f"task_{idx}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:12]}"
        else:
            item["id"] = str(item["id"])
        item["doc_name"] = item.get("doc_name", "unknown_doc")
        item["is_discovery"] = bool(item.get("is_discovery", False))
    return data


def load_manifest(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("splits", {})


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(round(len(str(text)) / 4.0)))


def estimate_cost_usd(model_name: str, prompt_text: str, output_text: str) -> float:
    rate = MODEL_COST_PER_1K_TOKENS.get(model_name, 0.0)
    tok = estimate_tokens(prompt_text) + estimate_tokens(output_text)
    return round((tok / 1000.0) * rate, 6)


def route_with_variant(router: IntelligentRouter, variant: str, user_query: str, document_text: str):
    q = user_query.lower()
    if variant == "no_reasoning_override":
        if any(word in q for word in router.extraction_keywords):
            return router.EXTRACTION_MODEL, "extraction"
        if len(document_text) > 15000:
            return router.EXTRACTION_MODEL, "extraction"
        return router.REASONING_MODEL, "reasoning"

    if variant == "no_keyword":
        if len(document_text) > 15000:
            return router.EXTRACTION_MODEL, "extraction"
        return router.REASONING_MODEL, "reasoning"

    if variant == "no_length":
        if any(word in q for word in router.reasoning_keywords):
            return router.REASONING_MODEL, "reasoning"
        if any(word in q for word in router.extraction_keywords):
            return router.EXTRACTION_MODEL, "extraction"
        return router.REASONING_MODEL, "reasoning"

    return router.route(user_query, document_text)


def run_hybrid_test(args: argparse.Namespace) -> None:
    router = IntelligentRouter()
    dataset = load_data()
    split_manifest = load_manifest(args.split_manifest)
    if args.split_filter != "all" and not split_manifest:
        print(f"Warning: split manifest '{args.split_manifest}' not found. Falling back to split_filter=all.")
        args.split_filter = "all"
    if not dataset:
        raise RuntimeError("No data loaded. Run ingest scripts first.")

    results = []
    routing_stats = {"extraction": 0, "reasoning": 0}
    print(f"Starting hybrid test on {len(dataset)} tasks (split={args.split_filter}, variant={args.router_variant})")

    for item in tqdm(dataset):
        split = split_manifest.get(item["id"], "unknown")
        if args.split_filter != "all" and split != args.split_filter:
            continue

        try:
            if item["type"] == "extraction":
                user_query = item.get("query_intent", f"Extract the clause related to {item.get('target', '')}")
            else:
                user_query = item["input"]["question"]

            selected_model, intent = route_with_variant(router, args.router_variant, user_query, str(item.get("input", "")))
            routing_stats[intent] += 1

            start = time.perf_counter()
            if item["type"] == "extraction":
                query = item.get("query_intent", item.get("target", ""))
                res = evaluate_single_extraction(item["input"], query, selected_model)
                score = res["metrics"].get("jaccard", res["metrics"].get("f2", 0.0))
            else:
                res = evaluate_reasoning(
                    item["input"].get("rule", ""),
                    item["input"]["facts"],
                    {
                        "question": item["input"]["question"],
                        "options": item["input"].get("options", []),
                        "answer": item["target"],
                    },
                    selected_model,
                )
                score = res["metrics"]["accuracy"]
            latency_ms = round((time.perf_counter() - start) * 1000.0, 3)

            output_text = str(res.get("model_output", ""))
            prompt_text = str(item.get("input", ""))
            results.append(
                {
                    "id": item["id"],
                    "split": split,
                    "router_variant": args.router_variant,
                    "router_intent": intent,
                    "model_selected": selected_model,
                    "task_type": item["type"],
                    "doc_name": item.get("doc_name", "unknown_doc"),
                    "query": user_query,
                    "score": round(float(score), 6),
                    "ground_truth": item.get("target", ""),
                    "output": output_text,
                    "latency_ms": latency_ms,
                    "est_prompt_tokens": estimate_tokens(prompt_text),
                    "est_output_tokens": estimate_tokens(output_text),
                    "est_cost_usd": estimate_cost_usd(selected_model, prompt_text, output_text),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            if args.rate_limit_s > 0:
                time.sleep(args.rate_limit_s)
        except Exception as exc:
            print(f"Error processing item {item.get('id', 'unknown')}: {exc}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Hybrid test complete.")
    print(f"Saved: {args.output}")
    print(f"Routing decisions: {routing_stats}")
    if results:
        avg_score = sum(r["score"] for r in results) / len(results)
        avg_cost = sum(r["est_cost_usd"] for r in results) / len(results)
        avg_latency = sum(r["latency_ms"] for r in results) / len(results)
        print(f"Average score: {avg_score:.4f}")
        print(f"Average est cost (USD): {avg_cost:.6f}")
        print(f"Average latency (ms): {avg_latency:.2f}")


if __name__ == "__main__":
    run_hybrid_test(parse_args())
