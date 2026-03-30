import argparse
import difflib
import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from core_logic import MODEL_COST_PER_1K_TOKENS, MODELS_TO_TEST, evaluate_reasoning, evaluate_single_extraction


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


ERROR_OUTPUT_RE = re.compile(r"(azure error|error:\s*no client|internal_server_error|backend returned unexpected response)", re.IGNORECASE)


@dataclass
class SplitConfig:
    test_size: float
    val_size: float
    seed: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_json_list(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return []
        data = json.loads(content)
    return data if isinstance(data, list) else []


def stable_task_id(item: Dict, idx: int) -> str:
    if item.get("id"):
        return str(item["id"])
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
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"task_{idx}_{digest}"


def load_dataset() -> List[Dict]:
    tender_data = load_json_list("data/real_tenders_extraction_qa.json")
    if not tender_data:
        tender_data = load_json_list("data/real_tenders.json")
    legalbench_data = load_json_list("data/legalbench_data.json")
    full = CONTROL_GROUP + tender_data + legalbench_data

    normalized = []
    for idx, item in enumerate(full):
        x = dict(item)
        x["id"] = stable_task_id(x, idx)
        x["doc_name"] = x.get("doc_name", "unknown_doc")
        x["is_discovery"] = bool(x.get("is_discovery", False))
        normalized.append(x)
    return normalized


def build_split_manifest(dataset: List[Dict], cfg: SplitConfig) -> Dict[str, str]:
    ids = [item["id"] for item in dataset]
    task_types = [str(item.get("type", "unknown")) for item in dataset]

    idx_all = np.arange(len(ids))
    train_val_idx, test_idx = train_test_split(
        idx_all,
        test_size=cfg.test_size,
        random_state=cfg.seed,
        stratify=task_types,
    )
    val_ratio_adj = cfg.val_size / max(1e-8, 1.0 - cfg.test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio_adj,
        random_state=cfg.seed,
        stratify=np.array(task_types)[train_val_idx],
    )

    manifest = {}
    for i in train_idx:
        manifest[ids[int(i)]] = "train"
    for i in val_idx:
        manifest[ids[int(i)]] = "val"
    for i in test_idx:
        manifest[ids[int(i)]] = "test"
    return manifest


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(round(len(text) / 4.0)))


def estimate_cost_usd(model_name: str, prompt_text: str, output_text: str) -> float:
    rate = MODEL_COST_PER_1K_TOKENS.get(model_name, 0.0)
    tok = estimate_tokens(prompt_text) + estimate_tokens(output_text)
    return round((tok / 1000.0) * rate, 6)


def normalize_reasoning_score(res: Dict, ground_truth: str) -> Dict:
    parsed_answer = str(res.get("parsed_answer", "")).lower().strip()
    truth = str(ground_truth).lower().strip()
    model_output = str(res.get("model_output", "")).lower()

    is_correct = res["metrics"]["accuracy"] == 1.0
    soft_sim = difflib.SequenceMatcher(None, parsed_answer, truth).ratio()
    mentions_gt = 1.0 if truth and truth in model_output else 0.0

    score_main = max(soft_sim, 0.5 * mentions_gt)
    if is_correct:
        score_main = max(score_main, 1.0)

    metrics = {
        "accuracy": res["metrics"]["accuracy"],
        "soft_score": round(score_main, 3),
        "parsed_answer": parsed_answer,
    }
    return {"score": round(score_main, 6), "metrics": metrics}


def evaluate_item(item: Dict, model: str, extraction_chunking: bool, canonical_not_found: bool) -> Dict:
    if item["type"] == "extraction":
        query = item.get("query_intent", item.get("target", ""))
        target = item.get("target", query)
        res = evaluate_single_extraction(
            item.get("input", ""),
            query,
            model,
            expected_text=target,
            enable_chunking=extraction_chunking,
            canonicalize_missing_enabled=canonical_not_found,
        )
        score = res["metrics"].get("jaccard", res["metrics"].get("f2", 0.0))
        return {
            "score": round(float(score), 6),
            "metrics": res["metrics"],
            "model_output": res.get("model_output", ""),
            "query": query,
        }

    if item["type"] == "reasoning":
        answer_obj = {
            "question": item["input"]["question"],
            "options": item["input"].get("options", []),
            "answer": item["target"],
        }
        res = evaluate_reasoning(item["input"].get("rule", ""), item["input"]["facts"], answer_obj, model)
        norm = normalize_reasoning_score(res, item["target"])
        return {
            "score": norm["score"],
            "metrics": norm["metrics"],
            "model_output": res.get("model_output", ""),
            "query": item["input"]["question"],
        }

    raise ValueError(f"Unknown task type: {item.get('type')}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproducible baseline runner for legal routing experiments.")
    p.add_argument("--output", default="results.json")
    p.add_argument("--manifest-output", default="split_manifest.json")
    p.add_argument("--run-meta-output", default="run_metadata.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--rate-limit-s", type=float, default=0.5)
    p.add_argument("--max-retries", type=int, default=3, help="Retries per model-task call on transient API failure.")
    p.add_argument("--retry-backoff-s", type=float, default=1.0, help="Exponential backoff base seconds for retries.")
    p.add_argument(
        "--extraction-chunking",
        choices=["on", "off"],
        default="on",
        help="Enable chunk-retrieve extraction for long documents.",
    )
    p.add_argument(
        "--canonical-not-found",
        choices=["on", "off"],
        default="on",
        help="Enable NOT_FOUND canonicalization in extraction scoring.",
    )
    return p.parse_args()


def is_failed_model_output(output_text: str) -> bool:
    if output_text is None:
        return True
    return bool(ERROR_OUTPUT_RE.search(str(output_text)))


def run() -> None:
    args = parse_args()
    set_seed(args.seed)
    dataset = load_dataset()
    split_cfg = SplitConfig(test_size=args.test_size, val_size=args.val_size, seed=args.seed)
    split_manifest = build_split_manifest(dataset, split_cfg)

    results = []
    print(f"Loaded tasks: {len(dataset)}")
    print(f"Models: {len(MODELS_TO_TEST)}")

    for model in MODELS_TO_TEST:
        print(f"Evaluating model: {model}")
        for item in tqdm(dataset, desc=f"{model}"):
            attempt = 0
            out = None
            latency_ms = 0.0
            eval_status = "failed"
            while attempt <= args.max_retries:
                start = time.perf_counter()
                out = evaluate_item(
                    item,
                    model,
                    extraction_chunking=(args.extraction_chunking == "on"),
                    canonical_not_found=(args.canonical_not_found == "on"),
                )
                latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
                if not is_failed_model_output(out.get("model_output", "")):
                    eval_status = "ok"
                    break
                if attempt < args.max_retries:
                    sleep_s = args.retry_backoff_s * (2 ** attempt)
                    time.sleep(max(0.0, sleep_s))
                attempt += 1
            if out is None:
                out = {"score": 0.0, "metrics": {}, "model_output": "Error: evaluation did not produce output", "query": ""}

            prompt_text = str(item.get("input", ""))
            model_output = str(out.get("model_output", ""))
            output_for_cost = model_output if eval_status == "ok" else ""
            record = {
                "id": item["id"],
                "split": split_manifest.get(item["id"], "unknown"),
                "seed": args.seed,
                "task_type": item["type"],
                "is_discovery": item.get("is_discovery", False),
                "model_name": model,
                "doc_name": item.get("doc_name", "unknown_doc"),
                "ground_truth": item.get("target", ""),
                "input": item.get("input", ""),
                "query": out["query"],
                "full_output": model_output,
                "score": out["score"] if eval_status == "ok" else 0.0,
                "metrics": out["metrics"],
                "eval_status": eval_status,
                "attempts_used": int(attempt + 1),
                "extraction_chunking": args.extraction_chunking,
                "canonical_not_found": args.canonical_not_found,
                "latency_ms": latency_ms,
                "est_prompt_tokens": estimate_tokens(prompt_text),
                "est_output_tokens": estimate_tokens(output_for_cost),
                "est_cost_usd": estimate_cost_usd(model, prompt_text, output_for_cost),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            results.append(record)
            if args.rate_limit_s > 0:
                time.sleep(args.rate_limit_s)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(args.manifest_output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": args.seed,
                "test_size": args.test_size,
                "val_size": args.val_size,
                "splits": split_manifest,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    run_meta = {
        "seed": args.seed,
        "task_count": len(dataset),
        "model_count": len(MODELS_TO_TEST),
        "output_file": args.output,
        "split_manifest_file": args.manifest_output,
        "started_for_submission": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(args.run_meta_output, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False)

    print(f"Saved: {args.output}")
    print(f"Saved: {args.manifest_output}")
    print(f"Saved: {args.run_meta_output}")


if __name__ == "__main__":
    run()
