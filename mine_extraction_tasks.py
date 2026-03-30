import argparse
import hashlib
import json
import os
import re
from typing import Dict, List, Tuple


DEFAULT_INPUT = "data/real_tenders.json"
DEFAULT_OUTPUT = "data/real_tenders_extraction_qa.json"
NOT_FOUND_TOKEN = "NOT_FOUND"


CLAUSE_PATTERNS = [
    {
        "name": "indemnity",
        "keywords": ["indemnity", "indemnify"],
        "queries": [
            "Extract the professional indemnity clause exactly as written.",
            "Find and quote the indemnity insurance line.",
        ],
    },
    {
        "name": "liability",
        "keywords": ["liability"],
        "queries": [
            "Extract the commercial liability clause exactly.",
            "Locate and copy the liability insurance statement.",
        ],
    },
    {
        "name": "insurance",
        "keywords": ["insurance", "workers compensation"],
        "queries": [
            "Extract the workers compensation insurance clause.",
            "Find the insurance requirement text exactly.",
        ],
    },
    {
        "name": "payment",
        "keywords": ["payment", "interim payment", "paid"],
        "queries": [
            "Extract the payment valuation clause.",
            "Find and quote the payment-related sentence.",
        ],
    },
    {
        "name": "variation",
        "keywords": ["variation", "variations", "unit rates", "schedule of rates"],
        "queries": [
            "Extract the clause describing price treatment for contract variations.",
            "Find the variations pricing clause exactly.",
        ],
    },
    {
        "name": "price",
        "keywords": ["price", "prices", "lump sum"],
        "queries": [
            "Extract the clause that defines pricing scope.",
            "Locate and quote the price inclusion clause.",
        ],
    },
    {
        "name": "safety",
        "keywords": ["health and safety", "safety", "employees"],
        "queries": [
            "Extract the contractor health and safety obligation clause.",
            "Find and quote the safety compliance sentence.",
        ],
    },
    {
        "name": "law",
        "keywords": ["required by law", "law", "statutes", "regulations"],
        "queries": [
            "Extract the clause referring to legal/statutory compliance.",
            "Find the sentence that requires compliance with law.",
        ],
    },
    {
        "name": "dispute",
        "keywords": ["dispute", "arbitration", "arbitrator"],
        "queries": [
            "Extract the arbitration/dispute resolution clause.",
            "Locate and quote the dispute settlement sentence.",
        ],
    },
    {
        "name": "completion",
        "keywords": ["completion of the works", "completion"],
        "queries": [
            "Extract the clause mentioning completion of the works.",
            "Find the sentence about arbitration timing relative to completion.",
        ],
    },
]


MISSING_CLAUSE_NEGATIVES = [
    "Extract the force majeure clause addressing pandemics and lockdowns.",
    "Find the clause setting the arbitration seat in London.",
    "Extract the confidentiality clause governing non-disclosure of tender data.",
    "Locate the liquidated damages formula expressed as a daily percentage.",
    "Extract the cyber insurance requirement clause.",
    "Find the clause that allows termination for convenience with 30 days' notice.",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mine clause-level extraction QA tasks from tender/contract text.")
    p.add_argument("--input", default=DEFAULT_INPUT)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--target-min", type=int, default=60)
    p.add_argument("--negatives-per-doc", type=int, default=3)
    p.add_argument("--long-variants-per-doc", type=int, default=3)
    return p.parse_args()


def load_records(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def split_spans(text: str) -> List[str]:
    chunks = re.split(r"(?<=[\.\?!:;])\s+|\n+", text)
    out = []
    seen = set()
    for c in chunks:
        span = normalize_space(c)
        if len(span) < 20 or len(span) > 360:
            continue
        key = span.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(span)
    return out


def make_id(doc_name: str, query: str, target: str, tag: str) -> str:
    raw = f"{doc_name}|{query}|{target}|{tag}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:14]
    return f"extract_{digest}"


def build_positive_tasks(doc_name: str, doc_text: str) -> List[Dict]:
    spans = split_spans(doc_text)
    tasks = []
    for pattern in CLAUSE_PATTERNS:
        hits = []
        for span in spans:
            low = span.lower()
            if any(k in low for k in pattern["keywords"]):
                hits.append(span)
        for i, span in enumerate(hits[:2]):
            query = pattern["queries"][i % len(pattern["queries"])]
            task = {
                "id": make_id(doc_name, query, span, tag=f"pos_{pattern['name']}_{i}"),
                "type": "extraction",
                "doc_name": doc_name,
                "is_discovery": False,
                "difficulty": "standard",
                "query_intent": query,
                "input": doc_text,
                "target": span,
                "clause_family": pattern["name"],
            }
            tasks.append(task)
    return tasks


def build_similar_clause_hard_cases(doc_name: str, doc_text: str) -> List[Dict]:
    spans = split_spans(doc_text)
    indemn = [s for s in spans if ("indemnity insurance" in s.lower() or "indemnity" in s.lower())]
    liability = [s for s in spans if ("liability insurance" in s.lower() or "liability" in s.lower())]
    out = []
    if indemn and liability:
        q1 = "Extract the professional indemnity insurance clause (not the liability insurance clause)."
        q2 = "Extract the commercial liability insurance clause (not the indemnity insurance clause)."
        out.append(
            {
                "id": make_id(doc_name, q1, indemn[0], "hard_sim_1"),
                "type": "extraction",
                "doc_name": doc_name,
                "is_discovery": False,
                "difficulty": "hard_similar_clause",
                "query_intent": q1,
                "input": doc_text,
                "target": indemn[0],
                "clause_family": "insurance_disambiguation",
            }
        )
        out.append(
            {
                "id": make_id(doc_name, q2, liability[0], "hard_sim_2"),
                "type": "extraction",
                "doc_name": doc_name,
                "is_discovery": False,
                "difficulty": "hard_similar_clause",
                "query_intent": q2,
                "input": doc_text,
                "target": liability[0],
                "clause_family": "insurance_disambiguation",
            }
        )
    return out


def build_negative_tasks(doc_name: str, doc_text: str, per_doc: int) -> List[Dict]:
    out = []
    for i, query in enumerate(MISSING_CLAUSE_NEGATIVES[: max(0, per_doc)]):
        out.append(
            {
                "id": make_id(doc_name, query, NOT_FOUND_TOKEN, f"neg_{i}"),
                "type": "extraction",
                "doc_name": doc_name,
                "is_discovery": False,
                "difficulty": "hard_missing_clause",
                "query_intent": query,
                "input": doc_text,
                "target": NOT_FOUND_TOKEN,
                "clause_family": "missing_clause_negative",
            }
        )
    return out


def build_long_variants(tasks: List[Dict], per_doc: int) -> List[Dict]:
    out = []
    added = 0
    for t in tasks:
        if t.get("target", "").strip().lower() == "no related clause":
            continue
        if added >= per_doc:
            break
        long_input = f"{t['input']}\n\nAPPENDIX REPEAT FOR STRESS TEST\n{t['input'][:12000]}"
        v = dict(t)
        v["id"] = make_id(t["doc_name"], t["query_intent"], t["target"], f"long_{added}")
        v["difficulty"] = "hard_long_document"
        v["input"] = long_input
        out.append(v)
        added += 1
    return out


def dedupe_tasks(tasks: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for t in tasks:
        input_digest = hashlib.sha1(normalize_space(t.get("input", "")).encode("utf-8")).hexdigest()[:12]
        key = (t.get("doc_name"), t.get("query_intent"), t.get("target"), input_digest)
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    docs: List[Tuple[str, str]] = []
    for r in records:
        if r.get("type") != "extraction":
            continue
        text = str(r.get("input", "")).strip()
        if not text:
            continue
        docs.append((str(r.get("doc_name", "unknown_doc")), text))
    if not docs:
        raise RuntimeError(f"No extraction documents found in {args.input}")

    generated: List[Dict] = []
    for doc_name, doc_text in docs:
        positives = build_positive_tasks(doc_name, doc_text)
        similar = build_similar_clause_hard_cases(doc_name, doc_text)
        negatives = build_negative_tasks(doc_name, doc_text, per_doc=args.negatives_per_doc)
        long_variants = build_long_variants(positives + similar, per_doc=args.long_variants_per_doc)
        generated.extend(positives)
        generated.extend(similar)
        generated.extend(negatives)
        generated.extend(long_variants)

    generated = dedupe_tasks(generated)
    if len(generated) < args.target_min:
        raise RuntimeError(
            f"Generated only {len(generated)} tasks (< target {args.target_min}). "
            "Increase source documents or lower --target-min."
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(generated, f, indent=2, ensure_ascii=False)

    by_diff: Dict[str, int] = {}
    for t in generated:
        by_diff[t.get("difficulty", "unknown")] = by_diff.get(t.get("difficulty", "unknown"), 0) + 1
    print(f"Saved {len(generated)} extraction QA tasks to {args.output}")
    print(f"Difficulty breakdown: {by_diff}")


if __name__ == "__main__":
    main()
