"""Local prompt and scoring helpers copied from the legacy baseline module."""

from __future__ import annotations

import collections
import re
import string


NOT_FOUND_TOKEN = "NOT_FOUND"


def normalize_text(text):
    """
    Normalize text for overlap metrics:
    - casing/whitespace cleanup
    - punctuation removal
    - section symbol normalization
    """
    if not text:
        return ""
    text = str(text)
    text = text.replace("§", " section ")
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("`", "'")
    text = re.sub(r"\b(sec\.|cl\.)\b", "section", text, flags=re.IGNORECASE)
    text = text.lower()
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()


def canonicalize_missing(text: str) -> str:
    t = normalize_text(text)
    markers = {
        "not found",
        "notfound",
        "no related clause",
        "no clause found",
        "no relevant clause",
        "none",
        "na",
        "n a",
    }
    if t in markers:
        return NOT_FOUND_TOKEN
    return str(text)


def postprocess_extraction_output(text: str) -> str:
    if not text:
        return ""
    out = str(text).strip()
    out = re.sub(r"^```[a-zA-Z]*\s*", "", out).strip()
    out = re.sub(r"\s*```$", "", out).strip()
    out = re.sub(r"^(answer|final answer|output)\s*:\s*", "", out, flags=re.IGNORECASE).strip()
    out = re.sub(r"^\s*['\"]|['\"]\s*$", "", out).strip()
    out = re.sub(r"\s+", " ", out).strip()
    canon = canonicalize_missing(out)
    if canon == NOT_FOUND_TOKEN:
        return NOT_FOUND_TOKEN
    return out


def get_jaccard(gt, pred):
    """Calculates Jaccard similarity on token sets."""
    gt_clean = normalize_text(gt)
    pred_clean = normalize_text(pred)

    gt_words = set(gt_clean.split())
    pred_words = set(pred_clean.split())

    if len(gt_words) == 0 and len(pred_words) == 0:
        return 1.0

    intersection = len(gt_words.intersection(pred_words))
    union = len(gt_words.union(pred_words))

    return intersection / union if union > 0 else 0.0


def calculate_advanced_metrics(prediction, ground_truth, canonicalize_missing_enabled: bool = True):
    """
    Calculates Precision, Recall, F1, and F2 (Beta=2),
    with canonical NOT_FOUND handling.
    """
    pred_raw = postprocess_extraction_output(prediction)
    gt_raw = canonicalize_missing(ground_truth) if canonicalize_missing_enabled else str(ground_truth)
    if not canonicalize_missing_enabled:
        pred_raw = str(prediction or "").strip()

    if gt_raw == NOT_FOUND_TOKEN and pred_raw == NOT_FOUND_TOKEN:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "f2": 1.0,
            "jaccard": 1.0,
            "is_lazy": False,
        }

    if gt_raw == NOT_FOUND_TOKEN and pred_raw != NOT_FOUND_TOKEN:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "f2": 0.0,
            "jaccard": 0.0,
            "is_lazy": False,
        }

    pred_tokens = normalize_text(pred_raw).split()
    truth_tokens = normalize_text(gt_raw).split()

    is_lazy = pred_raw == NOT_FOUND_TOKEN and len(truth_tokens) > 0

    common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())

    precision = 0.0 if len(pred_tokens) == 0 else num_same / len(pred_tokens)
    recall = 0.0 if len(truth_tokens) == 0 else num_same / len(truth_tokens)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (2 * precision * recall) / (precision + recall)

    if (4 * precision) + recall == 0:
        f2 = 0.0
    else:
        f2 = (5 * precision * recall) / ((4 * precision) + recall)

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "f2": round(f2, 3),
        "jaccard": round(get_jaccard(gt_raw, pred_raw), 3),
        "is_lazy": is_lazy,
    }


def build_extraction_prompt(requirement_text: str, document_text: str) -> str:
    return f"""
    ### ROLE
    You are a high-precision Legal Extraction AI.

    ### TASK
    Extract the exact span from DOCUMENT that satisfies REQUIREMENT.

    ### REQUIREMENT
    "{requirement_text}"

    ### DOCUMENT
    "{document_text}"

    ### STRICT OUTPUT RULES
    1. Return only the exact copied span from DOCUMENT. No paraphrase.
    2. Do not add explanations, headers, or formatting.
    3. If the span is missing, output exactly: {NOT_FOUND_TOKEN}
    4. Output one span only.
    """
