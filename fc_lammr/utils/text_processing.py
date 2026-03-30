"""Text processing helpers for FC-LAMMR."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from fc_lammr.data_structures import Model, Pattern, RouterState, StruggleSignal, TaskType


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp for audit records."""
    return datetime.now(timezone.utc).isoformat()


def build_vectorizer(corpus: list[str]) -> TfidfVectorizer:
    """Fit a stable TF-IDF vectorizer on the provided corpus."""
    safe_corpus = [text for text in (corpus or []) if str(text).strip()] or ["placeholder_token"]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=512,
    )
    vectorizer.fit(safe_corpus)
    return vectorizer


def vectorize_text(vectorizer: TfidfVectorizer, text: str) -> list[float]:
    """Transform text into a dense list of TF-IDF weights."""
    return vectorizer.transform([text]).toarray()[0].astype(float).tolist()


def cosine_similarity_safe(left: Iterable[float], right: Iterable[float]) -> float:
    """Compute cosine similarity while tolerating empty or mismatched vectors."""
    left_array = np.asarray(list(left), dtype=float)
    right_array = np.asarray(list(right), dtype=float)
    if left_array.size == 0 or right_array.size == 0:
        return 0.0
    if left_array.size != right_array.size:
        width = max(left_array.size, right_array.size)
        left_array = np.pad(left_array, (0, width - left_array.size))
        right_array = np.pad(right_array, (0, width - right_array.size))
    denominator = float(np.linalg.norm(left_array) * np.linalg.norm(right_array))
    if math.isclose(denominator, 0.0):
        return 0.0
    return float(np.dot(left_array, right_array) / denominator)


def normalise_belief(belief: dict) -> dict:
    """Normalise a routing belief distribution so it sums to 1.0."""
    total = float(sum(belief.values()))
    if math.isclose(total, 0.0):
        even = 1.0 / max(len(belief), 1)
        return {key: even for key in belief}
    return {key: float(value) / total for key, value in belief.items()}


def make_audit_entry(layer: str, decision: str, belief_state: dict, rationale: str, **extra: object) -> dict:
    """Create a JSON-serialisable audit entry with a consistent schema."""
    entry = {
        "timestamp": utc_timestamp(),
        "layer": layer,
        "decision": decision,
        "belief_state": {
            (key.value if isinstance(key, Model) else str(key)): float(value)
            for key, value in belief_state.items()
        },
        "rationale": rationale,
    }
    for key, value in extra.items():
        if isinstance(value, Model):
            entry[key] = value.value
        elif isinstance(value, TaskType):
            entry[key] = value.value
        elif isinstance(value, StruggleSignal):
            entry[key] = value.value
        elif isinstance(value, list):
            entry[key] = [
                item.value if isinstance(item, (Model, TaskType, StruggleSignal)) else item
                for item in value
            ]
        else:
            entry[key] = value
    return entry


def pattern_to_dict(pattern: Pattern, combined_text: str | None = None) -> dict:
    """Serialise a Pattern to JSON-friendly data."""
    payload = asdict(pattern)
    payload["task_type"] = pattern.task_type.value
    payload["model_used"] = pattern.model_used.value
    if combined_text is not None:
        payload["combined_text"] = combined_text
    return payload


def pattern_from_dict(payload: dict) -> tuple[Pattern, str]:
    """Deserialise a Pattern and preserve the source text for refitting."""
    pattern = Pattern(
        query_features=list(payload.get("query_features", [])),
        task_type=TaskType(payload["task_type"]),
        model_used=Model(payload["model_used"]),
        outcome_score=float(payload["outcome_score"]),
        query_text_preview=str(payload.get("query_text_preview", "")),
    )
    combined_text = str(payload.get("combined_text") or payload.get("query_text_preview", ""))
    return pattern, combined_text


def export_audit_log(state: RouterState, filepath: str) -> None:
    """Persist the audit log so legal reviewers can inspect routing decisions."""
    serialisable = []
    for entry in state.audit_log:
        serialisable.append(json.loads(json.dumps(entry)))
    Path(filepath).write_text(json.dumps(serialisable, indent=2), encoding="utf-8")


def tokenise(text: str) -> list[str]:
    """Basic tokenizer used by the evaluation layer."""
    return re.findall(r"\w+", str(text or "").lower())
