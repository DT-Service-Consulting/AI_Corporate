"""Pattern recognition layer for FC-LAMMR."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fc_lammr.data_structures import Model, Pattern, RouterState
from fc_lammr.utils.text_processing import (
    build_vectorizer,
    cosine_similarity_safe,
    make_audit_entry,
    normalise_belief,
    pattern_from_dict,
    pattern_to_dict,
    vectorize_text,
)


LOGGER = logging.getLogger(__name__)


class PatternRecognitionLayer:
    """Fast pattern matching for routine legal queries."""

    def __init__(self, library_path: str, similarity_threshold: float = 0.82):
        self.library_path = Path(library_path)
        self.similarity_threshold = similarity_threshold
        self.patterns: list[Pattern] = []
        self.combined_texts: list[str] = []
        self.vectorizer = build_vectorizer(["placeholder_token"])
        self._last_combined_text: str = ""
        self._last_features: list[float] = []
        self._vocabulary_frozen = False
        self.rerouted_log_path = Path("results") / "rerouted_tasks_log.json"
        self._load_library()

    def _load_library(self) -> None:
        if not self.library_path.exists():
            self.library_path.write_text("[]", encoding="utf-8")
        raw_payload = json.loads(self.library_path.read_text(encoding="utf-8") or "[]")
        self.patterns = []
        self.combined_texts = []
        for item in raw_payload:
            pattern, combined_text = pattern_from_dict(item)
            self.patterns.append(pattern)
            self.combined_texts.append(combined_text)

    def prefetch_and_fit_vocabulary(self, all_queries: list[str], all_documents: list[str]) -> None:
        """
        Fits the TF-IDF vectorizer on the full evaluation dataset
        before any routing decisions are made.
        """
        combined = [f"{query} {document}".strip() for query, document in zip(all_queries, all_documents)]
        self.vectorizer = build_vectorizer(combined or ["placeholder_token"])
        self._vocabulary_frozen = True
        for index, text in enumerate(self.combined_texts):
            self.patterns[index].query_features = vectorize_text(self.vectorizer, text)
        logging.info(
            "TF-IDF vocabulary pre-fitted on %s documents. Vocabulary size: %s. Vocabulary is now frozen for evaluation.",
            len(combined),
            len(getattr(self.vectorizer, "vocabulary_", {})),
        )

    def _persist_library(self) -> None:
        payload = [
            pattern_to_dict(pattern, combined_text=text)
            for pattern, text in zip(self.patterns, self.combined_texts)
        ]
        self.library_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def extract_features(self, query: str, document: str) -> list:
        if not self._vocabulary_frozen:
            raise RuntimeError(
                "TF-IDF vocabulary not pre-fitted. Call prefetch_and_fit_vocabulary() before beginning evaluation. "
                "Skipping this step invalidates PRL results."
            )
        combined_text = f"{query}\n{document}".strip()
        self._last_combined_text = combined_text
        self._last_features = vectorize_text(self.vectorizer, combined_text)
        return list(self._last_features)

    def find_best_match(self, features: list) -> tuple[Optional[Pattern], float]:
        if not self.patterns:
            return None, 0.0
        best_pattern = None
        best_score = 0.0
        for pattern in self.patterns:
            score = cosine_similarity_safe(features, pattern.query_features)
            if score > best_score:
                best_pattern = pattern
                best_score = score
        return best_pattern, best_score

    def route(self, state: RouterState) -> RouterState:
        features = self.extract_features(state.query, state.document)
        pattern, similarity = self.find_best_match(features)
        if pattern is None or similarity < self.similarity_threshold:
            state.audit_log.append(
                make_audit_entry(
                    layer="PRL",
                    decision="no_match",
                    belief_state={},
                    rationale=f"No stored pattern exceeded similarity threshold {self.similarity_threshold:.2f}.",
                    similarity=similarity,
                )
            )
            return state
        state.task_type = pattern.task_type
        state.assigned_model = pattern.model_used
        other_model = Model.REASONING_MODEL if pattern.model_used == Model.EXTRACTION_MODEL else Model.EXTRACTION_MODEL
        state.routing_belief = normalise_belief({pattern.model_used: max(similarity, 0.51), other_model: max(0.01, 1.0 - max(similarity, 0.51))})
        state.effective_routing_mode = "PRL_MATCH"
        state.audit_log.append(
            make_audit_entry(
                layer="PRL",
                decision="pattern_match",
                belief_state=state.routing_belief,
                rationale=f"Matched stored pattern '{pattern.query_text_preview}' with cosine similarity {similarity:.3f}.",
                task_type=state.task_type,
                assigned_model=state.assigned_model,
            )
        )
        return state

    def _log_rerouted_task(self, state: RouterState, outcome_score: float) -> None:
        self.rerouted_log_path.parent.mkdir(parents=True, exist_ok=True)
        existing = []
        if self.rerouted_log_path.exists():
            existing = json.loads(self.rerouted_log_path.read_text(encoding="utf-8") or "[]")
        existing.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "query": state.query,
                "document_preview": state.document[:500],
                "reroute_count": state.reroute_count,
                "final_model": state.assigned_model.value if state.assigned_model else None,
                "score": outcome_score,
            }
        )
        self.rerouted_log_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    def write_back(self, state: RouterState, outcome_score: float) -> None:
        if state.reroute_triggered:
            logging.info(
                "Skipping pattern write-back for rerouted task. Reroute count: %s. Final model: %s. Score: %s.",
                state.reroute_count,
                state.assigned_model.value if state.assigned_model else "unknown",
                outcome_score,
            )
            self._log_rerouted_task(state, outcome_score)
            return
        if state.task_type is None or state.assigned_model is None:
            return
        combined_text = f"{state.query}\n{state.document}".strip()
        self.combined_texts.append(combined_text)
        self.patterns.append(
            Pattern(
                query_features=vectorize_text(self.vectorizer, combined_text),
                task_type=state.task_type,
                model_used=state.assigned_model,
                outcome_score=float(outcome_score),
                query_text_preview=state.query[:100],
            )
        )
        self._persist_library()
