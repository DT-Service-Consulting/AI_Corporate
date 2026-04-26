"""Pattern recognition layer for FC-LAMMR."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fc_lammr.data_structures import Model, Pattern, RouterState, TaskType
from fc_lammr.utils.text_processing import (
    build_vectorizer,
    cosine_similarity_safe,
    make_audit_entry,
    normalise_belief,
    pattern_from_dict,
    pattern_to_dict,
    tokenise,
    vectorize_text,
)


LOGGER = logging.getLogger(__name__)


class PatternRecognitionLayer:
    """Fast pattern matching for routine legal queries."""

    _TASK_TYPE_KEYWORDS: dict[TaskType, tuple[str, ...]] = {
        TaskType.EXTRACTION: ("extract", "exactly", "written", "quote", "quoted", "text", "verbatim"),
        TaskType.CLAUSE_IDENTIFICATION: ("clause", "section", "provision", "sentence", "locate", "find", "identify"),
        TaskType.REASONING: ("analyze", "assess", "interpret", "valid", "fair", "unfair", "liable", "breach", "hearsay"),
        TaskType.NLI: ("does", "whether", "describe", "support", "entail", "contradict", "consistent", "inconsistent"),
        TaskType.COMPLIANCE_CHECK: ("compliance", "comply", "compliant", "violate", "requirement", "obligation"),
        TaskType.RISK_ASSESSMENT: ("risk", "exposure", "liability", "indemnity", "penalty", "damages"),
        TaskType.TENDER_EVALUATION: ("tender", "bid", "procurement", "subcontractor", "eligibility", "qualification", "award"),
    }
    _TASK_FAMILY: dict[TaskType, str] = {
        TaskType.EXTRACTION: "extractive",
        TaskType.CLAUSE_IDENTIFICATION: "extractive",
        TaskType.REASONING: "reasoning",
        TaskType.NLI: "reasoning",
        TaskType.COMPLIANCE_CHECK: "reasoning",
        TaskType.RISK_ASSESSMENT: "reasoning",
        TaskType.TENDER_EVALUATION: "reasoning",
    }
    _TASK_COMPATIBILITY: dict[TaskType, set[TaskType]] = {
        TaskType.EXTRACTION: {TaskType.EXTRACTION, TaskType.CLAUSE_IDENTIFICATION},
        TaskType.CLAUSE_IDENTIFICATION: {TaskType.EXTRACTION, TaskType.CLAUSE_IDENTIFICATION},
        TaskType.REASONING: {TaskType.REASONING, TaskType.NLI},
        TaskType.NLI: {TaskType.NLI, TaskType.REASONING},
        TaskType.COMPLIANCE_CHECK: {TaskType.COMPLIANCE_CHECK, TaskType.RISK_ASSESSMENT, TaskType.TENDER_EVALUATION},
        TaskType.RISK_ASSESSMENT: {TaskType.RISK_ASSESSMENT, TaskType.COMPLIANCE_CHECK, TaskType.TENDER_EVALUATION},
        TaskType.TENDER_EVALUATION: {TaskType.TENDER_EVALUATION, TaskType.COMPLIANCE_CHECK, TaskType.RISK_ASSESSMENT},
    }

    def __init__(
        self,
        library_path: str,
        similarity_threshold: float = 0.82,
        *,
        cold_library_threshold: float = 0.91,
        min_pattern_quality: float = 0.75,
    ):
        self.library_path = Path(library_path)
        self.similarity_threshold = similarity_threshold
        self.cold_library_threshold = cold_library_threshold
        self.min_pattern_quality = min_pattern_quality
        self.patterns: list[Pattern] = []
        self.combined_texts: list[str] = []
        self.vectorizer = build_vectorizer(["placeholder_token"])
        self._last_combined_text: str = ""
        self._last_features: list[float] = []
        self._vocabulary_frozen = False
        self.rerouted_log_path = (Path.cwd() / "results" / "fc_lammr" / "rerouted_tasks_log.json").resolve()
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
        self.library_path.parent.mkdir(parents=True, exist_ok=True)
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

    def _effective_similarity_threshold(self) -> float:
        if len(self.patterns) < 50:
            return self.cold_library_threshold
        return self.similarity_threshold

    def _query_task_signals(self, query: str) -> tuple[list[tuple[TaskType, int]], dict[str, int], bool]:
        lowered = str(query or "").lower()
        tokens = set(tokenise(lowered))
        task_scores: list[tuple[TaskType, int]] = []
        family_scores: dict[str, int] = {"extractive": 0, "reasoning": 0}
        for task_type, keywords in self._TASK_TYPE_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                if " " in keyword:
                    score += int(keyword in lowered)
                else:
                    score += int(keyword in tokens)
            if score > 0:
                task_scores.append((task_type, score))
                family_scores[self._TASK_FAMILY[task_type]] += score
        task_scores.sort(key=lambda pair: (-pair[1], pair[0].value))
        ranked = task_scores[:3]
        ordered_family_scores = sorted(family_scores.values(), reverse=True)
        dominant = ordered_family_scores[0] if ordered_family_scores else 0
        runner_up = ordered_family_scores[1] if len(ordered_family_scores) > 1 else 0
        ambiguous = runner_up > 0 and abs(dominant - runner_up) <= 1
        return ranked, family_scores, ambiguous

    def _task_type_consistent(self, query: str, matched_task_type: TaskType) -> tuple[bool, dict]:
        ranked_signals, family_scores, ambiguous = self._query_task_signals(query)
        if not ranked_signals:
            return True, {"ranked_signals": [], "family_scores": family_scores, "ambiguous": False}

        matched_family = self._TASK_FAMILY[matched_task_type]
        top_task_types = [task_type for task_type, _ in ranked_signals]
        top_families = {self._TASK_FAMILY[task_type] for task_type in top_task_types}
        if ambiguous and len(top_families) > 1:
            return False, {
                "ranked_signals": [task_type.value for task_type in top_task_types],
                "family_scores": family_scores,
                "ambiguous": True,
            }

        dominant_family = max(family_scores, key=family_scores.get)
        if matched_family != dominant_family:
            return False, {
                "ranked_signals": [task_type.value for task_type in top_task_types],
                "family_scores": family_scores,
                "ambiguous": ambiguous,
            }

        compatible = any(
            matched_task_type in self._TASK_COMPATIBILITY[signalled_task_type]
            for signalled_task_type in top_task_types
        )
        return compatible, {
            "ranked_signals": [task_type.value for task_type in top_task_types],
            "family_scores": family_scores,
            "ambiguous": ambiguous,
        }

    def route(self, state: RouterState) -> RouterState:
        features = self.extract_features(state.query, state.document)
        pattern, similarity = self.find_best_match(features)
        threshold = self._effective_similarity_threshold()
        if pattern is None or similarity < threshold:
            state.audit_log.append(
                make_audit_entry(
                    layer="PRL",
                    decision="no_match",
                    belief_state={},
                    rationale=f"No stored pattern exceeded similarity threshold {threshold:.2f}.",
                    similarity=similarity,
                    threshold=threshold,
                )
            )
            return state
        if pattern.outcome_score < self.min_pattern_quality:
            state.audit_log.append(
                make_audit_entry(
                    layer="PRL",
                    decision="match_rejected_low_quality",
                    belief_state={},
                    rationale=(
                        f"Best pattern matched at cosine similarity {similarity:.3f}, but stored outcome score "
                        f"{pattern.outcome_score:.2f} is below the trust floor {self.min_pattern_quality:.2f}."
                    ),
                    similarity=similarity,
                    threshold=threshold,
                    matched_task_type=pattern.task_type,
                    matched_model=pattern.model_used,
                    outcome_score=pattern.outcome_score,
                )
            )
            return state
        is_consistent, signal_details = self._task_type_consistent(state.query, pattern.task_type)
        if not is_consistent:
            state.audit_log.append(
                make_audit_entry(
                    layer="PRL",
                    decision="match_rejected_task_type_inconsistent",
                    belief_state={},
                    rationale=(
                        f"Best pattern matched at cosine similarity {similarity:.3f}, but query-level task cues "
                        f"were inconsistent with stored task type {pattern.task_type.value}."
                    ),
                    similarity=similarity,
                    threshold=threshold,
                    matched_task_type=pattern.task_type,
                    matched_model=pattern.model_used,
                    keyword_task_signals=signal_details["ranked_signals"],
                    keyword_family_scores=signal_details["family_scores"],
                    ambiguous_query=signal_details["ambiguous"],
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
                threshold=threshold,
                outcome_score=pattern.outcome_score,
                library_size=len(self.patterns),
            )
        )
        return state

    def _log_rerouted_task(self, state: RouterState, outcome_score: float) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": state.query,
            "document_preview": state.document[:500],
            "reroute_count": state.reroute_count,
            "final_model": state.assigned_model.value if state.assigned_model else None,
            "score": outcome_score,
        }
        try:
            self.rerouted_log_path.parent.mkdir(parents=True, exist_ok=True)
            existing = []
            if self.rerouted_log_path.exists():
                existing = json.loads(self.rerouted_log_path.read_text(encoding="utf-8") or "[]")
            existing.append(entry)
            self.rerouted_log_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
        except (OSError, ValueError, TypeError) as exc:
            fallback_path = self.rerouted_log_path.with_name("rerouted_tasks_log_fallback.jsonl")
            try:
                fallback_path.parent.mkdir(parents=True, exist_ok=True)
                with fallback_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
                LOGGER.warning(
                    "Primary reroute log write failed at %s (%s). Wrote fallback entry to %s instead.",
                    self.rerouted_log_path,
                    exc,
                    fallback_path,
                )
            except OSError:
                LOGGER.exception("Failed to persist rerouted task log to both primary and fallback locations.")

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
        if self._vocabulary_frozen:
            logging.info(
                "Skipping pattern write-back during frozen evaluation run. Final model: %s. Score: %s.",
                state.assigned_model.value if state.assigned_model else "unknown",
                outcome_score,
            )
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
