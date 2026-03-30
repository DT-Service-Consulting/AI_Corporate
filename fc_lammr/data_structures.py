"""Shared data structures for FC-LAMMR."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TaskType(Enum):
    EXTRACTION = "extraction"
    REASONING = "reasoning"
    RISK_ASSESSMENT = "risk_assessment"
    NLI = "nli"
    COMPLIANCE_CHECK = "compliance_check"
    CLAUSE_IDENTIFICATION = "clause_identification"
    TENDER_EVALUATION = "tender_evaluation"


class Model(Enum):
    EXTRACTION_MODEL = "extraction_model"
    REASONING_MODEL = "reasoning_model"


class Phase(Enum):
    IDENTIFICATION = "identification"
    INTERPRETATION = "interpretation"
    RISK_ASSESSMENT = "risk_assessment"


class StruggleSignal(Enum):
    UNCERTAINTY_SIGNAL = "uncertainty_signal"
    CONTRADICTION_SIGNAL = "contradiction_signal"
    GROUNDING_FAILURE = "grounding_failure"


@dataclass
class TomInferenceQuality:
    """
    Records exactly how the ToM inference result was obtained.
    This is the primary evidence for whether ToM routing was clean enough
    to count in the primary research comparison.
    """

    parse_path: str
    raw_model_response: str
    task_type_before_normalisation: Optional[str]
    confidence_before_normalisation: Optional[str]
    normalisation_applied: bool
    tom_is_valid_for_research: bool


@dataclass
class RouterState:
    """
    Central state object passed between all three layers.
    Every mutation to routing state must go through this object.
    This is the audit trail for route explainability.
    """

    query: str
    document: str
    task_type: Optional[TaskType] = None
    intent_hypothesis: Optional[str] = None
    inferred_phase: Optional[Phase] = None
    risk_weight: float = 0.0
    routing_belief: dict = field(default_factory=dict)
    assigned_model: Optional[Model] = None
    reroute_triggered: bool = False
    reroute_count: int = 0
    struggle_signals_detected: list = field(default_factory=list)
    partial_output: Optional[str] = None
    final_output: Optional[str] = None
    audit_log: list = field(default_factory=list)
    tom_inference_quality: Optional[TomInferenceQuality] = None
    effective_routing_mode: Optional[str] = None
    result_status: Optional[str] = None
    content_filter_details: Optional[dict] = None
    llm_call_failed: bool = False


@dataclass
class Pattern:
    """
    Stored pattern in the pattern library.
    Features are stored as a list of floats (TF-IDF vector).
    """

    query_features: list
    task_type: TaskType
    model_used: Model
    outcome_score: float
    query_text_preview: str
