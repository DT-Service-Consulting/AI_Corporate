"""Fluid rerouting layer for FC-LAMMR."""

from __future__ import annotations

import logging
import re

from fc_lammr.data_structures import Model, RouterState, StruggleSignal
from fc_lammr.utils.text_processing import make_audit_entry, normalise_belief


LOGGER = logging.getLogger(__name__)


class FluidReroutingLayer:
    """Monitor model outputs and re-route when quality degrades."""

    UNCERTAINTY_PHRASES: list = [
        "it is unclear",
        "this may depend on",
        "i cannot determine",
        "subject to interpretation",
        "ambiguous",
        "may vary",
        "further clarification required",
        "open to interpretation",
        "not definitively stated",
        "could be construed as",
    ]

    def __init__(self, reroute_threshold: float = 0.65, signal_penalty: float = 0.15):
        self.reroute_threshold = reroute_threshold
        self.signal_penalty = signal_penalty

    def detect_struggle_signals(self, partial_output: str, document: str) -> list[StruggleSignal]:
        output = str(partial_output or "")
        doc_lower = str(document or "").lower()
        signals: list[StruggleSignal] = []
        output_lower = output.lower()
        for phrase in self.UNCERTAINTY_PHRASES:
            signals.extend([StruggleSignal.UNCERTAINTY_SIGNAL] * output_lower.count(phrase))
        sentences = re.split(r"(?<=[.!?])\s+", output_lower)
        for index, sentence in enumerate(sentences[1:], start=1):
            previous = sentences[index - 1]
            if ("however" in sentence or "but" in sentence) and (" not " in sentence or " no " in sentence):
                if any(token in previous for token in [" is ", " does ", " can ", " will "]):
                    signals.append(StruggleSignal.CONTRADICTION_SIGNAL)
        clause_refs = re.findall(r"\b(?:clause|section)\s+\d+(?:\.\d+)*\b", output, flags=re.IGNORECASE)
        defined_terms = re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", output)
        for entity in clause_refs + defined_terms:
            if entity.lower() not in doc_lower:
                signals.append(StruggleSignal.GROUNDING_FAILURE)
        return signals

    def update_belief(self, state: RouterState, signals: list[StruggleSignal]) -> RouterState:
        if state.assigned_model is None:
            return state
        state.struggle_signals_detected.extend(signals)
        for signal in signals:
            other_model = Model.REASONING_MODEL if state.assigned_model == Model.EXTRACTION_MODEL else Model.EXTRACTION_MODEL
            penalty = min(self.signal_penalty, state.routing_belief.get(state.assigned_model, 0.0))
            state.routing_belief[state.assigned_model] = max(0.0, state.routing_belief.get(state.assigned_model, 0.0) - penalty)
            state.routing_belief[other_model] = state.routing_belief.get(other_model, 0.0) + penalty
            state.routing_belief = normalise_belief(state.routing_belief)
        return state

    def sanitise_partial_output_for_handoff(self, partial_output: str, signals: list[StruggleSignal]) -> str:
        """Flag problematic partial output segments before handoff."""
        text = str(partial_output or "")
        sentences = re.split(r"(?<=[.!?])\s+", text)
        updated = []
        for sentence in sentences:
            lower = sentence.lower()
            flagged = sentence
            if any(phrase in lower for phrase in self.UNCERTAINTY_PHRASES):
                flagged = f"[UNCERTAIN: {flagged}]"
            if ("however" in lower or "but" in lower) and (" not " in lower or " no " in lower):
                flagged = f"[CONTRADICTS PRIOR STATEMENT: {flagged}]"
            if re.search(r"\b(?:clause|section)\s+\d+(?:\.\d+)*\b", sentence, flags=re.IGNORECASE):
                flagged = f"[UNVERIFIED REFERENCE - NOT FOUND IN SOURCE DOCUMENT: {flagged}]"
            updated.append(flagged)
        return " ".join(updated).strip()

    def build_handoff_prompt(self, state: RouterState) -> str:
        signals = ", ".join(signal.value for signal in state.struggle_signals_detected) or "none"
        sanitised = self.sanitise_partial_output_for_handoff(state.partial_output or "", state.struggle_signals_detected)
        return (
            "HANDOFF CONTEXT: The following partial analysis was produced by a previous model and contains flagged issues. "
            "Sections marked [UNCERTAIN], [CONTRADICTS PRIOR STATEMENT], or [UNVERIFIED REFERENCE] should be re-verified "
            "against the source document before being included in your response. Do not treat flagged sections as established facts.\n"
            f"Original query: {state.query}\n"
            f"Completed so far: {sanitised or 'No partial output was captured.'}\n"
            "What remains: finish the legal analysis, resolve uncertainty, and ground every conclusion in the document.\n"
            f"Re-route trigger: {signals}\n"
            f"Document excerpt basis:\n{state.document}"
        )

    def monitor_and_reroute(self, state: RouterState, partial_output: str) -> RouterState:
        state.partial_output = partial_output
        signals = self.detect_struggle_signals(partial_output, state.document)
        state = self.update_belief(state, signals)
        if state.assigned_model is None:
            return state
        current_belief = state.routing_belief.get(state.assigned_model, 0.0)
        if current_belief < self.reroute_threshold:
            original_model = state.assigned_model
            new_model = Model.REASONING_MODEL if original_model == Model.EXTRACTION_MODEL else Model.EXTRACTION_MODEL
            state.reroute_triggered = True
            state.reroute_count += 1
            state.assigned_model = new_model
            other_model = Model.REASONING_MODEL if new_model == Model.EXTRACTION_MODEL else Model.EXTRACTION_MODEL
            state.routing_belief = normalise_belief({new_model: 0.72, other_model: 0.28})
            state.audit_log.append(
                make_audit_entry(
                    layer="FRL",
                    decision="reroute",
                    belief_state=state.routing_belief,
                    rationale="Detected enough struggle signals to drop confidence below the reroute threshold.",
                    original_model=original_model,
                    new_model=new_model,
                    trigger_signals=signals,
                    belief_at_trigger=current_belief,
                    partial_output=partial_output,
                )
            )
        else:
            state.audit_log.append(
                make_audit_entry(
                    layer="FRL",
                    decision="monitor_continue",
                    belief_state=state.routing_belief,
                    rationale="Belief remained above the reroute threshold after monitoring.",
                    trigger_signals=signals,
                )
            )
        return state
