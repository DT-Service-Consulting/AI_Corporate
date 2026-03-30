"""Main FC-LAMMR orchestration layer."""

from __future__ import annotations

import logging

from fc_lammr.data_structures import Model, RouterState
from fc_lammr.evaluation_layer import EvaluationLayer
from fc_lammr.fluid_rerouting_layer import FluidReroutingLayer
from fc_lammr.pattern_recognition_layer import PatternRecognitionLayer
from fc_lammr.tom_inference_layer import TheoryOfMindInferenceLayer
from fc_lammr.utils.llm_client import OpenAICompatibleLLMClient, get_project_deployment_config, validate_deployment_config
from fc_lammr.utils.prompt_helpers import build_extraction_prompt, postprocess_extraction_output


LOGGER = logging.getLogger(__name__)


class FCLAMMRRouter:
    """Orchestrator that coordinates all three layers."""

    MAX_REROUTES: int = 2

    def __init__(
        self,
        llm_client=None,
        pattern_library_path: str = "fc_lammr/pattern_library.json",
        prl_threshold: float = 0.82,
        reroute_threshold: float = 0.65,
        signal_penalty: float = 0.15,
    ):
        validate_deployment_config()
        config = get_project_deployment_config()
        self.extraction_deployment = config["EXTRACTION_DEPLOYMENT_NAME"]
        self.reasoning_deployment = config["REASONING_DEPLOYMENT_NAME"]
        self.tomil_deployment = config["TOMIL_DEPLOYMENT_NAME"]
        self.llm_client = llm_client if hasattr(llm_client, "create_chat_completion") else OpenAICompatibleLLMClient(llm_client, provider="azure")
        self.prl = PatternRecognitionLayer(pattern_library_path, similarity_threshold=prl_threshold)
        self.tomil = TheoryOfMindInferenceLayer(self.llm_client, inference_model=self.tomil_deployment)
        self.frl = FluidReroutingLayer(reroute_threshold=reroute_threshold, signal_penalty=signal_penalty)
        self.evaluator = EvaluationLayer()
        self.last_handoff_prompt: str | None = None

    def _deployment_for(self, model: Model) -> str:
        return self.extraction_deployment if model == Model.EXTRACTION_MODEL else self.reasoning_deployment

    def _build_reasoning_prompt(self, state: RouterState) -> str:
        return (
            "### ROLE\n"
            "You are a precision legal reasoning assistant.\n\n"
            "### QUERY\n"
            f"{state.query}\n\n"
            "### DOCUMENT\n"
            f"{state.document}\n\n"
            "### INSTRUCTIONS\n"
            "1. Analyze the legal implications of the document for the query.\n"
            "2. State your reasoning briefly but clearly.\n"
            "3. End with a clear verdict when appropriate using the format: ANSWER: <verdict>.\n"
            "4. Ground every conclusion in the provided document.\n"
        )

    def _execute_model(self, model: Model, prompt: str, document: str, *, is_extraction: bool = False) -> tuple[str, str | None]:
        """Execute a model call and report explicit failure status instead of crashing."""
        system_prompt = "You are a precise legal extraction engine." if is_extraction else "You are a careful legal reasoning engine."
        try:
            response = self.llm_client.create_chat_completion(
                model=self._deployment_for(model),
                temperature=0.1,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{prompt}\n\nDOCUMENT:\n{document}"},
                ],
            )
            content = str(response.choices[0].message.content or "").strip()
            return (postprocess_extraction_output(content) if is_extraction else content), None
        except RuntimeError as exc:
            message = str(exc)
            LOGGER.warning("Execution fallback engaged for model=%s", model.value)
            if message.startswith("CONTENT_FILTER_BLOCKED::"):
                return "", "CONTENT_FILTER_BLOCKED"
            if message.startswith("FAILED_LLM_CALL::"):
                return "", "FAILED_LLM_CALL"
            return "", "FAILED_LLM_CALL"
        except Exception:
            LOGGER.warning("Execution fallback engaged for model=%s", model.value)
            return "", "FAILED_LLM_CALL"

    def _score_outcome(self, state: RouterState) -> float:
        if not state.final_output:
            return 0.0
        if state.reroute_triggered:
            return 0.85
        return 0.75 if len(state.final_output.split()) >= 5 else 0.6

    def route(self, query: str, document: str) -> RouterState:
        state = RouterState(query=query, document=document)
        state = self.prl.route(state)
        if state.assigned_model is None:
            state = self.tomil.infer_intent(state)
            if state.tom_inference_quality and state.tom_inference_quality.parse_path == "TOMIL_PARSE_FAILURE":
                state.effective_routing_mode = state.result_status or "TOMIL_PARSE_FAILURE"
                return state
            features = self.prl.extract_features(query, document)
            state = self.tomil.compute_routing_belief(state, features)
            if state.tom_inference_quality and state.tom_inference_quality.normalisation_applied:
                state.effective_routing_mode = "TOMIL_NORMALISED"
            else:
                state.effective_routing_mode = "TOMIL_SUCCESS"
        else:
            state.effective_routing_mode = "PRL_MATCH"

        current_model = state.assigned_model or Model.REASONING_MODEL
        is_extraction = state.task_type is None or state.task_type.value in {"extraction", "clause_identification"}
        prompt = build_extraction_prompt(query, document) if is_extraction else self._build_reasoning_prompt(state)

        while True:
            output, failure_status = self._execute_model(current_model, prompt, document, is_extraction=is_extraction)
            if failure_status:
                state.result_status = failure_status
                state.effective_routing_mode = failure_status
                return state
            state = self.frl.monitor_and_reroute(state, output)
            if not state.reroute_triggered:
                state.final_output = output
                break
            state.effective_routing_mode = "REROUTED"
            if state.reroute_count >= self.MAX_REROUTES:
                logging.warning(
                    "Task reached MAX_REROUTES (%s). Forcing completion with current model: %s.",
                    self.MAX_REROUTES,
                    state.assigned_model.value if state.assigned_model else "unknown",
                )
                state.final_output = output
                break
            self.last_handoff_prompt = self.frl.build_handoff_prompt(state)
            prompt = self.last_handoff_prompt
            current_model = state.assigned_model or current_model

        self.prl.write_back(state, outcome_score=self._score_outcome(state))
        return state

    def explain_route(self, state: RouterState) -> str:
        deciding_layer = "unknown"
        for entry in state.audit_log:
            if entry.get("decision") in {"pattern_match", "routing_belief_computed"}:
                deciding_layer = entry.get("layer", deciding_layer)
        belief_scores = {
            (model.value if hasattr(model, "value") else str(model)): round(score, 4)
            for model, score in state.routing_belief.items()
        }
        reroute_summary = f"reroute_occurred={state.reroute_triggered}, signals={[signal.value for signal in state.struggle_signals_detected]}"
        return (
            f"layer_decision={deciding_layer}; "
            f"intent_hypothesis={state.intent_hypothesis}; "
            f"routing_belief_scores={belief_scores}; "
            f"reroute_summary={reroute_summary}; "
            f"risk_weight_applied={state.risk_weight}"
        )
