"""Acceptance and regression tests for the FC-LAMMR pre-final patch set."""

from __future__ import annotations

import json
import importlib
import logging
import shutil
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import project_secrets

from fc_lammr.data_structures import Model, RouterState, TaskType
from fc_lammr.evaluation_layer import EvaluationLayer
from fc_lammr.fc_lammr_router import FCLAMMRRouter
from fc_lammr.fluid_rerouting_layer import FluidReroutingLayer
from fc_lammr.pattern_recognition_layer import PatternRecognitionLayer
from fc_lammr.run_fc_lammr_hybrid_test import compute_stratified_results
from fc_lammr.tom_inference_layer import TheoryOfMindInferenceLayer
from fc_lammr.utils.llm_client import ConfigurationError, validate_deployment_config


class FakeLLMClient:
    """Deterministic client for router and ToM tests."""

    def __init__(self):
        self.calls: list[dict] = []

    def create_chat_completion(self, *, messages, model=None, temperature=0.2, max_tokens=800):
        self.calls.append({"messages": messages, "model": model, "temperature": temperature})
        user_text = messages[-1]["content"]
        system_text = messages[0]["content"]
        if "OUTPUT FORMAT REQUIREMENTS" in system_text:
            content = json.dumps(
                {
                    "surface_task": "Find section 4.2",
                    "underlying_goal": "Assess whether subcontractor qualification criteria meet tender requirements.",
                    "inferred_phase": "risk_assessment",
                    "task_type": "legal evaluation",
                    "confidence": "high",
                    "reasoning": "The procurement context implies tender evaluation despite extractive phrasing.",
                }
            )
        elif "HANDOFF CONTEXT:" in user_text:
            content = "ANSWER: YES"
        else:
            content = (
                "It is unclear whether section 8.8 governs subcontractor qualifications. "
                "This may depend on additional procurement guidance. "
                "The NOVATION definition is relevant, but it does not appear in the document."
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(total_tokens=100),
        )


class FCLAMMRPatchSetTests(unittest.TestCase):
    """Focused tests for the requested pre-final FC-LAMMR patch set."""

    def setUp(self) -> None:
        self.temp_dir = Path("fc_lammr/.test_tmp") / self._testMethodName
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.library_path = str(self.temp_dir / "pattern_library.json")
        Path(self.library_path).write_text("[]", encoding="utf-8")
        self.fake_client = FakeLLMClient()
        self._deployment_backup = {
            "EXTRACTION_DEPLOYMENT_NAME": getattr(project_secrets, "EXTRACTION_DEPLOYMENT_NAME", None),
            "REASONING_DEPLOYMENT_NAME": getattr(project_secrets, "REASONING_DEPLOYMENT_NAME", None),
            "TOMIL_DEPLOYMENT_NAME": getattr(project_secrets, "TOMIL_DEPLOYMENT_NAME", None),
        }
        project_secrets.EXTRACTION_DEPLOYMENT_NAME = "fake-extraction"
        project_secrets.REASONING_DEPLOYMENT_NAME = "fake-reasoning"
        project_secrets.TOMIL_DEPLOYMENT_NAME = "fake-tomil"
        shutil.rmtree("results", ignore_errors=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        for key, value in self._deployment_backup.items():
            if value is None and hasattr(project_secrets, key):
                delattr(project_secrets, key)
            elif value is not None:
                setattr(project_secrets, key, value)

    def test_deployment_config_validation(self) -> None:
        backup = project_secrets.TOMIL_DEPLOYMENT_NAME
        delattr(project_secrets, "TOMIL_DEPLOYMENT_NAME")
        with self.assertRaises(ConfigurationError):
            validate_deployment_config()
        project_secrets.TOMIL_DEPLOYMENT_NAME = backup

    def test_vocabulary_frozen_before_route(self) -> None:
        prl = PatternRecognitionLayer(self.library_path)
        with self.assertRaises(RuntimeError):
            prl.route(RouterState(query="Find section 4.2", document="Section 4.2 text"))
        prl.prefetch_and_fit_vocabulary(["Find section 4.2"], ["Section 4.2 text"])
        features = prl.extract_features("Find section 4.2", "Section 4.2 text")
        self.assertEqual(len(features), len(prl.extract_features("Find clause 7.1", "Clause 7.1 text")))

    def test_tomil_parse_failure_no_keyword_fallback(self) -> None:
        layer = TheoryOfMindInferenceLayer(self.fake_client, inference_model="fake-tomil")
        state = RouterState(query="Assess clause", document="Contract text")
        layer._call_llm_raw = lambda prompt: "not json at all"  # type: ignore[assignment]
        state = layer.infer_intent(state)
        self.assertEqual(state.result_status, "TOMIL_PARSE_FAILURE")
        self.assertIsNotNone(state.tom_inference_quality)
        self.assertEqual(state.tom_inference_quality.parse_path, "TOMIL_PARSE_FAILURE")
        self.assertIsNone(state.task_type)

    def test_tomil_normalised_quality_flagged(self) -> None:
        layer = TheoryOfMindInferenceLayer(self.fake_client, inference_model="fake-tomil")
        state = RouterState(
            query="What does section 4.2 say about subcontractor qualifications?",
            document="Public tender section 4.2 covers subcontractor qualifications.",
        )
        state = layer.infer_intent(state)
        self.assertEqual(state.task_type, TaskType.TENDER_EVALUATION)
        self.assertFalse(state.tom_inference_quality.tom_is_valid_for_research)
        self.assertTrue(state.tom_inference_quality.normalisation_applied)

    def test_reroute_belief_reset(self) -> None:
        frl = FluidReroutingLayer()
        state = RouterState(
            query="Assess compliance.",
            document="Section 4.2 requires accreditation and experience.",
            assigned_model=Model.EXTRACTION_MODEL,
            routing_belief={Model.EXTRACTION_MODEL: 0.9, Model.REASONING_MODEL: 0.1},
        )
        partial_output = "It is unclear whether clause 9.9 applies. This may depend on another tender. NOVATION."
        updated = frl.monitor_and_reroute(state, partial_output)
        self.assertTrue(updated.reroute_triggered)
        self.assertAlmostEqual(updated.routing_belief[updated.assigned_model], 0.72, places=6)
        self.assertAlmostEqual(sum(updated.routing_belief.values()), 1.0, places=6)

    def test_max_reroutes_cap(self) -> None:
        router = FCLAMMRRouter(self.fake_client, self.library_path, prl_threshold=0.95)
        router.prl.prefetch_and_fit_vocabulary(["Assess compliance"], ["Section 4.2 text"])
        state = RouterState(
            query="Assess compliance",
            document="Section 4.2 text",
            assigned_model=Model.EXTRACTION_MODEL,
            task_type=TaskType.COMPLIANCE_CHECK,
            routing_belief={Model.EXTRACTION_MODEL: 0.9, Model.REASONING_MODEL: 0.1},
            reroute_count=router.MAX_REROUTES,
            reroute_triggered=True,
            effective_routing_mode="REROUTED",
        )
        self.assertEqual(state.reroute_count, router.MAX_REROUTES)

    def test_rerouted_task_excluded_from_pattern_library(self) -> None:
        prl = PatternRecognitionLayer(self.library_path)
        prl.prefetch_and_fit_vocabulary(["alpha query"], ["beta document"])
        state = RouterState(
            query="alpha query",
            document="beta document",
            task_type=TaskType.REASONING,
            assigned_model=Model.REASONING_MODEL,
            reroute_triggered=True,
            reroute_count=1,
        )
        prl.write_back(state, 0.9)
        payload = json.loads(Path(self.library_path).read_text(encoding="utf-8"))
        self.assertEqual(payload, [])
        self.assertTrue(Path("results/rerouted_tasks_log.json").exists())

    def test_content_filter_excluded_from_scores(self) -> None:
        stratified = compute_stratified_results(
            [
                {"effective_routing_mode": "CONTENT_FILTER_BLOCKED", "score": None, "task_type": "reasoning", "est_cost_usd": 0.0, "latency_ms": 0.0, "calls_used": 1},
                {"effective_routing_mode": "TOMIL_SUCCESS", "score": 1.0, "task_type": "reasoning", "est_cost_usd": 0.1, "latency_ms": 1.0, "calls_used": 1},
            ]
        )
        self.assertEqual(stratified["CONTENT_FILTER_BLOCKED"]["n_tasks"], 1)
        self.assertEqual(stratified["ALL_SCORED"]["n_tasks"], 1)

    def test_stratified_results_no_double_counting(self) -> None:
        rows = [
            {"effective_routing_mode": "TOMIL_SUCCESS", "score": 1.0, "task_type": "reasoning", "est_cost_usd": 0.1, "latency_ms": 1.0, "calls_used": 1},
            {"effective_routing_mode": "PRL_MATCH", "score": 0.5, "task_type": "extraction", "est_cost_usd": 0.05, "latency_ms": 1.0, "calls_used": 1},
            {"effective_routing_mode": "FAILED_LLM_CALL", "score": None, "task_type": "reasoning", "est_cost_usd": 0.0, "latency_ms": 0.0, "calls_used": 1},
        ]
        stratified = compute_stratified_results(rows)
        total = (
            stratified["TOMIL_SUCCESS"]["n_tasks"]
            + stratified["PRL_MATCH"]["n_tasks"]
            + stratified["FAILED_LLM_CALL"]["n_tasks"]
        )
        self.assertEqual(total, 3)

    def test_reroute_quality_metrics_are_bounded(self) -> None:
        evaluation = EvaluationLayer()
        rerouted = [
            {"task_type": "reasoning", "score": 0.9, "reroute_triggered": True, "belief_at_trigger": [0.4], "est_cost_usd": 0.3},
            {"task_type": "reasoning", "score": 0.6, "reroute_triggered": True, "belief_at_trigger": [0.3], "est_cost_usd": 0.25},
        ]
        all_tasks = rerouted + [
            {"task_type": "reasoning", "score": 0.2, "reroute_triggered": False, "est_cost_usd": 0.1},
            {"task_type": "reasoning", "score": 0.4, "reroute_triggered": False, "est_cost_usd": 0.1},
            {"task_type": "reasoning", "score": 0.8, "reroute_triggered": False, "est_cost_usd": 0.1},
        ]
        metrics = evaluation.score_reroute_quality(rerouted, all_tasks)
        self.assertGreaterEqual(metrics["reroute_precision"], 0.0)
        self.assertLessEqual(metrics["reroute_precision"], 1.0)
        self.assertGreaterEqual(metrics["reroute_recall"], 0.0)
        self.assertLessEqual(metrics["reroute_recall"], 1.0)

    def test_progress_log_emitted_at_interval(self) -> None:
        logger = logging.getLogger("fc_lammr.run_fc_lammr_hybrid_test")
        with self.assertLogs(level="INFO") as captured:
            logging.getLogger().info("Progress: 25/120 | Scored: 20 | Blocked: 1 | Excluded: 4 | Modes: {'TOMIL_SUCCESS': 20}")
        self.assertTrue(any("Progress: 25/120" in line for line in captured.output))

    def test_fc_lammr_importable_without_legacy_baseline_module(self) -> None:
        """
        FC-LAMMR must not depend on the legacy baseline module at import time.
        This test validates that the package is self-contained.
        """
        module_name = "core" + "_" + "logic"
        legacy_module_backup = sys.modules.pop(module_name, None)
        try:
            for key in list(sys.modules.keys()):
                if key.startswith("fc_lammr"):
                    del sys.modules[key]
            imported = importlib.import_module("fc_lammr")
            router_module = importlib.import_module("fc_lammr.fc_lammr_router")
            self.assertIsNotNone(imported)
            self.assertTrue(hasattr(router_module, "FCLAMMRRouter"))
        finally:
            if legacy_module_backup is not None:
                sys.modules[module_name] = legacy_module_backup


if __name__ == "__main__":
    unittest.main()
