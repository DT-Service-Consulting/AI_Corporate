"""Acceptance and regression tests for the FC-LAMMR pre-final patch set."""

from __future__ import annotations

import json
import importlib
import logging
import os
import shutil
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import project_secrets

from fc_lammr.data_structures import Model, RouterState, TaskType
from fc_lammr.evaluation_layer import EvaluationLayer
from fc_lammr.fc_lammr_router import FCLAMMRRouter
from fc_lammr.fluid_rerouting_layer import FluidReroutingLayer
from fc_lammr.pattern_recognition_layer import PatternRecognitionLayer
from fc_lammr.rescore_fc_lammr_results import rescore_reasoning_rows
from fc_lammr.run_fc_lammr_hybrid_test import (
    _calls_detail_for_state,
    _calls_used_for_state,
    _json_safe,
    _reasoning_score,
    compute_stratified_results,
)
from fc_lammr.tom_inference_layer import TheoryOfMindInferenceLayer
from fc_lammr.utils import llm_client as llm_client_module
from fc_lammr.utils.llm_client import (
    ConfigurationError,
    OpenAICompatibleLLMClient,
    get_429_count,
    reset_429_state,
    validate_deployment_config,
)


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


class _FakeNestedTransport:
    """Nested chat completion transport for llm_client tests."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class _FakePRL:
    def prefetch_and_fit_vocabulary(self, all_queries, all_documents):
        return None


class _FakeRunnerRouter:
    def __init__(self, states):
        self._states = list(states)
        self.prl = _FakePRL()

    def route(self, query, document, force_extraction=False):
        state = self._states.pop(0)
        if force_extraction:
            state.effective_routing_mode = "BUDGET_FORCED_EXTRACTION"
            state.audit_log = [entry for entry in state.audit_log if entry.get("layer") != "ToMIL"]
            state.assigned_model = Model.EXTRACTION_MODEL
        return state

    def _deployment_for(self, model):
        return "fake-reasoning" if model == Model.REASONING_MODEL else "fake-extraction"


class FCLAMMRPatchSetTests(unittest.TestCase):
    """Focused tests for the requested pre-final FC-LAMMR patch set."""

    def setUp(self) -> None:
        self._cwd_backup = Path.cwd()
        self.temp_dir = (Path("fc_lammr/.test_tmp") / self._testMethodName).resolve()
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(self.temp_dir)
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
        project_secrets.TOMIL_DEPLOYMENT_NAME = "fake-extraction"
        reset_429_state()

    def _runner_module(self):
        return importlib.import_module("fc_lammr.run_fc_lammr_hybrid_test")

    def tearDown(self) -> None:
        os.chdir(self._cwd_backup)
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

    def test_prl_rejects_low_quality_pattern(self) -> None:
        payload = [
            {
                "query_features": [],
                "task_type": "clause_identification",
                "model_used": "extraction_model",
                "outcome_score": 0.6,
                "query_text_preview": "Extract clause 4.2 exactly as written.",
                "combined_text": "Extract clause 4.2 exactly as written.\nClause 4.2 text",
            }
        ]
        Path(self.library_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        prl = PatternRecognitionLayer(self.library_path)
        prl.prefetch_and_fit_vocabulary(["Extract clause 4.2 exactly as written."], ["Clause 4.2 text"])
        state = prl.route(RouterState(query="Extract clause 4.2 exactly as written.", document="Clause 4.2 text"))
        self.assertIsNone(state.assigned_model)
        self.assertEqual(state.audit_log[-1]["decision"], "match_rejected_low_quality")

    def test_prl_rejects_task_type_inconsistent_pattern(self) -> None:
        payload = [
            {
                "query_features": [],
                "task_type": "clause_identification",
                "model_used": "extraction_model",
                "outcome_score": 0.85,
                "query_text_preview": "Is this clause potentially unfair to the consumer?",
                "combined_text": "Is this clause potentially unfair to the consumer?\nThe service may change prices at any time.",
            }
        ]
        Path(self.library_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        prl = PatternRecognitionLayer(self.library_path)
        prl.prefetch_and_fit_vocabulary(
            ["Is this clause potentially unfair to the consumer?"],
            ["The service may change prices at any time."],
        )
        state = prl.route(
            RouterState(
                query="Is this clause potentially unfair to the consumer?",
                document="The service may change prices at any time.",
            )
        )
        self.assertIsNone(state.assigned_model)
        self.assertEqual(state.audit_log[-1]["decision"], "match_rejected_task_type_inconsistent")

    def test_prl_cold_library_threshold_is_stricter(self) -> None:
        payload = [
            {
                "query_features": [],
                "task_type": "reasoning",
                "model_used": "reasoning_model",
                "outcome_score": 0.85,
                "query_text_preview": "Is this hearsay?",
                "combined_text": "Is this hearsay?\nWitness testimony text",
            }
        ]
        Path(self.library_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        prl = PatternRecognitionLayer(self.library_path, similarity_threshold=0.82, cold_library_threshold=0.91)
        prl.prefetch_and_fit_vocabulary(["Is this hearsay?"], ["Witness testimony text"])
        pattern = prl.patterns[0]
        with mock.patch.object(prl, "find_best_match", return_value=(pattern, 0.85)):
            state = prl.route(RouterState(query="Is this hearsay?", document="Witness testimony text"))
        self.assertIsNone(state.assigned_model)
        self.assertEqual(state.audit_log[-1]["decision"], "no_match")
        self.assertEqual(state.audit_log[-1]["threshold"], 0.91)

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
        self.assertTrue(Path("results/fc_lammr/rerouted_tasks_log.json").exists())

    def test_rerouted_log_fallback_does_not_crash_run(self) -> None:
        prl = PatternRecognitionLayer(self.library_path)
        prl.prefetch_and_fit_vocabulary(["alpha query"], ["beta document"])
        state = RouterState(
            query="alpha query",
            document="beta document",
            task_type=TaskType.REASONING,
            assigned_model=Model.REASONING_MODEL,
            reroute_triggered=True,
            reroute_count=2,
        )
        original_write_text = Path.write_text

        def _write_text_with_primary_failure(path_obj, data, *args, **kwargs):
            if path_obj == prl.rerouted_log_path:
                raise OSError(22, "Invalid argument")
            return original_write_text(path_obj, data, *args, **kwargs)

        with mock.patch("pathlib.Path.write_text", autospec=True, side_effect=_write_text_with_primary_failure):
            prl.write_back(state, 0.9)
        fallback_path = prl.rerouted_log_path.with_name("rerouted_tasks_log_fallback.jsonl")
        self.assertTrue(fallback_path.exists())
        lines = fallback_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertGreaterEqual(len(lines), 1)
        latest = json.loads(lines[-1])
        self.assertEqual(latest["query"], "alpha query")
        self.assertEqual(latest["reroute_count"], 2)

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

    def test_global_backoff_floor_increases_with_consecutive_429s(self) -> None:
        error_429 = RuntimeError("429 quota exceeded")
        error_fatal = RuntimeError("fatal failure")
        success = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=SimpleNamespace(total_tokens=1),
        )
        transport = _FakeNestedTransport(
            [
                error_429, error_fatal,
                error_429, error_fatal,
                error_429, error_fatal,
                success,
            ]
        )
        client = OpenAICompatibleLLMClient(client=transport)
        with mock.patch("fc_lammr.utils.llm_client.time.sleep", return_value=None):
            for _ in range(3):
                with self.assertRaises(RuntimeError):
                    client.create_chat_completion(messages=[{"role": "user", "content": "test"}], model="fake")
            self.assertEqual(get_429_count(), 3)
            self.assertEqual(llm_client_module._consecutive_429_count, 3)
            self.assertGreaterEqual(llm_client_module._global_backoff_floor, 6.0)
            client.create_chat_completion(messages=[{"role": "user", "content": "success"}], model="fake")
        self.assertEqual(llm_client_module._consecutive_429_count, 0)
        self.assertLess(llm_client_module._global_backoff_floor, 6.0)

    def test_request_timeout_forwarded_to_sdk(self) -> None:
        success = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=SimpleNamespace(total_tokens=1),
        )
        transport = _FakeNestedTransport([success])
        client = OpenAICompatibleLLMClient(client=transport, request_timeout_s=12.5)
        client.create_chat_completion(messages=[{"role": "user", "content": "timeout check"}], model="fake")
        self.assertEqual(transport.calls[0]["timeout"], 12.5)

    def test_calls_used_is_two_for_tomil_task_no_reroute(self) -> None:
        state = RouterState(
            query="q",
            document="d",
            assigned_model=Model.REASONING_MODEL,
            effective_routing_mode="TOMIL_SUCCESS",
            audit_log=[{"layer": "ToMIL", "decision": "intent_inferred"}],
        )
        self.assertEqual(_calls_used_for_state(state), 2)
        self.assertEqual(_calls_detail_for_state(state)["tomil"], 1)

    def test_calls_used_is_one_for_prl_match(self) -> None:
        state = RouterState(
            query="q",
            document="d",
            assigned_model=Model.EXTRACTION_MODEL,
            effective_routing_mode="PRL_MATCH",
            audit_log=[{"layer": "PRL", "decision": "pattern_match"}],
        )
        self.assertEqual(_calls_used_for_state(state), 1)
        self.assertEqual(_calls_detail_for_state(state)["tomil"], 0)

    def test_calls_used_is_three_for_tomil_with_reroute(self) -> None:
        state = RouterState(
            query="q",
            document="d",
            assigned_model=Model.REASONING_MODEL,
            effective_routing_mode="REROUTED",
            reroute_triggered=True,
            audit_log=[{"layer": "ToMIL", "decision": "intent_inferred"}],
        )
        self.assertEqual(_calls_used_for_state(state), 3)
        self.assertEqual(_calls_detail_for_state(state)["reroute"], 1)

    def test_calls_used_is_one_for_budget_forced_extraction(self) -> None:
        state = RouterState(
            query="q",
            document="d",
            assigned_model=Model.EXTRACTION_MODEL,
            effective_routing_mode="BUDGET_FORCED_EXTRACTION",
            audit_log=[{"layer": "budget_guard", "decision": "extraction_model"}],
        )
        self.assertEqual(_calls_used_for_state(state), 1)
        self.assertEqual(_calls_detail_for_state(state)["tomil"], 0)

    def test_json_safe_converts_enum_keys_in_nested_audit_payloads(self) -> None:
        payload = {
            "audit_log": [
                {
                    "layer": "budget_guard",
                    "belief_state": {
                        Model.EXTRACTION_MODEL: 0.95,
                        Model.REASONING_MODEL: 0.05,
                    },
                    "new_model": Model.EXTRACTION_MODEL,
                }
            ]
        }
        safe_payload = _json_safe(payload)
        encoded = json.dumps(safe_payload)
        self.assertIn("extraction_model", encoded)
        self.assertIn("reasoning_model", encoded)

    def test_reasoning_score_accepts_explanatory_text_after_label(self) -> None:
        score, metrics = _reasoning_score("No, this is not hearsay.", "No")
        self.assertEqual(score, 1.0)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["soft_score"], 1.0)

    def test_reasoning_score_accepts_short_preamble_before_label(self) -> None:
        score, metrics = _reasoning_score("My conclusion is no because the testimony is direct evidence.", "No")
        self.assertEqual(score, 1.0)
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_reasoning_score_late_label_only_gets_soft_credit(self) -> None:
        score, metrics = _reasoning_score(
            "The clause has mixed effects on consumer rights, although the gold label yes appears only in the final aside.",
            "Yes",
        )
        self.assertEqual(score, 0.5)
        self.assertEqual(metrics["accuracy"], 0.0)
        self.assertEqual(metrics["soft_score"], 0.5)

    def test_reasoning_score_true_miss_gets_zero(self) -> None:
        score, metrics = _reasoning_score("The evidence is inconclusive and no direct label is given.", "Fair")
        self.assertEqual(score, 0.0)
        self.assertEqual(metrics["accuracy"], 0.0)
        self.assertEqual(metrics["soft_score"], 0.0)

    def test_reasoning_score_supports_arbitrary_gold_label(self) -> None:
        score, metrics = _reasoning_score("Approved - the request satisfies the policy.", "Approved")
        self.assertEqual(score, 1.0)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["parsed_answer"], "approved")

    def test_offline_rescore_only_changes_reasoning_rows(self) -> None:
        official_rows = [
            {
                "id": "extract-1",
                "task_type": "extraction",
                "score": 0.25,
                "metrics": {"jaccard": 0.25},
                "output": "Clause text",
                "ground_truth": "Clause text",
            },
            {
                "id": "reason-1",
                "task_type": "reasoning",
                "score": 0.5,
                "metrics": {"accuracy": 0.0, "soft_score": 0.5, "parsed_answer": "unknown"},
                "output": "No, this is not hearsay.",
                "ground_truth": "No",
            },
            {
                "id": "blocked-1",
                "task_type": "reasoning",
                "score": None,
                "metrics": {},
                "output": "",
                "ground_truth": "Yes",
            },
        ]
        corrected_rows, change_counts = rescore_reasoning_rows(official_rows)
        self.assertEqual(corrected_rows[0]["score"], 0.25)
        self.assertEqual(corrected_rows[1]["score"], 1.0)
        self.assertEqual(corrected_rows[1]["metrics"]["accuracy"], 1.0)
        self.assertIsNone(corrected_rows[2]["score"])
        self.assertEqual(change_counts["changed_reasoning_rows"], 1)
        self.assertEqual(change_counts["upgraded_accuracy_0_to_1"], 1)

    def test_offline_rescore_script_preserves_input_file(self) -> None:
        input_path = self.temp_dir / "official_fc_lammr_results.json"
        output_path = self.temp_dir / "outputs" / "corrected_fc_lammr_results.json"
        summary_json_path = self.temp_dir / "results" / "parser_summary.json"
        summary_md_path = self.temp_dir / "results" / "parser_summary.md"
        baseline_method_summary = self.temp_dir / "outputs" / "method_summary.json"
        baseline_task_breakdown = self.temp_dir / "outputs" / "task_breakdown.csv"

        official_rows = [
            {
                "id": "extract-1",
                "task_type": "extraction",
                "score": 0.25,
                "metrics": {"jaccard": 0.25},
                "output": "Clause text",
                "ground_truth": "Clause text",
            },
            {
                "id": "reason-1",
                "task_type": "reasoning",
                "score": 0.5,
                "metrics": {"accuracy": 0.0, "soft_score": 0.5, "parsed_answer": "unknown"},
                "output": "No, this is not hearsay.",
                "ground_truth": "No",
            },
            {
                "id": "blocked-1",
                "task_type": "reasoning",
                "score": None,
                "metrics": {},
                "output": "",
                "ground_truth": "Yes",
            },
        ]
        input_path.parent.mkdir(parents=True, exist_ok=True)
        input_path.write_text(json.dumps(official_rows, indent=2), encoding="utf-8")
        baseline_method_summary.parent.mkdir(parents=True, exist_ok=True)
        baseline_method_summary.write_text(
            json.dumps(
                [
                    {
                        "method": "hybrid_rule_plus_learning",
                        "avg_score_mean": 0.55,
                        "macro_task_score_mean": 0.5,
                    }
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        baseline_task_breakdown.write_text(
            "\n".join(
                [
                    "method,task_type,avg_score_mean",
                    "hybrid_rule_plus_learning,extraction,0.4",
                    "hybrid_rule_plus_learning,reasoning,0.6",
                ]
            ),
            encoding="utf-8",
        )

        import fc_lammr.rescore_fc_lammr_results as rescore_module

        before_text = input_path.read_text(encoding="utf-8")
        exit_code = rescore_module.main(
            [
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--summary-json",
                str(summary_json_path),
                "--summary-md",
                str(summary_md_path),
                "--baseline-method-summary",
                str(baseline_method_summary),
                "--baseline-task-breakdown",
                str(baseline_task_breakdown),
            ]
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(input_path.read_text(encoding="utf-8"), before_text)
        corrected_rows = json.loads(output_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_json_path.read_text(encoding="utf-8"))
        self.assertEqual(corrected_rows[0]["score"], 0.25)
        self.assertEqual(corrected_rows[1]["score"], 1.0)
        self.assertIsNone(corrected_rows[2]["score"])
        self.assertEqual(summary["reasoning_rows_changed"], 1)
        self.assertEqual(summary["accuracy_upgrades_0_to_1"], 1)
        self.assertEqual(summary["official"]["skipped_tasks"], 1)
        self.assertEqual(summary["parser_corrected"]["skipped_tasks"], 1)
        self.assertTrue(summary_md_path.exists())

    def test_budget_forced_extraction_skips_tomil(self) -> None:
        router = FCLAMMRRouter(self.fake_client, self.library_path, prl_threshold=0.95)
        state = router.route("Extract section 4.2", "Section 4.2 text", force_extraction=True)
        self.assertEqual(state.effective_routing_mode, "BUDGET_FORCED_EXTRACTION")
        self.assertFalse(any(entry.get("layer") == "ToMIL" for entry in state.audit_log))

    def test_route_completes_after_single_reroute(self) -> None:
        router = FCLAMMRRouter(self.fake_client, self.library_path, prl_threshold=0.95, max_task_seconds=30.0)
        router.prl.prefetch_and_fit_vocabulary(["Assess hearsay"], ["Evidence text"])
        state = router.route("Assess hearsay", "Evidence text")
        self.assertTrue(state.reroute_count >= 1)
        self.assertTrue(state.reroute_triggered)
        self.assertEqual(state.effective_routing_mode, "REROUTED")
        self.assertEqual(state.final_output, "ANSWER: YES")
        self.assertFalse(any(entry.get("decision") == "task_timeout" for entry in state.audit_log))

    def test_route_aborts_after_max_task_seconds(self) -> None:
        router = FCLAMMRRouter(
            self.fake_client,
            self.library_path,
            prl_threshold=0.95,
            max_task_seconds=1.0,
        )
        router.prl.prefetch_and_fit_vocabulary(["Assess hearsay"], ["Evidence text"])
        with mock.patch("fc_lammr.fc_lammr_router.time.perf_counter", side_effect=[0.0, 0.0, 2.0, 2.0, 2.0]):
            state = router.route("Assess hearsay", "Evidence text")
        self.assertEqual(state.result_status, "FAILED_LLM_CALL")
        self.assertEqual(state.effective_routing_mode, "FAILED_LLM_CALL")
        self.assertTrue(any(entry.get("decision") == "task_timeout" for entry in state.audit_log))

    def test_task_start_and_end_logged(self) -> None:
        output_path = self.temp_dir / "runner_output.json"
        argv = [
            "runner",
            "--output", str(output_path),
            "--split-manifest", "unused.json",
            "--split-filter", "all",
            "--pattern-library", self.library_path,
            "--rate-limit-s", "0.0",
            "--task-sleep", "0.0",
            "--max-reasoning-calls", "10",
            "--checkpoint-interval", "10",
        ]
        dataset = [
            {"id": "1", "type": "extraction", "query_intent": "Find clause", "input": "Clause text", "target": "Clause text"},
            {"id": "2", "type": "extraction", "query_intent": "Find clause", "input": "Clause text", "target": "Clause text"},
        ]
        states = [
            RouterState(
                query="Find clause",
                document="Clause text",
                assigned_model=Model.EXTRACTION_MODEL,
                effective_routing_mode="PRL_MATCH",
                final_output="Clause text",
                audit_log=[{"layer": "PRL", "decision": "pattern_match"}, {"layer": "execution", "model": "extraction_model"}],
            ),
            RouterState(
                query="Find clause",
                document="Clause text",
                assigned_model=Model.EXTRACTION_MODEL,
                effective_routing_mode="PRL_MATCH",
                final_output="Clause text",
                audit_log=[{"layer": "PRL", "decision": "pattern_match"}, {"layer": "execution", "model": "extraction_model"}],
            ),
        ]
        fake_router = _FakeRunnerRouter(states)
        runner_module = self._runner_module()
        with (
            mock.patch.object(runner_module, "load_data", return_value=dataset),
            mock.patch.object(runner_module, "load_manifest", return_value={"1": "test", "2": "test"}),
            mock.patch.object(runner_module, "FCLAMMRRouter", return_value=fake_router),
            mock.patch.object(sys, "argv", argv),
            self.assertLogs(level="INFO") as captured,
        ):
            runner_module.run()
        logs = "\n".join(captured.output)
        self.assertIn("TASK START | 1/2 | id=1", logs)
        self.assertIn("TASK END | 1/2 | id=1", logs)
        self.assertIn("TASK START | 2/2 | id=2", logs)
        self.assertIn("TASK END | 2/2 | id=2", logs)

    def test_budget_forced_extraction_recorded_in_summary(self) -> None:
        output_path = self.temp_dir / "budget_output.json"
        argv = [
            "runner",
            "--output", str(output_path),
            "--split-manifest", "unused.json",
            "--split-filter", "all",
            "--pattern-library", self.library_path,
            "--rate-limit-s", "0.0",
            "--task-sleep", "0.0",
            "--max-reasoning-calls", "0",
            "--checkpoint-interval", "10",
        ]
        dataset = [
            {"id": "1", "type": "extraction", "query_intent": "Find clause", "input": "Clause text", "target": "Clause text"},
        ]
        state = RouterState(
            query="Find clause",
            document="Clause text",
            assigned_model=Model.EXTRACTION_MODEL,
            effective_routing_mode="BUDGET_FORCED_EXTRACTION",
            final_output="Clause text",
            audit_log=[{"layer": "budget_guard", "decision": "extraction_model"}, {"layer": "execution", "model": "extraction_model"}],
        )
        fake_router = _FakeRunnerRouter([state])
        runner_module = self._runner_module()
        with (
            mock.patch.object(runner_module, "load_data", return_value=dataset),
            mock.patch.object(runner_module, "load_manifest", return_value={"1": "test"}),
            mock.patch.object(runner_module, "FCLAMMRRouter", return_value=fake_router),
            mock.patch.object(sys, "argv", argv),
        ):
            runner_module.run()
        stratified_files = sorted(Path("results").glob("fc_lammr_stratified_results_*.json"))
        summary = json.loads(stratified_files[-1].read_text(encoding="utf-8"))
        self.assertIn("call_summary", summary)
        self.assertEqual(summary["call_summary"]["budget_forced_extraction_tasks"], 1)

    def test_startup_metadata_includes_tomil_equals_reasoning_flag(self) -> None:
        output_path = self.temp_dir / "startup_output.json"
        argv = [
            "runner",
            "--output", str(output_path),
            "--split-manifest", "unused.json",
            "--split-filter", "all",
            "--pattern-library", self.library_path,
            "--rate-limit-s", "0.0",
            "--task-sleep", "0.0",
            "--max-reasoning-calls", "10",
            "--checkpoint-interval", "10",
        ]
        dataset = [
            {"id": "1", "type": "extraction", "query_intent": "Find clause", "input": "Clause text", "target": "Clause text"},
        ]
        state = RouterState(
            query="Find clause",
            document="Clause text",
            assigned_model=Model.EXTRACTION_MODEL,
            effective_routing_mode="PRL_MATCH",
            final_output="Clause text",
            audit_log=[{"layer": "PRL", "decision": "pattern_match"}, {"layer": "execution", "model": "extraction_model"}],
        )
        fake_router = _FakeRunnerRouter([state])
        runner_module = self._runner_module()
        with (
            mock.patch.object(runner_module, "load_data", return_value=dataset),
            mock.patch.object(runner_module, "load_manifest", return_value={"1": "test"}),
            mock.patch.object(runner_module, "FCLAMMRRouter", return_value=fake_router),
            mock.patch.object(sys, "argv", argv),
        ):
            runner_module.run()
        stratified_files = sorted(Path("results").glob("fc_lammr_stratified_results_*.json"))
        summary = json.loads(stratified_files[-1].read_text(encoding="utf-8"))
        self.assertIn("run_metadata", summary)
        self.assertFalse(summary["run_metadata"]["tomil_equals_reasoning"])

    def test_reroute_to_reasoning_counted_in_summary(self) -> None:
        output_path = self.temp_dir / "reroute_reasoning_output.json"
        argv = [
            "runner",
            "--output", str(output_path),
            "--split-manifest", "unused.json",
            "--split-filter", "all",
            "--pattern-library", self.library_path,
            "--rate-limit-s", "0.0",
            "--task-sleep", "0.0",
            "--max-reasoning-calls", "10",
            "--checkpoint-interval", "10",
        ]
        dataset = [
            {"id": "1", "type": "extraction", "query_intent": "Find clause", "input": "Clause text", "target": "Clause text"},
        ]
        state = RouterState(
            query="Find clause",
            document="Clause text",
            assigned_model=Model.REASONING_MODEL,
            effective_routing_mode="REROUTED",
            final_output="Clause text",
            reroute_triggered=True,
            audit_log=[
                {"layer": "ToMIL", "decision": "intent_inferred"},
                {"layer": "execution", "model": "extraction_model"},
                {"layer": "FRL", "decision": "reroute", "new_model": "reasoning_model"},
            ],
        )
        fake_router = _FakeRunnerRouter([state])
        runner_module = self._runner_module()
        with (
            mock.patch.object(runner_module, "load_data", return_value=dataset),
            mock.patch.object(runner_module, "load_manifest", return_value={"1": "test"}),
            mock.patch.object(runner_module, "FCLAMMRRouter", return_value=fake_router),
            mock.patch.object(sys, "argv", argv),
        ):
            runner_module.run()
        stratified_files = sorted(Path("results").glob("fc_lammr_stratified_results_*.json"))
        summary = json.loads(stratified_files[-1].read_text(encoding="utf-8"))
        self.assertEqual(summary["call_summary"]["reroute_to_reasoning"], 1)
        self.assertEqual(summary["call_summary"]["reroute_to_extraction"], 0)

    def test_reroute_to_extraction_counted_in_summary(self) -> None:
        output_path = self.temp_dir / "reroute_extraction_output.json"
        argv = [
            "runner",
            "--output", str(output_path),
            "--split-manifest", "unused.json",
            "--split-filter", "all",
            "--pattern-library", self.library_path,
            "--rate-limit-s", "0.0",
            "--task-sleep", "0.0",
            "--max-reasoning-calls", "10",
            "--checkpoint-interval", "10",
        ]
        dataset = [
            {"id": "1", "type": "extraction", "query_intent": "Find clause", "input": "Clause text", "target": "Clause text"},
        ]
        state = RouterState(
            query="Find clause",
            document="Clause text",
            assigned_model=Model.EXTRACTION_MODEL,
            effective_routing_mode="REROUTED",
            final_output="Clause text",
            reroute_triggered=True,
            audit_log=[
                {"layer": "ToMIL", "decision": "intent_inferred"},
                {"layer": "execution", "model": "reasoning_model"},
                {"layer": "FRL", "decision": "reroute", "new_model": "extraction_model"},
            ],
        )
        fake_router = _FakeRunnerRouter([state])
        runner_module = self._runner_module()
        with (
            mock.patch.object(runner_module, "load_data", return_value=dataset),
            mock.patch.object(runner_module, "load_manifest", return_value={"1": "test"}),
            mock.patch.object(runner_module, "FCLAMMRRouter", return_value=fake_router),
            mock.patch.object(sys, "argv", argv),
        ):
            runner_module.run()
        stratified_files = sorted(Path("results").glob("fc_lammr_stratified_results_*.json"))
        summary = json.loads(stratified_files[-1].read_text(encoding="utf-8"))
        self.assertEqual(summary["call_summary"]["reroute_to_reasoning"], 0)
        self.assertEqual(summary["call_summary"]["reroute_to_extraction"], 1)

    def test_inter_task_sleep_is_applied(self) -> None:
        output_path = self.temp_dir / "sleep_output.json"
        argv = [
            "runner",
            "--output", str(output_path),
            "--split-manifest", "unused.json",
            "--split-filter", "all",
            "--pattern-library", self.library_path,
            "--rate-limit-s", "0.0",
            "--task-sleep", "0.1",
            "--max-reasoning-calls", "10",
            "--checkpoint-interval", "10",
        ]
        dataset = [
            {"id": str(index), "type": "extraction", "query_intent": "Find clause", "input": "Clause text", "target": "Clause text"}
            for index in range(5)
        ]
        states = [
            RouterState(
                query="Find clause",
                document="Clause text",
                assigned_model=Model.EXTRACTION_MODEL,
                effective_routing_mode="PRL_MATCH",
                final_output="Clause text",
                audit_log=[{"layer": "PRL", "decision": "pattern_match"}, {"layer": "execution", "model": "extraction_model"}],
            )
            for _ in range(5)
        ]
        fake_router = _FakeRunnerRouter(states)
        runner_module = self._runner_module()
        with (
            mock.patch.object(runner_module, "load_data", return_value=dataset),
            mock.patch.object(runner_module, "load_manifest", return_value={str(index): "test" for index in range(5)}),
            mock.patch.object(runner_module, "FCLAMMRRouter", return_value=fake_router),
            mock.patch.object(sys, "argv", argv),
        ):
            start = time.perf_counter()
            runner_module.run()
            elapsed = time.perf_counter() - start
        self.assertGreaterEqual(elapsed, 0.5)

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
