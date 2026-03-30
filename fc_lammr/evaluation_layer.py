"""Evaluation utilities for FC-LAMMR."""

from __future__ import annotations

from collections import defaultdict

from fc_lammr.data_structures import RouterState, TaskType
from fc_lammr.utils.text_processing import tokenise


class EvaluationLayer:
    """Task-aware scoring layer extended with reroute-quality metrics."""

    def _f_beta(self, predicted_tokens: list[str], truth_tokens: list[str], beta: float = 2.0) -> float:
        predicted_set = set(predicted_tokens)
        truth_set = set(truth_tokens)
        if not predicted_set and not truth_set:
            return 1.0
        if not predicted_set or not truth_set:
            return 0.0
        overlap = len(predicted_set & truth_set)
        precision = overlap / len(predicted_set)
        recall = overlap / len(truth_set)
        beta_sq = beta * beta
        denominator = (beta_sq * precision) + recall
        if denominator == 0:
            return 0.0
        return ((1 + beta_sq) * precision * recall) / denominator

    def score_extraction(self, predicted: str, ground_truth: str) -> dict:
        predicted_tokens = tokenise(predicted)
        truth_tokens = tokenise(ground_truth)
        predicted_set = set(predicted_tokens)
        truth_set = set(truth_tokens)
        union = predicted_set | truth_set
        jaccard = 1.0 if not union else len(predicted_set & truth_set) / len(union)
        f2 = self._f_beta(predicted_tokens, truth_tokens, beta=2.0)
        return {"jaccard": jaccard, "f2": f2, "combined": (0.4 * jaccard) + (0.6 * f2)}

    def score_reasoning(self, predicted: str, ground_truth: str, task_type: TaskType) -> dict:
        predicted_tokens = set(tokenise(predicted))
        truth_tokens = set(tokenise(ground_truth))
        correctness = 1.0 if not truth_tokens else len(predicted_tokens & truth_tokens) / len(truth_tokens)
        has_verdict = True
        if task_type in {TaskType.COMPLIANCE_CHECK, TaskType.TENDER_EVALUATION}:
            has_verdict = any(verdict in predicted.lower() for verdict in ["yes", "no", "partial"])
        combined = (0.8 * correctness) + (0.2 * float(has_verdict))
        return {"correctness": correctness, "has_verdict": has_verdict, "combined": combined}

    def score_reroute_quality(self, rerouted_tasks: list[dict], all_scored_tasks: list[dict]) -> dict:
        """
        Computes whether rerouting decisions were well-calibrated.
        """
        if not rerouted_tasks:
            return {
                "reroute_precision": 0.0,
                "reroute_recall": 0.0,
                "mean_reroute_belief_at_trigger": 0.0,
                "reroute_cost_overhead": 0.0,
            }
        by_task = defaultdict(list)
        for task in all_scored_tasks:
            if task.get("score") is not None:
                by_task[task.get("task_type", "unknown")].append(float(task["score"]))
        improved = 0
        trigger_beliefs = []
        cost_overheads = []
        for task in rerouted_tasks:
            baseline = by_task.get(task.get("task_type", "unknown"), [0.0])
            if float(task.get("score", 0.0)) >= (sum(baseline) / max(len(baseline), 1)):
                improved += 1
            trigger_beliefs.extend(task.get("belief_at_trigger", []))
            same_family_non_rerouted = [x for x in all_scored_tasks if x.get("task_type") == task.get("task_type") and not x.get("reroute_triggered")]
            if same_family_non_rerouted:
                avg_cost = sum(float(x.get("est_cost_usd", 0.0)) for x in same_family_non_rerouted) / len(same_family_non_rerouted)
                cost_overheads.append(float(task.get("est_cost_usd", 0.0)) - avg_cost)
        low_thresholds = {
            task_type: sorted(scores)[max(0, int(len(scores) * 0.25) - 1)] if scores else 0.0
            for task_type, scores in by_task.items()
        }
        poor_outputs = [task for task in all_scored_tasks if task.get("score") is not None and float(task["score"]) <= low_thresholds.get(task.get("task_type", "unknown"), 0.0)]
        rerouted_poor = [task for task in poor_outputs if task.get("reroute_triggered")]
        return {
            "reroute_precision": float(improved / len(rerouted_tasks)),
            "reroute_recall": float(len(rerouted_poor) / len(poor_outputs)) if poor_outputs else 0.0,
            "mean_reroute_belief_at_trigger": float(sum(trigger_beliefs) / len(trigger_beliefs)) if trigger_beliefs else 0.0,
            "reroute_cost_overhead": float(sum(cost_overheads) / len(cost_overheads)) if cost_overheads else 0.0,
        }

    def evaluate(self, state: RouterState, ground_truth: str) -> dict:
        if state.task_type in {TaskType.EXTRACTION, TaskType.CLAUSE_IDENTIFICATION}:
            task_metrics = self.score_extraction(state.final_output or "", ground_truth)
        else:
            task_metrics = self.score_reasoning(state.final_output or "", ground_truth, state.task_type or TaskType.REASONING)
        task_metrics.update(
            {
                "reroute_triggered": state.reroute_triggered,
                "risk_weight_applied": state.risk_weight,
            }
        )
        return task_metrics
