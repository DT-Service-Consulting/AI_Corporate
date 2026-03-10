import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from intelligent_router import IntelligentRouter


MAVERICK_NAME = "Llama-4-Maverick-17B-128E-Instruct-FP8"
LLAMA70_NAME = "Llama-3.3-70B-Instruct"
ERROR_OUTPUT_RE = re.compile(r"(?:azure error|error:\s*no client|internal_server_error|backend returned unexpected response)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Publication-focused evaluation, ablations, and error exports.")
    p.add_argument("--results", default="results.json")
    p.add_argument("--output-dir", default="outputs/research_eval")
    p.add_argument("--seeds", default="42,43,44,45,46")
    p.add_argument("--max-error-cases", type=int, default=40)
    p.add_argument("--bootstrap-samples", type=int, default=4000)
    p.add_argument("--ci-level", type=float, default=0.95)
    p.add_argument("--paired-alpha", type=float, default=0.05)
    p.add_argument("--reference-method", default="always_maverick")
    p.add_argument("--uncertainty-threshold", type=float, default=0.70)
    p.add_argument(
        "--dynamic-split",
        action="store_true",
        help="Ignore precomputed split labels and create per-seed stratified train/test splits.",
    )
    p.add_argument("--dynamic-test-size", type=float, default=0.2)
    p.add_argument("--recursive-max-depth", type=int, default=3)
    p.add_argument("--recursive-low-confidence", type=float, default=0.62)
    p.add_argument("--recursive-high-confidence", type=float, default=0.78)
    p.add_argument(
        "--task-balance-power",
        type=float,
        default=1.5,
        help="0 disables task reweighting. >0 upweights minority task types by inverse-frequency^power.",
    )
    p.add_argument(
        "--accuracy-priority",
        type=float,
        default=0.9,
        help="Weight for accuracy in composite ranking (0..1). Cost gets 1-weight.",
    )
    return p.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def infer_text_input(row: pd.Series) -> str:
    if "query" in row and isinstance(row["query"], str) and row["query"].strip():
        return row["query"]
    if "input" in row and isinstance(row["input"], str) and row["input"].strip():
        return row["input"]
    if "input" in row and isinstance(row["input"], dict):
        if row["task_type"] == "reasoning":
            return str(row["input"].get("question", ""))
        return str(row["input"])
    return f"{row.get('task_type', 'unknown')}:{row.get('doc_name', '')}"


def infer_doc_text(row: pd.Series) -> str:
    if "input" in row and isinstance(row["input"], str):
        return row["input"]
    if "input" in row and isinstance(row["input"], dict):
        facts = row["input"].get("facts", "")
        rule = row["input"].get("rule", "")
        return f"{rule}\n{facts}".strip()
    return ""


def load_pairs(results_path: str) -> pd.DataFrame:
    with open(results_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("results.json is empty.")
    needed = {"id", "model_name", "score", "task_type"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"results.json missing required columns: {missing}")

    if "split" not in df.columns:
        df["split"] = "all"
    if "query" not in df.columns:
        df["query"] = ""
    if "input" not in df.columns:
        df["input"] = ""
    if "est_cost_usd" not in df.columns:
        df["est_cost_usd"] = np.nan
    if "latency_ms" not in df.columns:
        df["latency_ms"] = np.nan
    if "full_output" not in df.columns:
        df["full_output"] = ""
    if "eval_status" not in df.columns:
        df["eval_status"] = ""

    if df["eval_status"].astype(str).str.len().gt(0).any():
        df["is_valid_eval"] = df["eval_status"].astype(str).str.lower().eq("ok")
    else:
        df["is_valid_eval"] = ~df["full_output"].astype(str).str.contains(ERROR_OUTPUT_RE, na=False)

    df_m = df[df["model_name"].str.contains("Maverick", case=False, na=False)].copy()
    df_l = df[df["model_name"].str.contains("70B", case=False, na=False)].copy()
    if df_m.empty or df_l.empty:
        raise RuntimeError("Need both Maverick and 70B rows in results.json.")

    keep_cols = ["id", "task_type", "split", "doc_name", "ground_truth", "query", "input"]
    for col in keep_cols:
        if col not in df_m.columns:
            df_m[col] = ""
        if col not in df_l.columns:
            df_l[col] = ""

    pm = df_m[keep_cols + ["score", "est_cost_usd", "latency_ms", "is_valid_eval"]].rename(
        columns={"score": "score_mav", "est_cost_usd": "cost_mav", "latency_ms": "latency_mav", "is_valid_eval": "valid_mav"}
    )
    pl = df_l[["id", "score", "est_cost_usd", "latency_ms", "is_valid_eval"]].rename(
        columns={"score": "score_70b", "est_cost_usd": "cost_70b", "latency_ms": "latency_70b", "is_valid_eval": "valid_70b"}
    )
    merged = pd.merge(pm, pl, on="id", how="inner")
    before = len(merged)
    merged = merged[(merged["valid_mav"]) & (merged["valid_70b"])].copy()
    dropped = before - len(merged)
    if dropped > 0:
        print(f"Dropped {dropped} invalid paired rows due to failed model calls.")
    if merged.empty:
        raise RuntimeError("No valid paired rows remain after filtering failed model calls.")
    merged["query_text"] = merged.apply(infer_text_input, axis=1)
    merged["doc_text"] = merged.apply(infer_doc_text, axis=1)
    merged["best_label"] = np.where(merged["score_70b"] > merged["score_mav"], LLAMA70_NAME, MAVERICK_NAME)
    return merged


def rule_decision(router: IntelligentRouter, query: str, doc_text: str, variant: str) -> str:
    q = query.lower()
    if variant == "no_keyword":
        if len(doc_text) > 15000:
            return MAVERICK_NAME
        return LLAMA70_NAME
    if variant == "no_length":
        if any(w in q for w in router.reasoning_keywords):
            return LLAMA70_NAME
        if any(w in q for w in router.extraction_keywords):
            return MAVERICK_NAME
        return LLAMA70_NAME
    if variant == "no_reasoning_override":
        if any(w in q for w in router.extraction_keywords):
            return MAVERICK_NAME
        if len(doc_text) > 15000:
            return MAVERICK_NAME
        return LLAMA70_NAME
    model, _ = router.route(query, doc_text)
    return model


def selected_score(row: pd.Series, model_name: str) -> float:
    return float(row["score_mav"] if model_name == MAVERICK_NAME else row["score_70b"])


def selected_cost(row: pd.Series, model_name: str) -> float:
    return float(row["cost_mav"] if model_name == MAVERICK_NAME else row["cost_70b"])


def selected_latency(row: pd.Series, model_name: str) -> float:
    return float(row["latency_mav"] if model_name == MAVERICK_NAME else row["latency_70b"])


def bootstrap_mean_ci(values: np.ndarray, n_samples: int, ci_level: float, seed: int) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_samples, arr.size))
    means = np.mean(arr[idx], axis=1)
    alpha = (1.0 - ci_level) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def paired_bootstrap_test(
    a: np.ndarray,
    b: np.ndarray,
    n_samples: int,
    ci_level: float,
    seed: int,
) -> Dict[str, float]:
    av = np.asarray(a, dtype=float)
    bv = np.asarray(b, dtype=float)
    valid = (~np.isnan(av)) & (~np.isnan(bv))
    av = av[valid]
    bv = bv[valid]
    if av.size == 0:
        return {"n": 0, "mean_diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p_value": float("nan")}
    diff = av - bv
    if diff.size == 1:
        return {"n": 1, "mean_diff": float(diff[0]), "ci_low": float(diff[0]), "ci_high": float(diff[0]), "p_value": 1.0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_samples, diff.size))
    boot = np.mean(diff[idx], axis=1)
    alpha = (1.0 - ci_level) / 2.0
    ci_low = float(np.quantile(boot, alpha))
    ci_high = float(np.quantile(boot, 1.0 - alpha))
    p_pos = float(np.mean(boot >= 0.0))
    p_neg = float(np.mean(boot <= 0.0))
    p_two_sided = float(min(1.0, 2.0 * min(p_pos, p_neg)))
    return {
        "n": int(diff.size),
        "mean_diff": float(np.mean(diff)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_two_sided,
    }


def evaluate_policy(df: pd.DataFrame, chooser, seed: int, n_bootstrap: int, ci_level: float):
    chosen = df.apply(lambda r: chooser(r), axis=1).values
    score_mav = df["score_mav"].astype(float).values
    score_70b = df["score_70b"].astype(float).values
    cost_mav = df["cost_mav"].astype(float).values
    cost_70b = df["cost_70b"].astype(float).values
    lat_mav = df["latency_mav"].astype(float).values
    lat_70b = df["latency_70b"].astype(float).values

    chose_70b_mask = chosen == LLAMA70_NAME
    scores = np.where(chose_70b_mask, score_70b, score_mav)
    costs = np.where(chose_70b_mask, cost_70b, cost_mav)
    lats = np.where(chose_70b_mask, lat_70b, lat_mav)

    task_type_scores = df.assign(chosen_score=scores).groupby("task_type")["chosen_score"].mean().to_dict()
    macro_task_score = float(np.mean(list(task_type_scores.values()))) if task_type_scores else 0.0
    ci_low, ci_high = bootstrap_mean_ci(scores, n_samples=n_bootstrap, ci_level=ci_level, seed=seed)
    return {
        "n": int(len(df)),
        "avg_score": float(np.mean(scores)) if scores.size else 0.0,
        "avg_score_ci_low": ci_low,
        "avg_score_ci_high": ci_high,
        "macro_task_score": macro_task_score,
        "avg_cost_usd": float(np.nanmean(costs)) if costs.size else 0.0,
        "avg_latency_ms": float(np.nanmean(lats)) if lats.size else 0.0,
        "usage_70b_pct": float(np.mean(chose_70b_mask.astype(float)) * 100.0) if chosen.size else 0.0,
    }, chosen, scores, costs, lats


def build_task_weights(task_types: pd.Series, power: float) -> np.ndarray:
    if power <= 0:
        return np.ones(len(task_types), dtype=float)
    freq = task_types.value_counts().to_dict()
    weights = np.array([(1.0 / max(1, int(freq.get(t, 1)))) ** power for t in task_types.tolist()], dtype=float)
    weights = weights / np.mean(weights)
    return weights


def train_learning_router(df_train: pd.DataFrame, seed: int, task_balance_power: float):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000)
    X_train_txt = vec.fit_transform(df_train["query_text"].astype(str).tolist())
    X_train_num = np.c_[
        df_train["query_text"].astype(str).str.len().values,
        df_train["doc_text"].astype(str).str.len().values,
    ]
    X_train = hstack([X_train_txt, X_train_num])
    y_train = np.where(df_train["best_label"] == MAVERICK_NAME, 1, 0)
    sample_weight = build_task_weights(df_train["task_type"], power=task_balance_power)

    clf = RandomForestClassifier(n_estimators=250, max_depth=12, random_state=seed, class_weight="balanced")
    clf.fit(X_train, y_train, sample_weight=sample_weight)
    return vec, clf


def predict_learning_router(df: pd.DataFrame, vec: TfidfVectorizer, clf: RandomForestClassifier) -> np.ndarray:
    X_txt = vec.transform(df["query_text"].astype(str).tolist())
    X_num = np.c_[
        df["query_text"].astype(str).str.len().values,
        df["doc_text"].astype(str).str.len().values,
    ]
    X = hstack([X_txt, X_num])
    pred = clf.predict(X)
    return np.where(pred == 1, MAVERICK_NAME, LLAMA70_NAME)


def recursive_router_decision(
    row: pd.Series,
    idx: int,
    learn_pred: np.ndarray,
    learn_conf: np.ndarray,
    router: IntelligentRouter,
    max_depth: int,
    low_conf: float,
    high_conf: float,
) -> str:
    """
    Recursive routing policy over existing signals.
    Starts from learned prediction, then recursively refines with rule signals.
    """
    q = str(row["query_text"]).lower()
    doc_len = len(str(row["doc_text"]))
    pred = str(learn_pred[idx])
    conf = float(learn_conf[idx])
    depth = 0

    while depth < max_depth:
        if conf >= high_conf:
            break

        has_reasoning_kw = any(w in q for w in router.reasoning_keywords)
        has_extraction_kw = any(w in q for w in router.extraction_keywords)
        long_doc = doc_len > 15000

        if conf < low_conf:
            if long_doc or has_extraction_kw:
                pred = MAVERICK_NAME
                conf = max(conf, 0.66)
            elif has_reasoning_kw:
                pred = LLAMA70_NAME
                conf = max(conf, 0.66)
            else:
                pred = MAVERICK_NAME
                conf = max(conf, 0.64)
        else:
            rule_pred = rule_decision(router, row["query_text"], row["doc_text"], variant="full")
            if rule_pred == pred:
                conf = min(0.99, conf + 0.08)
            else:
                if long_doc:
                    pred = MAVERICK_NAME
                    conf = max(conf, 0.70)
                else:
                    conf = min(0.99, conf + 0.03)

        depth += 1

    return pred


def by_task_breakdown(df: pd.DataFrame, chooser) -> List[Dict]:
    rows = []
    for t, g in df.groupby("task_type"):
        metrics, _, _, _, _ = evaluate_policy(g, chooser, seed=0, n_bootstrap=1000, ci_level=0.95)
        metrics["task_type"] = t
        rows.append(metrics)
    return rows


def collect_error_cases(df: pd.DataFrame, chooser, max_cases: int) -> List[Dict]:
    rows = []
    for _, r in df.iterrows():
        chosen = chooser(r)
        chosen_score = selected_score(r, chosen)
        oracle = max(float(r["score_mav"]), float(r["score_70b"]))
        regret = float(oracle - chosen_score)
        rows.append(
            {
                "id": r["id"],
                "task_type": r["task_type"],
                "split": r["split"],
                "doc_name": r.get("doc_name", ""),
                "query_text": r["query_text"],
                "chosen_model": chosen,
                "score_mav": float(r["score_mav"]),
                "score_70b": float(r["score_70b"]),
                "chosen_score": chosen_score,
                "oracle_score": oracle,
                "regret": regret,
            }
        )
    rows.sort(key=lambda x: x["regret"], reverse=True)
    return rows[:max_cases]


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    df = load_pairs(args.results)

    router = IntelligentRouter()
    all_results = []
    task_rows = []
    instance_rows = []
    seed_values = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    for seed in seed_values:
        if args.dynamic_split:
            train_df, eval_df = train_test_split(
                df,
                test_size=float(args.dynamic_test_size),
                random_state=seed,
                stratify=df["task_type"],
            )
            train_df = train_df.copy()
            eval_df = eval_df.copy()
        else:
            test_df = df[df["split"] == "test"].copy()
            if test_df.empty:
                test_df = df.copy()
            train_df = df[df["split"].isin(["train", "val"])].copy()
            if train_df.empty:
                train_df, _ = train_test_split(df, test_size=0.3, random_state=seed, stratify=df["task_type"])
            eval_df = test_df

        vec, clf = train_learning_router(train_df, seed=seed, task_balance_power=args.task_balance_power)
        learn_pred = predict_learning_router(eval_df, vec, clf)
        learn_prob = clf.predict_proba(
            hstack(
                [
                    vec.transform(eval_df["query_text"].astype(str).tolist()),
                    np.c_[
                        eval_df["query_text"].astype(str).str.len().values,
                        eval_df["doc_text"].astype(str).str.len().values,
                    ],
                ]
            )
        )
        # Robust handling for single-class training folds.
        # If only one class is seen in training, predict_proba has one column.
        if learn_prob.ndim != 2 or learn_prob.shape[1] == 0:
            learn_mav_prob = np.where(learn_pred == MAVERICK_NAME, 1.0, 0.0)
        elif learn_prob.shape[1] == 1:
            only_cls = int(clf.classes_[0])
            if only_cls == 1:
                learn_mav_prob = np.ones(len(eval_df), dtype=float)
            else:
                learn_mav_prob = np.zeros(len(eval_df), dtype=float)
        else:
            # Map probability columns to class labels to avoid hardcoded index assumptions.
            class_to_idx = {int(c): i for i, c in enumerate(clf.classes_)}
            m_idx = class_to_idx.get(1, None)
            if m_idx is None:
                learn_mav_prob = np.where(learn_pred == MAVERICK_NAME, 1.0, 0.0)
            else:
                learn_mav_prob = learn_prob[:, m_idx]
        learn_conf = np.maximum(learn_mav_prob, 1.0 - learn_mav_prob)
        label_true = np.where(eval_df["best_label"] == MAVERICK_NAME, 1, 0)
        label_pred = np.where(learn_pred == MAVERICK_NAME, 1, 0)
        ml_accuracy = float(accuracy_score(label_true, label_pred))

        train_task_pref = {}
        for task_type, g in train_df.groupby("task_type"):
            train_task_pref[task_type] = (
                LLAMA70_NAME if float(g["score_70b"].mean()) >= float(g["score_mav"].mean()) else MAVERICK_NAME
            )
        global_pref = LLAMA70_NAME if float(train_df["score_70b"].mean()) >= float(train_df["score_mav"].mean()) else MAVERICK_NAME
        rng = np.random.default_rng(seed)

        methods = {
            "always_maverick": lambda r: MAVERICK_NAME,
            "always_70b": lambda r: LLAMA70_NAME,
            "rule_only": lambda r: rule_decision(router, r["query_text"], r["doc_text"], variant="full"),
            "rule_no_keyword": lambda r: rule_decision(router, r["query_text"], r["doc_text"], variant="no_keyword"),
            "rule_no_length": lambda r: rule_decision(router, r["query_text"], r["doc_text"], variant="no_length"),
            "rule_no_reasoning_override": lambda r: rule_decision(
                router, r["query_text"], r["doc_text"], variant="no_reasoning_override"
            ),
            "learning_only": lambda r, p=learn_pred: p[eval_df.index.get_loc(r.name)],
            "learning_uncertainty_fallback_maverick": lambda r, p=learn_pred, c=learn_conf: (
                MAVERICK_NAME
                if float(c[eval_df.index.get_loc(r.name)]) < float(args.uncertainty_threshold)
                else p[eval_df.index.get_loc(r.name)]
            ),
            "learning_uncertainty_fallback_70b": lambda r, p=learn_pred, c=learn_conf: (
                LLAMA70_NAME
                if float(c[eval_df.index.get_loc(r.name)]) < float(args.uncertainty_threshold)
                else p[eval_df.index.get_loc(r.name)]
            ),
            "task_prior_router": lambda r, pref=train_task_pref, gp=global_pref: pref.get(r["task_type"], gp),
            "random_50_50": lambda r, pr=rng: (LLAMA70_NAME if float(pr.random()) >= 0.5 else MAVERICK_NAME),
            "hybrid_rule_plus_learning": lambda r, p=learn_pred: (
                LLAMA70_NAME
                if any(w in r["query_text"].lower() for w in router.reasoning_keywords)
                else p[eval_df.index.get_loc(r.name)]
            ),
            "recursive_routing_policy": lambda r, p=learn_pred, c=learn_conf: recursive_router_decision(
                r,
                eval_df.index.get_loc(r.name),
                p,
                c,
                router,
                max_depth=max(1, int(args.recursive_max_depth)),
                low_conf=float(args.recursive_low_confidence),
                high_conf=float(args.recursive_high_confidence),
            ),
            "oracle_upper_bound": lambda r: LLAMA70_NAME if float(r["score_70b"]) >= float(r["score_mav"]) else MAVERICK_NAME,
        }

        for name, chooser in methods.items():
            summary, chosen_models, scores, costs, lats = evaluate_policy(
                eval_df,
                chooser,
                seed=seed,
                n_bootstrap=args.bootstrap_samples,
                ci_level=args.ci_level,
            )
            summary.update({"method": name, "seed": seed, "ml_router_label_acc": ml_accuracy})
            all_results.append(summary)
            oracle_scores = np.maximum(eval_df["score_mav"].astype(float).values, eval_df["score_70b"].astype(float).values)
            regrets = oracle_scores - scores
            for idx, row in enumerate(eval_df.itertuples(index=False)):
                instance_rows.append(
                    {
                        "seed": seed,
                        "id": row.id,
                        "task_type": row.task_type,
                        "split": row.split,
                        "method": name,
                        "chosen_model": str(chosen_models[idx]),
                        "score": float(scores[idx]),
                        "cost_usd": float(costs[idx]),
                        "latency_ms": float(lats[idx]),
                        "oracle_score": float(oracle_scores[idx]),
                        "regret_to_oracle": float(regrets[idx]),
                    }
                )
            for tr in by_task_breakdown(eval_df, chooser):
                tr.update({"method": name, "seed": seed})
                task_rows.append(tr)

        error_cases = collect_error_cases(eval_df, methods["hybrid_rule_plus_learning"], args.max_error_cases)
        with open(os.path.join(args.output_dir, f"error_cases_seed_{seed}.json"), "w", encoding="utf-8") as f:
            json.dump(error_cases, f, indent=2, ensure_ascii=False)

    summary_df = pd.DataFrame(all_results)
    instance_df = pd.DataFrame(instance_rows)
    agg = (
        summary_df.groupby("method")[
            [
                "avg_score",
                "avg_score_ci_low",
                "avg_score_ci_high",
                "macro_task_score",
                "avg_cost_usd",
                "avg_latency_ms",
                "usage_70b_pct",
                "ml_router_label_acc",
            ]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg.columns = ["_".join(c).strip("_") for c in agg.columns.to_flat_index()]
    # Accuracy-first ranking metric. Cost is retained as secondary signal.
    w = max(0.0, min(1.0, float(args.accuracy_priority)))
    score_norm = (agg["avg_score_mean"] - agg["avg_score_mean"].min()) / (
        (agg["avg_score_mean"].max() - agg["avg_score_mean"].min()) + 1e-9
    )
    cost_norm = (agg["avg_cost_usd_mean"] - agg["avg_cost_usd_mean"].min()) / (
        (agg["avg_cost_usd_mean"].max() - agg["avg_cost_usd_mean"].min()) + 1e-9
    )
    agg["rank_composite"] = (w * score_norm) - ((1.0 - w) * cost_norm)
    agg = agg.sort_values(
        by=["avg_score_mean", "rank_composite", "avg_cost_usd_mean"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    task_df = pd.DataFrame(task_rows)
    task_agg = (
        task_df.groupby(["method", "task_type"])[["avg_score", "avg_cost_usd", "avg_latency_ms", "usage_70b_pct"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    task_agg.columns = ["_".join(c).strip("_") for c in task_agg.columns.to_flat_index()]
    task_agg = task_agg.sort_values(
        by=["task_type", "avg_score_mean", "avg_cost_usd_mean"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    oracle_gap_df = (
        instance_df.groupby("method")[["score", "oracle_score", "regret_to_oracle"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    oracle_gap_df.columns = ["_".join(c).strip("_") for c in oracle_gap_df.columns.to_flat_index()]
    ci_rows = []
    for m, g in instance_df.groupby("method"):
        low, high = bootstrap_mean_ci(
            g["regret_to_oracle"].astype(float).values,
            n_samples=args.bootstrap_samples,
            ci_level=args.ci_level,
            seed=1337,
        )
        ci_rows.append({"method": m, "regret_ci_low": low, "regret_ci_high": high})
    oracle_gap_df = oracle_gap_df.merge(pd.DataFrame(ci_rows), on="method", how="left")
    oracle_gap_df = oracle_gap_df.sort_values(by=["regret_to_oracle_mean", "score_mean"], ascending=[True, False]).reset_index(drop=True)

    sig_rows = []
    ref_method = args.reference_method
    for method in sorted(instance_df["method"].unique()):
        if method == ref_method:
            continue
        pair = instance_df[instance_df["method"].isin([ref_method, method])].copy()
        pivot = pair.pivot_table(index=["seed", "id"], columns="method", values="score", aggfunc="first").dropna()
        if ref_method not in pivot.columns or method not in pivot.columns:
            continue
        test = paired_bootstrap_test(
            pivot[method].astype(float).values,
            pivot[ref_method].astype(float).values,
            n_samples=args.bootstrap_samples,
            ci_level=args.ci_level,
            seed=2026,
        )
        test.update({"method_a": method, "method_b": ref_method, "significant_at_alpha": bool(test["p_value"] < args.paired_alpha)})
        sig_rows.append(test)
    if sig_rows:
        sig_df = pd.DataFrame(sig_rows).sort_values(by=["p_value", "mean_diff"], ascending=[True, False]).reset_index(drop=True)
    else:
        sig_df = pd.DataFrame(columns=["method_a", "method_b", "n", "mean_diff", "ci_low", "ci_high", "p_value", "significant_at_alpha"])

    # Family-specific significance: compare methods on extraction and reasoning separately.
    fam_rows = []
    for fam in sorted(instance_df["task_type"].unique()):
        fam_df = instance_df[instance_df["task_type"] == fam].copy()
        if fam_df.empty:
            continue
        for method in sorted(fam_df["method"].unique()):
            if method == ref_method:
                continue
            pair = fam_df[fam_df["method"].isin([ref_method, method])].copy()
            pivot = pair.pivot_table(index=["seed", "id"], columns="method", values="score", aggfunc="first").dropna()
            if ref_method not in pivot.columns or method not in pivot.columns:
                continue
            test = paired_bootstrap_test(
                pivot[method].astype(float).values,
                pivot[ref_method].astype(float).values,
                n_samples=args.bootstrap_samples,
                ci_level=args.ci_level,
                seed=2026,
            )
            test.update(
                {
                    "task_type": fam,
                    "method_a": method,
                    "method_b": ref_method,
                    "significant_at_alpha": bool(test["p_value"] < args.paired_alpha),
                }
            )
            fam_rows.append(test)
    fam_sig_df = (
        pd.DataFrame(fam_rows)
        .sort_values(by=["task_type", "p_value", "mean_diff"], ascending=[True, True, False])
        .reset_index(drop=True)
        if fam_rows
        else pd.DataFrame(columns=["task_type", "method_a", "method_b", "n", "mean_diff", "ci_low", "ci_high", "p_value", "significant_at_alpha"])
    )

    task_counts = df["task_type"].value_counts().to_dict()
    min_task_count = int(min(task_counts.values())) if task_counts else 0
    imbalance_report = {
        "task_counts": task_counts,
        "min_task_count": min_task_count,
        "imbalance_warning": bool(min_task_count < 30),
        "note": "A task type with n < 30 has fragile uncertainty estimates; report claims with scope limits.",
    }

    summary_path = os.path.join(args.output_dir, "method_summary.csv")
    seeds_path = os.path.join(args.output_dir, "method_by_seed.csv")
    task_path = os.path.join(args.output_dir, "task_breakdown.csv")
    json_path = os.path.join(args.output_dir, "method_summary.json")
    oracle_gap_path = os.path.join(args.output_dir, "oracle_gap_summary.csv")
    sig_path = os.path.join(args.output_dir, "paired_significance.csv")
    instance_path = os.path.join(args.output_dir, "instance_level_scores.csv")
    imbalance_path = os.path.join(args.output_dir, "task_imbalance_report.json")
    fam_sig_path = os.path.join(args.output_dir, "paired_significance_by_task.csv")

    agg.to_csv(summary_path, index=False)
    summary_df.to_csv(seeds_path, index=False)
    task_agg.to_csv(task_path, index=False)
    oracle_gap_df.to_csv(oracle_gap_path, index=False)
    sig_df.to_csv(sig_path, index=False)
    instance_df.to_csv(instance_path, index=False)
    fam_sig_df.to_csv(fam_sig_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(agg.to_dict(orient="records"), f, indent=2, ensure_ascii=False)
    with open(imbalance_path, "w", encoding="utf-8") as f:
        json.dump(imbalance_report, f, indent=2, ensure_ascii=False)

    print(f"Saved: {summary_path}")
    print(f"Saved: {seeds_path}")
    print(f"Saved: {task_path}")
    print(f"Saved: {oracle_gap_path}")
    print(f"Saved: {sig_path}")
    print(f"Saved: {instance_path}")
    print(f"Saved: {fam_sig_path}")
    print(f"Saved: {imbalance_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
