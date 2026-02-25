import argparse
import json
import os
from collections import defaultdict
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Publication-focused evaluation, ablations, and error exports.")
    p.add_argument("--results", default="results.json")
    p.add_argument("--output-dir", default="outputs/research_eval")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--max-error-cases", type=int, default=40)
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

    pm = df_m[keep_cols + ["score", "est_cost_usd", "latency_ms"]].rename(
        columns={"score": "score_mav", "est_cost_usd": "cost_mav", "latency_ms": "latency_mav"}
    )
    pl = df_l[["id", "score", "est_cost_usd", "latency_ms"]].rename(
        columns={"score": "score_70b", "est_cost_usd": "cost_70b", "latency_ms": "latency_70b"}
    )
    merged = pd.merge(pm, pl, on="id", how="inner")
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


def evaluate_policy(df: pd.DataFrame, chooser) -> Dict[str, float]:
    chosen = df.apply(lambda r: chooser(r), axis=1)
    scores = [selected_score(row, model) for row, model in zip(df.to_dict("records"), chosen)]
    costs = [selected_cost(row, model) for row, model in zip(df.to_dict("records"), chosen)]
    lats = [selected_latency(row, model) for row, model in zip(df.to_dict("records"), chosen)]
    chose_70b = np.mean([1.0 if m == LLAMA70_NAME else 0.0 for m in chosen]) if len(chosen) else 0.0
    return {
        "n": int(len(df)),
        "avg_score": float(np.mean(scores)) if scores else 0.0,
        "avg_cost_usd": float(np.nanmean(costs)) if costs else 0.0,
        "avg_latency_ms": float(np.nanmean(lats)) if lats else 0.0,
        "usage_70b_pct": float(chose_70b * 100.0),
    }


def train_learning_router(df_train: pd.DataFrame, seed: int):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000)
    X_train_txt = vec.fit_transform(df_train["query_text"].astype(str).tolist())
    X_train_num = np.c_[
        df_train["query_text"].astype(str).str.len().values,
        df_train["doc_text"].astype(str).str.len().values,
    ]
    X_train = hstack([X_train_txt, X_train_num])
    y_train = np.where(df_train["best_label"] == MAVERICK_NAME, 1, 0)

    clf = RandomForestClassifier(n_estimators=250, max_depth=12, random_state=seed, class_weight="balanced")
    clf.fit(X_train, y_train)
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


def by_task_breakdown(df: pd.DataFrame, chooser) -> List[Dict]:
    rows = []
    for t, g in df.groupby("task_type"):
        metrics = evaluate_policy(g, chooser)
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

    test_df = df[df["split"] == "test"].copy()
    if test_df.empty:
        test_df = df.copy()

    router = IntelligentRouter()
    all_results = []
    task_rows = []
    seed_values = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    for seed in seed_values:
        train_df = df[df["split"].isin(["train", "val"])].copy()
        if train_df.empty:
            train_df, _ = train_test_split(df, test_size=0.3, random_state=seed, stratify=df["task_type"])
        eval_df = test_df

        vec, clf = train_learning_router(train_df, seed=seed)
        learn_pred = predict_learning_router(eval_df, vec, clf)
        label_true = np.where(eval_df["best_label"] == MAVERICK_NAME, 1, 0)
        label_pred = np.where(learn_pred == MAVERICK_NAME, 1, 0)
        ml_accuracy = float(accuracy_score(label_true, label_pred))

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
            "hybrid_rule_plus_learning": lambda r, p=learn_pred: (
                LLAMA70_NAME
                if any(w in r["query_text"].lower() for w in router.reasoning_keywords)
                else p[eval_df.index.get_loc(r.name)]
            ),
            "oracle_upper_bound": lambda r: LLAMA70_NAME if float(r["score_70b"]) >= float(r["score_mav"]) else MAVERICK_NAME,
        }

        for name, chooser in methods.items():
            summary = evaluate_policy(eval_df, chooser)
            summary.update({"method": name, "seed": seed, "ml_router_label_acc": ml_accuracy})
            all_results.append(summary)
            for tr in by_task_breakdown(eval_df, chooser):
                tr.update({"method": name, "seed": seed})
                task_rows.append(tr)

        error_cases = collect_error_cases(eval_df, methods["hybrid_rule_plus_learning"], args.max_error_cases)
        with open(os.path.join(args.output_dir, f"error_cases_seed_{seed}.json"), "w", encoding="utf-8") as f:
            json.dump(error_cases, f, indent=2, ensure_ascii=False)

    summary_df = pd.DataFrame(all_results)
    agg = (
        summary_df.groupby("method")[["avg_score", "avg_cost_usd", "avg_latency_ms", "usage_70b_pct", "ml_router_label_acc"]]
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

    summary_path = os.path.join(args.output_dir, "method_summary.csv")
    seeds_path = os.path.join(args.output_dir, "method_by_seed.csv")
    task_path = os.path.join(args.output_dir, "task_breakdown.csv")
    json_path = os.path.join(args.output_dir, "method_summary.json")

    agg.to_csv(summary_path, index=False)
    summary_df.to_csv(seeds_path, index=False)
    task_agg.to_csv(task_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(agg.to_dict(orient="records"), f, indent=2, ensure_ascii=False)

    print(f"Saved: {summary_path}")
    print(f"Saved: {seeds_path}")
    print(f"Saved: {task_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
