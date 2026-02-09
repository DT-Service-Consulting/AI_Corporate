import json
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import re

# --- CONFIGURATION ---
BASELINE_FILE = "results.json"
SEMANTIC_MODEL_NAME = 'all-MiniLM-L6-v2'
TEST_SIZE = 0.2
VAL_SIZE = 0.2

# Label constants for supervised router
MAVERICK_LABEL = 1
LLAMA70_LABEL = 0

class ArchitectureLab:
    def __init__(self):
        self.encoder = None

    def _initialize_encoder(self):
        """Load the embedding encoder. Use CPU fallback to avoid platform-specific issues."""
        if self.encoder is None:
            self.encoder = SentenceTransformer(
                SEMANTIC_MODEL_NAME,
                device="cpu",
                model_kwargs={
                    "low_cpu_mem_usage": False,
                    "device_map": None
                }
            )

    def get_vector(self, query):
        """Return the embedding vector for the provided query."""
        self._initialize_encoder()
        return self.encoder.encode([query])[0]

    def route_by_keyword(self, query):
        """Route based on presence of reasoning vs extraction keywords."""
        q = query.lower()
        reasoning_keywords = ["analyze", "evaluate", "compliant", "valid", "breach", "assess", "risk", "interpret", "violate", "hearsay"]
        extraction_keywords = ["extract", "list", "identify", "pull", "find", "enumerate", "summarize"]

        # Prefer extraction for explicit extraction verbs.
        if any(re.search(rf"\\b{re.escape(w)}\\b", q) for w in extraction_keywords):
            return "Maverick"

        if any(re.search(rf"\\b{re.escape(w)}\\b", q) for w in reasoning_keywords):
            return "70B"
        return "Maverick"

    def route_by_length(self, doc_text):
        """Prefer extraction model for very large documents."""
        if len(doc_text) > 15000:
            return "Maverick"
        return "70B"

    def route_by_confidence(self, mav_output):
        """Fallback to reasoning when the extraction model output is low-confidence or empty."""
        out = str(mav_output).lower().strip()
        if not out or len(out) < 5:
            return "70B"
        if "unable to" in out or "cannot find" in out or "sorry" in out:
            return "70B"
        return "Maverick"

def run_simulation():
    st.set_page_config(page_title="Gen 3 Router Lab", layout="wide")
    st.title("🛡️ Gen 3 Architecture: Supervised ML Routing")
    st.markdown("Training a Random Forest Classifier to route tasks based on vector embeddings.")

    st.sidebar.header("Router Settings")
    label_margin = st.sidebar.slider(
        "Label margin (70B - Maverick)",
        min_value=0.0,
        max_value=10.0,
        value=0.0,
        step=0.1,
        help="Route to 70B only if it beats Maverick by at least this margin."
    )
    normalize_scores = st.sidebar.checkbox(
        "Normalize scores for labeling",
        value=False,
        help="Z-score normalize model scores before creating labels (labels only)."
    )
    class_balance = st.sidebar.checkbox(
        "Balance router classes (class_weight)",
        value=True,
        help="Use class_weight='balanced' for the RF to mitigate label imbalance."
    )
    resample_training = st.sidebar.checkbox(
        "Resample training data",
        value=False,
        help="Downsample majority class to match minority (train/val only)."
    )
    auto_margin = st.sidebar.checkbox(
        "Auto margin to target 70B label rate",
        value=False,
        help="Automatically choose the margin to reach a target 70B label rate."
    )
    target_70b_rate = st.sidebar.slider(
        "Target 70B label rate (%)",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
        help="Used when auto margin is enabled."
    )

    st.sidebar.subheader("Ensembling Settings")
    ensemble_weight = st.sidebar.slider(
        "Weighted ensemble weight for 70B",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Score = w*70B + (1-w)*Maverick."
    )

    # --- 1. DATA LOADING & CLEANING ---
    try:
        with open(BASELINE_FILE, "r") as f:
            raw_data = json.load(f)
        df = pd.DataFrame(raw_data)
        
        # Normalization: Ensure 'input' and 'output' columns exist
        if 'full_output' in df.columns:
            df['output'] = df['full_output']
        if 'output' not in df.columns:
            st.error("Missing required column: output (or full_output).")
            st.stop()
        
        # If 'input' is missing (e.g. only have metadata), synthesize it
        if 'input' not in df.columns:
            def synthesize_input(row):
                if row.get('task_type') == 'extraction':
                    return f"Extract terms from {row.get('doc_name', 'document')}."
                return f"Analyze legal validity of {row.get('doc_name', 'clause')}."
            df['input'] = df.apply(synthesize_input, axis=1)

        # Basic schema validation
        required_cols = ["id", "model_name", "score"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
            st.stop()

        # Normalize score type
        df['score'] = pd.to_numeric(df['score'], errors='coerce')
        if df['score'].isna().any():
            st.warning("Some scores could not be parsed as numbers and were set to NaN.")

        # Prefer any available document text field
        doc_text_candidates = ["doc_text", "document", "text", "content", "full_text"]
        doc_text_source = next((c for c in doc_text_candidates if c in df.columns), None)
        if doc_text_source:
            df["doc_text"] = df[doc_text_source].astype(str)
        else:
            df["doc_text"] = ""

    except Exception as e:
        st.error(f"❌ Could not load '{BASELINE_FILE}'. Error: {e}")
        return

    # --- 2. Prepare merged model results (merge outputs) ---
    try:
        # Filter and Rename for Merge
        extra_cols = [c for c in ["task_type", "doc_text", "document", "doc_name"] if c in df.columns]
        df_maverick = df[df['model_name'].str.contains("Maverick", na=False, case=False)][['id', 'input', 'score', 'output'] + extra_cols]
        df_maverick = df_maverick.rename(columns={'score': 'score_mav', 'output': 'out_mav'})

        df_llama70 = df[df['model_name'].str.contains("70B", na=False, case=False)][['id', 'score', 'output']]
        df_llama70 = df_llama70.rename(columns={'score': 'score_70b', 'output': 'out_70b'})

        merged_df = pd.merge(df_maverick, df_llama70, on='id', how='inner')
    except Exception as e:
        st.error(f"❌ Data Merge Failed: {e}. Check results.json.")
        st.stop()
    
    if merged_df.empty:
        st.error("No overlapping tasks found. Please run 'run_experiment.py' properly.")
        st.stop()

    architecture = ArchitectureLab()

    # Optional score normalization for labeling
    if normalize_scores:
        mav_mean = merged_df['score_mav'].mean()
        mav_std = merged_df['score_mav'].std() or 1.0
        b70_mean = merged_df['score_70b'].mean()
        b70_std = merged_df['score_70b'].std() or 1.0
        merged_df['score_mav_label'] = (merged_df['score_mav'] - mav_mean) / mav_std
        merged_df['score_70b_label'] = (merged_df['score_70b'] - b70_mean) / b70_std
    else:
        merged_df['score_mav_label'] = merged_df['score_mav']
        merged_df['score_70b_label'] = merged_df['score_70b']

    # Warn if score signal is effectively binary
    mav_unique = merged_df['score_mav'].dropna().unique()
    b70_unique = merged_df['score_70b'].dropna().unique()
    if len(mav_unique) <= 2 and len(b70_unique) <= 2:
        st.warning("Scores look binary (few unique values). Consider using a softer metric.")

    # Auto margin to hit a target 70B label rate
    if auto_margin:
        deltas = (merged_df["score_70b_label"] - merged_df["score_mav_label"]).values
        deltas_sorted = np.sort(deltas)
        target_rate = target_70b_rate / 100.0
        # choose margin where P(delta > margin) ~= target_rate
        # margin is (1 - target_rate) quantile
        quantile = max(0.0, min(1.0, 1.0 - target_rate))
        label_margin = float(np.quantile(deltas_sorted, quantile))
        st.caption(f"Auto margin set to {label_margin:.3f} for target 70B label rate {target_70b_rate}%.")
    
    # --- 3. TRAIN THE SUPERVISED ROUTER (The Gen 3 Step) ---
    st.info(f"Training supervised router on {len(merged_df)} tasks.")
    
    embeddings = []
    router_features = []
    targets = []  # 1 = Maverick, 0 = 70B
    
    progress_bar = st.progress(0)
    
    # Feature Engineering Loop
    for idx, row in merged_df.reset_index(drop=True).iterrows():
        query = str(row['input'])[:300]  # Truncate for speed
        
        # Input Feature: Semantic Vector (384 floats)
        vec = architecture.get_vector(query)
        embeddings.append(vec)

        # Additional router features for ensembling/meta models
        if 'doc_text' in row and str(row['doc_text']).strip():
            doc_text = str(row['doc_text'])
        elif 'document' in row and str(row['document']).strip():
            doc_text = str(row['document'])
        else:
            doc_text = str(row['input'])
        kw_choice = architecture.route_by_keyword(query)
        kw_flag = 1 if kw_choice == "Maverick" else 0
        score_delta = row['score_70b_label'] - row['score_mav_label']
        router_features.append([
            float(row['score_mav_label']),
            float(row['score_70b_label']),
            float(score_delta),
            float(len(doc_text)),
            float(kw_flag)
        ])
        
        # Target Label: Did Maverick do good enough?
        # Logic: If Maverick is equal or better than 70B, use Maverick (Cheaper).
        # Otherwise, we MUST use 70B.
        score_delta = row['score_70b_label'] - row['score_mav_label']
        if score_delta <= label_margin:
            targets.append(MAVERICK_LABEL)
        else:
            targets.append(LLAMA70_LABEL)
            
        if idx % 10 == 0:
            progress_bar.progress(idx / len(merged_df))
    progress_bar.empty()
    
    # Random Forest Training with Train/Val/Test split
    X = np.array(embeddings)
    F = np.array(router_features)
    y = np.array(targets)
    indices = np.arange(len(merged_df))

    train_val_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    val_size_adjusted = VAL_SIZE / max(1e-6, (1.0 - TEST_SIZE))
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_size_adjusted, random_state=42, stratify=y[train_val_idx]
    )

    # Label distribution diagnostics
    def _label_counts(idx, labels):
        vals, counts = np.unique(labels[idx], return_counts=True)
        data = {int(v): int(c) for v, c in zip(vals, counts)}
        return {
            "Maverick": data.get(MAVERICK_LABEL, 0),
            "70B": data.get(LLAMA70_LABEL, 0),
            "Total": int(len(idx))
        }

    label_dist = pd.DataFrame([
        {"Split": "Train", **_label_counts(train_idx, y)},
        {"Split": "Val", **_label_counts(val_idx, y)},
        {"Split": "Test", **_label_counts(test_idx, y)},
    ])

    # Optional resampling to balance classes on train/val only
    def _balance_indices(idx, labels):
        idx = np.array(idx)
        labels = labels[idx]
        classes, counts = np.unique(labels, return_counts=True)
        if len(classes) < 2:
            return idx
        min_count = counts.min()
        balanced = []
        for c in classes:
            c_idx = idx[labels == c]
            if len(c_idx) > min_count:
                c_idx = np.random.choice(c_idx, size=min_count, replace=False)
            balanced.append(c_idx)
        return np.concatenate(balanced)

    if resample_training:
        train_idx = _balance_indices(train_idx, y)
        val_idx = _balance_indices(val_idx, y)

    # Simple hyperparameter selection on validation
    candidate_params = [
        {"n_estimators": 100, "max_depth": 6},
        {"n_estimators": 100, "max_depth": 8},
        {"n_estimators": 200, "max_depth": 8},
        {"n_estimators": 200, "max_depth": 10},
    ]
    best_params = None
    best_val_acc = -1.0
    for params in candidate_params:
        clf_candidate = RandomForestClassifier(
            random_state=42,
            class_weight="balanced" if class_balance else None,
            **params
        )
        clf_candidate.fit(X[train_idx], y[train_idx])
        val_acc = clf_candidate.score(X[val_idx], y[val_idx])
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = params

    clf = RandomForestClassifier(
        random_state=42,
        class_weight="balanced" if class_balance else None,
        **best_params
    )
    clf.fit(X[np.concatenate([train_idx, val_idx])], y[np.concatenate([train_idx, val_idx])])

    train_acc = clf.score(X[train_idx], y[train_idx])
    val_acc = clf.score(X[val_idx], y[val_idx])
    test_acc = clf.score(X[test_idx], y[test_idx])
    st.success(
        f"✅ Gen 3 Router Trained. Train: {train_acc*100:.1f}%, "
        f"Val: {val_acc*100:.1f}%, Test: {test_acc*100:.1f}%"
    )
    st.caption(f"Best RF params: {best_params}")

    # --- 3b. Ensembling Prep (Stacking + Soft MoE) ---
    meta_clf = None
    meta_decisions = None
    try:
        meta_clf = LogisticRegression(
            class_weight="balanced" if class_balance else None,
            max_iter=1000,
            solver="liblinear"
        )
        meta_clf.fit(F[np.concatenate([train_idx, val_idx])], y[np.concatenate([train_idx, val_idx])])
        meta_decisions = meta_clf.predict(F)
    except Exception as e:
        st.warning(f"Stacking model failed: {e}")

    proba_all = None
    if hasattr(clf, "predict_proba"):
        try:
            proba_all = clf.predict_proba(X)
        except Exception as e:
            st.warning(f"Probability prediction failed: {e}")

    st.markdown("### Label Distribution")
    st.dataframe(label_dist.set_index("Split"))
    if label_dist.loc[label_dist["Split"] == "Test", "70B"].iloc[0] == 0:
        st.warning("Test split has zero 70B labels. Consider increasing label margin or enabling normalization.")

    # Score distribution diagnostics
    st.markdown("### Score Distributions")
    score_df = pd.DataFrame({
        "Maverick": merged_df["score_mav"].astype(float),
        "70B": merged_df["score_70b"].astype(float)
    })
    fig_scores = px.histogram(
        score_df.melt(var_name="Model", value_name="Score"),
        x="Score", color="Model", barmode="overlay", opacity=0.6
    )
    st.plotly_chart(fig_scores, use_container_width=True)

    # --- 4. RUN THE SIMULATION ---
    results = []
    usage_stack = []
    usage_vote = []
    usage_cascade = []
    usage_soft = []
    usage_kw_conf = []
    
    # Bulk Predict for Strategy 2
    ml_decisions = clf.predict(X)  # [1, 0, 1, 1, 0...]
    X_test = X[test_idx]
    y_test = y[test_idx]
    y_pred = clf.predict(X_test)
    test_idx_set = set(test_idx.tolist())
    test_idx_list = sorted(test_idx_set)

    # For soft MoE
    proba_classes = list(clf.classes_) if hasattr(clf, "classes_") else [MAVERICK_LABEL, LLAMA70_LABEL]
    mav_idx = proba_classes.index(MAVERICK_LABEL) if MAVERICK_LABEL in proba_classes else 0

    for idx, row in merged_df.reset_index(drop=True).iterrows():
        if idx not in test_idx_set:
            continue
        query = str(row['input'])
        if 'doc_text' in row and str(row['doc_text']).strip():
            doc_text = str(row['doc_text'])
        elif 'document' in row and str(row['document']).strip():
            doc_text = str(row['document'])
        else:
            doc_text = str(row['input'])

        # Strategy 1: Keyword
        choice_kw = architecture.route_by_keyword(query)
        s1 = row['score_70b'] if choice_kw == "70B" else row['score_mav']

        # Strategy 2: Gen 3 ML Router (Using predictions)
        # If prediction is 1 (Maverick), use Maverick score. Else 70B score.
        s2 = row['score_mav'] if ml_decisions[idx] == MAVERICK_LABEL else row['score_70b']

        # Strategy 3: Length
        choice_len = architecture.route_by_length(doc_text)
        s3 = row['score_70b'] if choice_len == "70B" else row['score_mav']
        
        # Strategy 4: Confidence
        choice_conf = architecture.route_by_confidence(row['out_mav'])
        s4 = row['score_70b'] if choice_conf == "70B" else row['score_mav']
        
        # Strategy 5: Oracle (Theoretical Max)
        s5 = max(row['score_mav'], row['score_70b'])

        # Strategy 6: Weighted Score Ensemble
        s6 = (ensemble_weight * row['score_70b']) + ((1.0 - ensemble_weight) * row['score_mav'])

        # Strategy 7: Stacking Meta Router
        if meta_decisions is not None:
            choice_stack = "Maverick" if meta_decisions[idx] == MAVERICK_LABEL else "70B"
            s7 = row['score_70b'] if choice_stack == "70B" else row['score_mav']
            usage_stack.append(choice_stack)
        else:
            s7 = None

        # Strategy 8: Soft MoE (probability-weighted)
        if proba_all is not None:
            p_mav = float(proba_all[idx, mav_idx])
            p_70b = 1.0 - p_mav
            s8 = (p_mav * row['score_mav']) + (p_70b * row['score_70b'])
            usage_soft.append(p_70b)
        else:
            s8 = None

        # Strategy 9: Router Voting (Keyword + Length + Confidence + ML)
        votes = [
            choice_kw,
            choice_len,
            choice_conf,
            "Maverick" if ml_decisions[idx] == MAVERICK_LABEL else "70B"
        ]
        votes_mav = sum(1 for v in votes if v == "Maverick")
        votes_70b = len(votes) - votes_mav
        if votes_mav == votes_70b:
            choice_vote = "Maverick" if row['score_mav'] >= row['score_70b'] else "70B"
        else:
            choice_vote = "Maverick" if votes_mav > votes_70b else "70B"
        s9 = row['score_70b'] if choice_vote == "70B" else row['score_mav']
        usage_vote.append(choice_vote)

        # Strategy 10: Two-Stage Cascade (Keyword + Confidence backoff)
        choice_cascade = "70B" if (choice_kw == "70B" or choice_conf == "70B") else "Maverick"
        s10 = row['score_70b'] if choice_cascade == "70B" else row['score_mav']
        usage_cascade.append(choice_cascade)

        task_type = row.get('task_type', 'unknown')
        results.extend([
            {"Method": "1. Keyword Router", "Score": s1, "TaskType": task_type},
            {"Method": "2. Gen 3 ML Router (RF)", "Score": s2, "TaskType": task_type},
            {"Method": "3. Length Cascade", "Score": s3, "TaskType": task_type},
            {"Method": "4. Confidence Cascade", "Score": s4, "TaskType": task_type},
            {"Method": "5. The Oracle", "Score": s5, "TaskType": task_type},
            {"Method": "6. Weighted Ensemble (Score Blend)", "Score": s6, "TaskType": task_type},
        ])
        if s7 is not None:
            results.append({"Method": "7. Stacking Meta Router", "Score": s7, "TaskType": task_type})
        if s8 is not None:
            results.append({"Method": "8. Soft MoE (RF Prob Blend)", "Score": s8, "TaskType": task_type})
        results.extend([
            {"Method": "9. Router Voting", "Score": s9, "TaskType": task_type},
            {"Method": "10. Two-Stage Cascade (KW+Conf)", "Score": s10, "TaskType": task_type},
        ])

    # --- 5. VISUALIZATION ---

    res_df = pd.DataFrame(results)
    test_df = merged_df.reset_index(drop=True).iloc[test_idx_list]
    acc_summary = res_df.groupby("Method")['Score'].mean() * 100
    
    # Baselines
    acc_summary["0. Pure Maverick (17B)"] = test_df['score_mav'].mean() * 100
    acc_summary["0. Pure Llama-3 (70B)"] = test_df['score_70b'].mean() * 100
    acc_summary = acc_summary.sort_values(ascending=False)

    # Cost Estimation Data
    dashboard_data = []
    
    # Calculate ML Router actual usage
    test_ml_decisions = ml_decisions[test_idx_list]
    ml_mav_usage = (sum(test_ml_decisions) / len(test_ml_decisions)) * 100
    ml_70b_usage = 100 - ml_mav_usage

    def _usage_rate(choices):
        if not choices:
            return 0.0
        return (sum(1 for c in choices if c == "70B") / len(choices)) * 100

    stack_cost = _usage_rate(usage_stack)
    vote_cost = _usage_rate(usage_vote)
    cascade_cost = _usage_rate(usage_cascade)
    soft_cost = (float(np.mean(usage_soft)) * 100) if usage_soft else 0.0
    
    for method, acc in acc_summary.items():
        if "Pure Llama" in method: cost = 100
        elif "Pure Mav" in method: cost = 0
        elif "Keyword" in method: cost = 45 # Approximate
        elif "Gen 3" in method: cost = ml_70b_usage # Actual
        elif "Length" in method: cost = 85
        elif "Confidence" in method: cost = 30
        elif "Oracle" in method: cost = 55
        elif "Weighted Ensemble" in method: cost = ensemble_weight * 100
        elif "Stacking Meta Router" in method: cost = stack_cost
        elif "Soft MoE" in method: cost = soft_cost
        elif "Router Voting" in method: cost = vote_cost
        elif "Two-Stage Cascade" in method: cost = cascade_cost
        else: cost = 50
        
        dashboard_data.append({
            "Method": method, 
            "Accuracy": acc, 
            "70B Usage (Cost)": cost, 
            "Maverick Usage": 100-cost
        })
    
    metrics_df = pd.DataFrame(dashboard_data).set_index("Method").sort_values("Accuracy", ascending=False)

    # Charts
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("### 1. ROI Leaderboard")
        fig = px.bar(
            metrics_df, x=metrics_df.index, y="Accuracy", text_auto='.1f', 
            color=metrics_df.index,
            color_discrete_map={
                "5. The Oracle": "#00CC96",
                "2. Gen 3 ML Router (RF)": "#636EFA", # Blue Winner
                "0. Pure Llama-3 (70B)": "#EF553B"
            }
        )
        fig.update_layout(showlegend=False, yaxis_range=[40, 60], height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.markdown("### 2. Value Matrix")
        st.caption("Top Left = Best (High Acc, Low Cost)")
        fig2 = px.scatter(
            metrics_df, x="70B Usage (Cost)", y="Accuracy", 
            color=metrics_df.index, size=[25]*len(metrics_df)
        )
        fig2.update_layout(xaxis_autorange="reversed")
        st.plotly_chart(fig2, use_container_width=True)

    # --- 6. DIAGNOSTICS ---
    st.markdown("### 3. Router Diagnostics")
    cm = confusion_matrix(y_test, y_pred, labels=[MAVERICK_LABEL, LLAMA70_LABEL])
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Maverick", "Actual 70B"],
        columns=["Pred Maverick", "Pred 70B"]
    )
    st.dataframe(cm_df)
    st.text(classification_report(
        y_test, y_pred, labels=[MAVERICK_LABEL, LLAMA70_LABEL],
        target_names=["Maverick", "70B"],
        digits=3
    ))

    # Cost-Accuracy tradeoff for ML router thresholds
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X_test)
        classes = list(clf.classes_)
        mav_idx = classes.index(MAVERICK_LABEL)
        thresholds = np.linspace(0.05, 0.95, 19)
        curve_rows = []
        for t in thresholds:
            choose_mav = proba[:, mav_idx] >= t
            scores = np.where(choose_mav, test_df['score_mav'].values, test_df['score_70b'].values)
            acc = scores.mean() * 100
            cost = (1.0 - choose_mav.mean()) * 100
            curve_rows.append({"Threshold": t, "Accuracy": acc, "70B Usage (Cost)": cost})
        curve_df = pd.DataFrame(curve_rows)
        st.markdown("### 4. ML Router Threshold Tradeoff")
        fig3 = px.line(curve_df, x="70B Usage (Cost)", y="Accuracy", markers=True)
        st.plotly_chart(fig3, use_container_width=True)

    # Task-type breakdown if available
    if "TaskType" in res_df.columns and not (res_df["TaskType"].nunique() == 1 and res_df["TaskType"].iloc[0] == "unknown"):
        st.markdown("### 5. Accuracy by Task Type")
        task_pivot = (res_df.pivot_table(
            index="TaskType", columns="Method", values="Score", aggfunc="mean"
        ) * 100).sort_index()
        st.dataframe(task_pivot)

    # Label rate vs margin diagnostic
    st.markdown("### 6. Label Rate vs Margin")
    margins = np.linspace(0.0, 5.0, 11)
    label_rows = []
    for m in margins:
        delta = merged_df["score_70b_label"] - merged_df["score_mav_label"]
        choose_70b = (delta > m).mean() * 100
        label_rows.append({"Margin": m, "70B Label Rate (%)": choose_70b})
    label_df = pd.DataFrame(label_rows)
    fig4 = px.line(label_df, x="Margin", y="70B Label Rate (%)", markers=True)
    st.plotly_chart(fig4, use_container_width=True)

if __name__ == "__main__":
    run_simulation()
