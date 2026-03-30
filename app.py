import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st

from core_logic import (
    EXTRACTION_MODEL_NAME,
    REASONING_MODEL_NAME,
    MODELS_TO_TEST,
    evaluate_reasoning,
    evaluate_single_extraction,
    format_model_label,
)
from intelligent_router import IntelligentRouter


st.set_page_config(
    page_title="LegalReason-Eval Pro",
    page_icon="LV",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("LegalReason-Eval: Thesis Dashboard")
st.markdown(
    """
**Evaluating the 'Specialist vs. Generalist' Trade-off in Legal AI.**
1. **Benchmark:** Individual Model Performance (Extraction & Reasoning).
2. **Hybrid Ecosystem:** Intelligent Routing between GPT-4o Mini and GPT-4.1.
"""
)

st.sidebar.header("Navigation")
mode = st.sidebar.radio("Go to:", ["Benchmark Analysis", "Hybrid Ecosystem Proof", "Live Playground"])


def load_data(filename):
    if not os.path.exists(filename):
        return None
    with open(filename, "r", encoding="utf-8") as f:
        try:
            return pd.DataFrame(json.load(f))
        except Exception:
            return None


df_baseline = load_data("results.json")
df_hybrid = load_data("hybrid_system_results.json")


if mode == "Benchmark Analysis":
    if df_baseline is None or df_baseline.empty:
        st.warning("No baseline data found. Run `python run_experiment.py` first.")
    else:
        st.header("1. Individual Model Performance (The Baseline)")

        df_ex = df_baseline[df_baseline["task_type"] == "extraction"].copy()
        df_re = df_baseline[df_baseline["task_type"] == "reasoning"].copy()

        if not df_ex.empty:
            metrics_df = pd.json_normalize(df_ex["metrics"])
            df_ex = df_ex.reset_index(drop=True).join(metrics_df)

        st.subheader("A. Contract Extraction (Safety/Recall)")
        if not df_ex.empty:
            leaderboard = df_ex.groupby("model_name")[["f2", "precision", "recall", "is_lazy"]].mean()
            leaderboard["laziness_rate"] = leaderboard["is_lazy"] * 100
            leaderboard.index = [format_model_label(idx) for idx in leaderboard.index]
            st.dataframe(leaderboard.sort_values("f2", ascending=False).style.format("{:.3f}"), use_container_width=True)
            st.caption("Note: GPT-4o Mini is the extraction-first deployment in this setup.")

        st.subheader("B. Legal Reasoning (Logic/Knowledge)")
        if not df_re.empty:
            acc_df = df_re.groupby("model_name")["score"].mean() * 100
            acc_df.index = [format_model_label(idx) for idx in acc_df.index]
            st.dataframe(acc_df.sort_values(ascending=False).to_frame("Accuracy %").style.format("{:.1f}%"), use_container_width=True)
            st.caption("Note: GPT-4.1 is the reasoning-first deployment in this setup.")


elif mode == "Hybrid Ecosystem Proof":
    if df_hybrid is None:
        st.warning("No hybrid results found. Run `python run_hybrid_system.py` first.")
    else:
        st.header("2. The Hybrid System (The Solution)")
        st.success("Thesis hypothesis: a routed system outperforms any single model.")

        st.subheader("Router Decisions")
        intent_counts = df_hybrid["router_intent"].value_counts().reset_index()
        intent_counts.columns = ["Intent", "Count"]

        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(intent_counts, hide_index=True, use_container_width=True)
        with c2:
            fig = px.pie(
                intent_counts,
                values="Count",
                names="Intent",
                title="Task Distribution (Extraction vs. Reasoning)",
                color_discrete_sequence=px.colors.sequential.RdBu,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Performance Comparison")

        valid_hybrid = df_hybrid[~df_hybrid["id"].astype(str).str.startswith("tender_")]
        hybrid_score = valid_hybrid["score"].mean() * 100

        rows = []
        if df_baseline is not None:
            baseline_avgs = df_baseline[~df_baseline["is_discovery"]].groupby("model_name")["score"].mean() * 100
            for model, score in baseline_avgs.items():
                rows.append({"System": format_model_label(model), "Score": score, "Type": "Individual"})

        rows.append({"System": "Hybrid Router (Ours)", "Score": hybrid_score, "Type": "Ecosystem"})
        comp_df = pd.DataFrame(rows).sort_values("Score", ascending=False)

        fig_bar = px.bar(
            comp_df,
            x="System",
            y="Score",
            color="Type",
            title="Final System Performance (Higher is Better)",
            text_auto=".1f",
            color_discrete_map={"Individual": "gray", "Ecosystem": "green"},
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown(
            f"""
### Conclusion
The **Hybrid System** achieved a score of **{hybrid_score:.1f}%**, leveraging the specialist capabilities of both models.
- Uses **{format_model_label(EXTRACTION_MODEL_NAME)}** for extraction-first tasks.
- Uses **{format_model_label(REASONING_MODEL_NAME)}** for reasoning-heavy tasks.
"""
        )


elif mode == "Live Playground":
    st.header("Router Logic Lab")

    router = IntelligentRouter()
    user_query = st.text_input("Enter a Legal Query:", "Is this clause compliant with GDPR Article 17?")
    doc_snippet = st.text_area("Document Context (Optional):", height=100)

    if st.button("Check Route"):
        model, intent = router.route(user_query, doc_snippet)
        matched_keywords = [k for k in router.extraction_keywords if k in user_query.lower()]

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Detected Intent", intent.upper())
        with c2:
            st.metric("Selected Specialist", format_model_label(model))

        if intent == "extraction":
            st.info(
                f"Routing to **{format_model_label(EXTRACTION_MODEL_NAME)}** because keywords like "
                f"'{', '.join(matched_keywords)}' matched extraction logic."
            )
        else:
            st.info(f"Routing to **{format_model_label(REASONING_MODEL_NAME)}** because it requires reasoning/compliance analysis.")
