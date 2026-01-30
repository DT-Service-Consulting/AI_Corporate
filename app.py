import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
from core_logic import evaluate_single_extraction, evaluate_reasoning, MODELS_TO_TEST
from intelligent_router import IntelligentRouter

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="LegalReason-Eval Pro",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- HEADER ---
st.title("⚖️ LegalReason-Eval: Thesis Dashboard")
st.markdown("""
**Evaluating the 'Specialist vs. Generalist' Trade-off in Legal AI.**
1.  **Benchmark:** Individual Model Performance (Extraction & Reasoning).
2.  **Hybrid Ecosystem:** Intelligent Routing between 17B (Maverick) and 70B (Llama-3).
""")

# --- SIDEBAR ---
st.sidebar.header("Navigation")
mode = st.sidebar.radio("Go to:", ["📊 Benchmark Analysis", "🚀 Hybrid Ecosystem Proof", "🧪 Live Playground"])

# --- DATA LOADING ---
def load_data(filename):
    if not os.path.exists(filename):
        return None
    with open(filename, "r") as f:
        try:
            return pd.DataFrame(json.load(f))
        except:
            return None

df_baseline = load_data("results.json")
df_hybrid = load_data("hybrid_system_results.json")

# --- TAB 1: BENCHMARK ANALYSIS (Baselines) ---
if mode == "📊 Benchmark Analysis":
    if df_baseline is None or df_baseline.empty:
        st.warning("⚠️ No baseline data found. Run `python run_experiment.py` first.")
    else:
        st.header("1. Individual Model Performance (The Baseline)")
        
        # Split Data
        df_ex = df_baseline[df_baseline['task_type'] == 'extraction'].copy()
        df_re = df_baseline[df_baseline['task_type'] == 'reasoning'].copy()
        
        # Flatten metrics
        if not df_ex.empty:
             metrics_df = pd.json_normalize(df_ex['metrics'])
             df_ex = df_ex.reset_index(drop=True).join(metrics_df)

        # A. EXTRACTION LEADERBOARD (F2 Score)
        st.subheader("A. Contract Extraction (Safety/Recall)")
        if not df_ex.empty:
            leaderboard = df_ex.groupby("model_name")[['f2', 'precision', 'recall', 'is_lazy']].mean()
            leaderboard['laziness_rate'] = leaderboard['is_lazy'] * 100
            st.dataframe(leaderboard.sort_values("f2", ascending=False).style.format("{:.3f}"), use_container_width=True)
            st.caption("Note: Llama-4-Maverick (17B) usually wins here due to high Recall.")

        # B. REASONING LEADERBOARD (Accuracy)
        st.subheader("B. Legal Reasoning (Logic/Knowledge)")
        if not df_re.empty:
            acc_df = df_re.groupby("model_name")['score'].mean() * 100
            st.dataframe(acc_df.sort_values(ascending=False).to_frame("Accuracy %").style.format("{:.1f}%"), use_container_width=True)
            st.caption("Note: Llama-3.3-70B usually wins here due to larger world knowledge.")


# --- TAB 2: HYBRID ECOSYSTEM PROOF (The Thesis) ---
elif mode == "🚀 Hybrid Ecosystem Proof":
    if df_hybrid is None:
        st.warning("⚠️ No hybrid results found. Run `python run_hybrid_system.py` first.")
    else:
        st.header("2. The Hybrid System (The Solution)")
        st.success("Thesis Hypothesis: A routed system outperforms any single model.")
        
        # 1. Routing Statistics
        st.subheader("🧠 Router Decisions")
        intent_counts = df_hybrid['router_intent'].value_counts().reset_index()
        intent_counts.columns = ['Intent', 'Count']
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(intent_counts, hide_index=True, use_container_width=True)
        with c2:
            fig = px.pie(intent_counts, values='Count', names='Intent', title="Task Distribution (Extraction vs. Reasoning)", color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig, use_container_width=True)
            
        # 2. THE FINAL COMPARISON (Hybrid vs. Baselines)
        st.divider()
        st.subheader("🏆 Performance Comparison")
        
        # Calculate Hybrid Average Score
        # We filter out 'discovery' (tenders) for scoring purposes
        valid_hybrid = df_hybrid[~df_hybrid['id'].astype(str).str.startswith("tender_")]
        hybrid_score = valid_hybrid['score'].mean() * 100
        
        # Get Baseline Scores
        rows = []
        if df_baseline is not None:
            # Global Average per model
            baseline_avgs = df_baseline[~df_baseline['is_discovery']].groupby("model_name")['score'].mean() * 100
            for model, score in baseline_avgs.items():
                rows.append({"System": model, "Score": score, "Type": "Individual"})
        
        # Add Hybrid
        rows.append({"System": "Hybrid Router (Ours)", "Score": hybrid_score, "Type": "Ecosystem"})
        
        comp_df = pd.DataFrame(rows).sort_values("Score", ascending=False)
        
        # Display Chart
        fig_bar = px.bar(
            comp_df, x="System", y="Score", color="Type", 
            title="Final System Performance (Higher is Better)",
            text_auto='.1f',
            color_discrete_map={"Individual": "gray", "Ecosystem": "green"}
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        st.markdown(f"""
        ### Conclusion:
        The **Hybrid System** achieved a score of **{hybrid_score:.1f}%**, leveraging the specialist capabilities of both models.
        - Uses **Maverick** for high-recall extraction.
        - Uses **70B** for complex reasoning.
        """)

# --- TAB 3: LIVE PLAYGROUND ---
elif mode == "🧪 Live Playground":
    st.header("Router Logic Lab")
    
    # Initialize Router
    router = IntelligentRouter()
    
    user_query = st.text_input("Enter a Legal Query:", "Is this clause compliant with GDPR Article 17?")
    doc_snippet = st.text_area("Document Context (Optional):", height=100)
    
    if st.button("Check Route"):
        model, intent = router.route(user_query, doc_snippet)
        
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Detected Intent", intent.upper())
        with c2:
            st.metric("Selected Specialist", model.replace("Llama-4-Maverick-17B-128E-Instruct-FP8", "Maverick (17B)").replace("Llama-3.3-70B-Instruct", "Llama-3 (70B)"))
            
        if intent == "extraction":
            st.info(f"👉 Routing to **Maverick (17B)** because keywords like '{', '.join([k for k in router.extraction_keywords if k in user_query.lower()])}' matched extraction logic.")
        else:
            st.info(f"👉 Routing to **Llama-3 (70B)** because it requires reasoning/compliance analysis.")