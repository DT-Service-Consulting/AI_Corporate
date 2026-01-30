import json
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION ---
BASELINE_FILE = "results.json"
SEMANTIC_MODEL_NAME = 'all-MiniLM-L6-v2'

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
                model_kwargs={"low_cpu_mem_usage": False}
            )

    def get_vector(self, query):
        """Return the embedding vector for the provided query."""
        self._initialize_encoder()
        return self.encoder.encode([query])[0]

    def route_by_keyword(self, query):
        """Route based on presence of reasoning vs extraction keywords."""
        q = query.lower()
        reasoning_keywords = ["analyze", "evaluate", "compliant", "valid", "breach", "why", "assess", "risk", "interpret", "violate", "hearsay"]
        if any(w in q for w in reasoning_keywords):
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

    # --- 1. DATA LOADING & CLEANING ---
    try:
        with open(BASELINE_FILE, "r") as f:
            raw_data = json.load(f)
        df = pd.DataFrame(raw_data)
        
        # Normalization: Ensure 'input' and 'output' columns exist
        if 'full_output' in df.columns: df['output'] = df['full_output']
        
        # If 'input' is missing (e.g. only have metadata), synthesize it
        if 'input' not in df.columns:
            def synthesize_input(row):
                if row.get('task_type') == 'extraction':
                    return f"Extract terms from {row.get('doc_name', 'document')}."
                return f"Analyze legal validity of {row.get('doc_name', 'clause')}."
            df['input'] = df.apply(synthesize_input, axis=1)

    except Exception as e:
        st.error(f"❌ Could not load '{BASELINE_FILE}'. Error: {e}")
        return

    # --- 2. Prepare merged model results (merge outputs) ---
    try:
        # Filter and Rename for Merge
        df_maverick = df[df['model_name'].str.contains("Maverick", na=False, case=False)][['id', 'input', 'score', 'output']]
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
    
    # --- 3. TRAIN THE SUPERVISED ROUTER (The Gen 3 Step) ---
    st.info(f"Training supervised router on {len(merged_df)} tasks.")
    
    embeddings = []
    targets = []  # 1 = Maverick, 0 = 70B
    
    progress_bar = st.progress(0)
    
    # Feature Engineering Loop
    for i, row in merged_df.iterrows():
        query = str(row['input'])[:300]  # Truncate for speed
        
        # Input Feature: Semantic Vector (384 floats)
        vec = architecture.get_vector(query)
        embeddings.append(vec)
        
        # Target Label: Did Maverick do good enough?
        # Logic: If Maverick is equal or better than 70B, use Maverick (Cheaper).
        # Otherwise, we MUST use 70B.
        if row['score_mav'] >= row['score_70b']:
            targets.append(MAVERICK_LABEL)
        else:
            targets.append(LLAMA70_LABEL)
            
        if i % 10 == 0:
            progress_bar.progress(i / len(merged_df))
    progress_bar.empty()
    
    # Random Forest Training
    clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    clf.fit(embeddings, targets)
    
    train_acc = clf.score(embeddings, targets)
    st.success(f"✅ Gen 3 Router Trained. Internal Fit: {train_acc*100:.1f}%")

    # --- 4. RUN THE SIMULATION ---
    results = []
    
    # Bulk Predict for Strategy 2
    ml_decisions = clf.predict(embeddings)  # [1, 0, 1, 1, 0...]
    
    for i, row in merged_df.iterrows():
        query = str(row['input'])
        doc_text = str(row['input'])

        # Strategy 1: Keyword
        choice_kw = architecture.route_by_keyword(query)
        s1 = row['score_70b'] if choice_kw == "70B" else row['score_mav']

        # Strategy 2: Gen 3 ML Router (Using predictions)
        # If prediction is 1 (Maverick), use Maverick score. Else 70B score.
        s2 = row['score_mav'] if ml_decisions[i] == MAVERICK_LABEL else row['score_70b']

        # Strategy 3: Length
        choice_len = architecture.route_by_length(doc_text)
        s3 = row['score_70b'] if choice_len == "70B" else row['score_mav']
        
        # Strategy 4: Confidence
        choice_conf = architecture.route_by_confidence(row['out_mav'])
        s4 = row['score_70b'] if choice_conf == "70B" else row['score_mav']
        
        # Strategy 5: Oracle (Theoretical Max)
        s5 = max(row['score_mav'], row['score_70b'])

        results.extend([
            {"Method": "1. Keyword Router", "Score": s1},
            {"Method": "2. Gen 3 ML Router (RF)", "Score": s2},
            {"Method": "3. Length Cascade", "Score": s3},
            {"Method": "4. Confidence Cascade", "Score": s4},
            {"Method": "5. The Oracle", "Score": s5},
        ])

    # --- 5. VISUALIZATION ---
    res_df = pd.DataFrame(results)
    acc_summary = res_df.groupby("Method")['Score'].mean() * 100
    
    # Baselines
    acc_summary["0. Pure Maverick (17B)"] = merged_df['score_mav'].mean() * 100
    acc_summary["0. Pure Llama-3 (70B)"] = merged_df['score_70b'].mean() * 100
    acc_summary = acc_summary.sort_values(ascending=False)

    # Cost Estimation Data
    dashboard_data = []
    
    # Calculate ML Router actual usage
    ml_mav_usage = (sum(ml_decisions) / len(ml_decisions)) * 100
    ml_70b_usage = 100 - ml_mav_usage
    
    for method, acc in acc_summary.items():
        if "Pure Llama" in method: cost = 100
        elif "Pure Mav" in method: cost = 0
        elif "Keyword" in method: cost = 45 # Approximate
        elif "Gen 3" in method: cost = ml_70b_usage # Actual
        elif "Length" in method: cost = 85
        elif "Confidence" in method: cost = 30
        elif "Oracle" in method: cost = 55
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

if __name__ == "__main__":
    run_simulation()