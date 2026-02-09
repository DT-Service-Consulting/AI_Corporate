import json
import uuid
import time
import os
from datetime import datetime
from tqdm import tqdm
import difflib
from core_logic import evaluate_single_extraction, evaluate_reasoning, MODELS_TO_TEST

# --- DATASET LOADING STRATEGY ---

# 1. Load Real Tenders (Discovery / Extraction)
tender_data = []
if os.path.exists("data/real_tenders.json"):
    try:
        with open("data/real_tenders.json", "r") as f:
            tender_data = json.load(f)
        print(f"📂 Loaded {len(tender_data)} Real Tenders.")
    except Exception as e:
        print(f"⚠️ Error loading tenders: {e}")

# 2. Load LegalBench Tasks (Reasoning / Logic)
legalbench_data = []
if os.path.exists("data/legalbench_data.json"):
    try:
        with open("data/legalbench_data.json", "r") as f:
            legalbench_data = json.load(f)
        print(f"⚖️ Loaded {len(legalbench_data)} LegalBench Tasks.")
    except Exception as e:
        print(f"⚠️ Error loading LegalBench: {e}")

# 3. Benchmark Control Group
control_group = [
    {
        "id": "control_extract_1", 
        "type": "extraction", 
        "doc_name": "Control_Consulting_Agreement.txt",
        "input": "The Consultant shall receive a retainer of $10,000 per month.",
        "target": "retainer of $10,000 per month",
        "is_discovery": False
    },
    {
        "id": "control_reason_1", 
        "type": "reasoning", 
        "doc_name": "Control_Hearsay_Test",
        "input": {
            "rule": "Hearsay is an out-of-court statement offered to prove the truth of the matter asserted.",
            "facts": "Alice testifies that she saw the light turn red.",
            "question": "Is this hearsay?", 
            "options": ["Yes", "No"]
        },
        "target": "No",
        "is_discovery": False
    }
]

# --- MERGE & STABILIZE DATASETS ---
full_dataset = control_group + tender_data + legalbench_data

# ✅ CRITICAL FIX: Ensure every task has a STABLE ID before testing starts
# This ensures Maverick and 70B get the SAME ID for the SAME task.
print("🔧 Stabilizing Task IDs...")
for idx, item in enumerate(full_dataset):
    if 'id' not in item:
        # Create a deterministic ID based on index if missing
        item['id'] = f"task_{idx}_{item['type']}"

print(f"🎯 Total Tasks Queued: {len(full_dataset)}")
print("-" * 50)

def run():
    print(f"🚀 Starting Experiment on {len(MODELS_TO_TEST)} Models...")
    results = []

    for model in MODELS_TO_TEST:
        print(f"\n🔵 Testing Model: {model}...")
        
        for item in tqdm(full_dataset, desc="Processing"):
            
            # --- ROUTER LOGIC ---
            
            # CASE A: EXTRACTION
            if item['type'] == 'extraction':
                query = item.get('query_intent', item['target'])
                res = evaluate_single_extraction(item['input'], query, model)
                
                # Use a softer metric for router labeling and reporting
                # NOTE: discovery tasks still get a real score so extraction accuracy isn't forced to 0
                score_main = res['metrics'].get('jaccard', res['metrics'].get('f2', 0.0))
                metrics = res['metrics']

            # CASE B: REASONING
            elif item['type'] == 'reasoning':
                answer_obj = {'question': item['input']['question'], 'options': item['input']['options'], 'answer': item['target']}
                res = evaluate_reasoning(item['input']['rule'], item['input']['facts'], answer_obj, model)
                # Soft scoring for reasoning to avoid binary-only labels
                parsed_answer = str(res.get("parsed_answer", "")).lower().strip()
                ground_truth = str(item['target']).lower().strip()
                model_output = str(res.get("model_output", "")).lower()

                is_correct = res['metrics']['accuracy'] == 1.0
                soft_sim = difflib.SequenceMatcher(None, parsed_answer, ground_truth).ratio()
                mentions_gt = 1.0 if ground_truth and ground_truth in model_output else 0.0

                score_main = max(soft_sim, 0.5 * mentions_gt)
                if is_correct:
                    score_main = max(score_main, 1.0)

                metrics = {
                    "accuracy": res['metrics']['accuracy'],
                    "soft_score": round(score_main, 3),
                    "parsed_answer": parsed_answer
                }

            # --- SAVE RESULT ---
            record = {
                "id": item['id'], # <--- NOW USES THE STABLE ID
                "task_type": item['type'],
                "is_discovery": item.get('is_discovery', False),
                "model_name": model,
                "doc_name": item['doc_name'],
                "ground_truth": item['target'],
                "full_output": res.get('model_output', ''),
                "score": score_main,
                "metrics": metrics
            }
            results.append(record)
            
            # Rate Limit Safety
            time.sleep(0.5)

    # Save Final JSON
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Done! Results saved to 'results.json'.") 
    print(f"👉 NOW you can run 'streamlit run architecture_lab.py' successfully.")

if __name__ == "__main__":
    run()
