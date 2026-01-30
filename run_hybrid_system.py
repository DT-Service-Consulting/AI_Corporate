import json
import time
import os
from tqdm import tqdm
from core_logic import evaluate_single_extraction, evaluate_reasoning
from intelligent_router import IntelligentRouter

# --- CONFIGURATION ---
ROUTER = IntelligentRouter()
OUTPUT_FILE = "hybrid_system_results.json"

def load_data():
    data = []
    
    # List of files to attempt loading
    files_to_load = ["data/real_tenders.json", "data/legalbench_data.json"]
    
    for filename in files_to_load:
        if os.path.exists(filename):
            try:
                with open(filename, "r") as f:
                    file_content = f.read().strip()
                    if not file_content:
                        print(f"⚠️ Warning: '{filename}' is empty. Skipping.")
                        continue
                        
                    loaded_json = json.loads(file_content)
                    if isinstance(loaded_json, list):
                        data.extend(loaded_json)
                        print(f"   ✅ Loaded {len(loaded_json)} items from {filename}")
                    else:
                        print(f"   ⚠️ Warning: '{filename}' did not contain a list. Skipping.")
            except json.JSONDecodeError:
                print(f"   ❌ Error: '{filename}' is corrupted (Invalid JSON). Please delete and re-generate it.")
            except Exception as e:
                print(f"   ❌ Error loading '{filename}': {e}")
        else:
            print(f"   ℹ️ Note: '{filename}' not found (skipping).")
            
    return data

def run_hybrid_test():
    dataset = load_data()
    
    if not dataset:
        print("❌ Critical Error: No data loaded. Please run 'ingest_tenders.py' and 'ingest_legalbench.py' first.")
        return

    results = []
    print(f"\n🚀 Starting Hybrid Ecosystem Test on {len(dataset)} tasks...")
    
    # Track routing stats
    routing_stats = {"extraction": 0, "reasoning": 0}
    
    for item in tqdm(dataset):
        try:
            # 1. THE ROUTER DECISION
            if item['type'] == 'extraction':
                user_query = item.get('query_intent', "Extract the clause related to " + str(item.get('target', '')))
            else:
                user_query = item['input']['question']
                
            # Ask the Router which model to use
            selected_model, intent = ROUTER.route(user_query, str(item.get('input', '')))
            routing_stats[intent] += 1
            
            # 2. EXECUTE WITH SELECTED SPECIALIST
            score = 0.0
            res = {}
            
            if item['type'] == 'extraction':
                query = item.get('query_intent', item.get('target', ''))
                res = evaluate_single_extraction(item['input'], query, selected_model)
                score = res['metrics'].get('f2', 0.0) if not item.get('is_discovery', False) else 0.0
                
            elif item['type'] == 'reasoning':
                res = evaluate_reasoning(
                    item['input'].get('rule', ''), 
                    item['input']['facts'], 
                    {
                        'question': item['input']['question'], 
                        'options': item['input'].get('options', []), 
                        'answer': item['target']
                    }, 
                    selected_model
                )
                score = res['metrics']['accuracy']

            # 3. LOG RESULT
            results.append({
                "id": item['id'],
                "router_intent": intent,
                "model_selected": selected_model,
                "score": score,
                "ground_truth": item['target'],
                "output": res.get('model_output', '')
            })
            
            time.sleep(0.5) # Rate limit safety
            
        except Exception as e:
            print(f"   ⚠️ Error processing item {item.get('id', 'unknown')}: {e}")

    # 4. SAVE & REPORT
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n✅ Hybrid System Test Complete.")
    print(f"📊 Routing Decisions: {routing_stats}")
    
    # Calculate Average Score
    valid_scores = [r['score'] for r in results if not str(r['id']).startswith('tender_')]
    if valid_scores:
        avg_score = sum(valid_scores) / len(valid_scores)
        print(f"🏆 Final Hybrid System Score: {avg_score*100:.2f}%")

if __name__ == "__main__":
    run_hybrid_test()