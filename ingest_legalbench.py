import os
import requests
import tarfile
import io
import pandas as pd
import json
import random

# --- CONFIGURATION ---
ARCHIVE_URL = "https://huggingface.co/datasets/nguha/legalbench/resolve/main/data.tar.gz"
OUTPUT_FILE = "data/legalbench_data.json"

# Tasks we want to map
SELECTED_TASKS = {
    "unfair_tos": "Is this clause potentially unfair to the consumer?",
    "consumer_contracts_qa": "Analyze the validity of this consumer contract clause.",
    "privacy_policy_qa": "Does this policy comply with data protection standards?",
    "opp115_data_retention": "Does this clause describe how long user information is stored?",
    "opp115_data_security": "Does this clause describe how user information is protected?",
    "opp115_third_party_sharing_collection": "Does this describe third-party data sharing?",
    "hearsay": "Is this statement considered hearsay?",
    "legal_reasoning_causality": "Does the cause logically lead to the effect in a legal context?",
    "contract_qa": "Answer the question based on the contract text."
}

def generate_fallback_samples():
    """Generates fallback sample data if the download fails to allow continued processing."""
    print("WARNING: Download yielded 0 results. Switching to synthetic fallback dataset.")
    print("         This ensures the ingestion pipeline can continue for testing.")

    fallback_samples = []
    
    # 1. Synthetic Hearsay (Logic)
    fallback_samples.append({
        "id": "lb_synth_hearsay_1",
        "type": "reasoning",
        "doc_name": "Synthetic_Hearsay",
        "is_discovery": False,
        "input": {
            "rule": "Hearsay is an out-of-court statement offered to prove the truth of the matter asserted.",
            "facts": "Witness A says: 'I heard B say that the light was red.'",
            "question": "Is this statement considered hearsay?",
            "options": ["Yes", "No"]
        },
        "target": "Yes"
    })

    # 2. Synthetic GDPR (Knowledge)
    fallback_samples.append({
        "id": "lb_synth_gdpr_1",
        "type": "reasoning",
        "doc_name": "Synthetic_GDPR",
        "is_discovery": False,
        "input": {
            "rule": "",
            "facts": "We store your data indefinitely for marketing purposes without encryption.",
            "question": "Does this policy comply with data protection standards?",
            "options": ["Yes", "No"]
        },
        "target": "No"
    })
    
    return fallback_samples

def ingest():
    print("Starting LegalBench ingestion...")
    cleaned_dataset = []
    
    # 1. Try Downloading Real Data
    try:
        print(f"   ⬇️  Downloading archive from Hugging Face...")
        response = requests.get(ARCHIVE_URL, stream=True)
        
        if response.status_code == 200:
            with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
                print("   📦 Scanning archive...")
                
                for member in tar.getmembers():
                    # SKIP GHOST FILES (The cause of your previous errors)
                    if os.path.basename(member.name).startswith("._"):
                        continue
                        
                    # Check for Test Files (CSV or TSV)
                    if "test.csv" in member.name or "test.tsv" in member.name:
                        
                        # Identify Task Name from Folder Path
                        parts = member.name.split('/')
                        task_name = next((t for t in SELECTED_TASKS if t in parts), None)
                        
                        if task_name:
                            # Determine Separator
                            sep = '\t' if member.name.endswith('.tsv') else ','
                            
                            f = tar.extractfile(member)
                            try:
                                # Safe Parsing
                                df = pd.read_csv(f, sep=sep, on_bad_lines='skip', engine='python')
                                if not df.empty:
                                    # Limit to 5 rows to keep it fast
                                    df = df.head(170)
                                    
                                    for i, row in df.iterrows():
                                        # Normalize Columns
                                        text_col = next((c for c in df.columns if str(c).lower() in ['text', 'input', 'clause', 'sentence']), None)
                                        ans_col = next((c for c in df.columns if str(c).lower() in ['answer', 'label', 'target']), None)
                                        
                                        if text_col and ans_col:
                                            cleaned_dataset.append({
                                                "id": f"lb_{task_name}_{i}",
                                                "type": "reasoning",
                                                "doc_name": f"LegalBench_{task_name}",
                                                "is_discovery": False,
                                                "input": {
                                                    "rule": "",
                                                    "facts": str(row[text_col]),
                                                    "question": SELECTED_TASKS[task_name],
                                                    "options": ["Yes", "No"]
                                                },
                                                "target": str(row[ans_col])
                                            })
                            except Exception as e:
                                pass # Skip bad files silently
        else:
            print(f"   ❌ HTTP Error: {response.status_code}")

    except Exception as e:
        print(f"   ❌ Download/Extraction failed: {e}")

    # 2. Check results and activate fallback samples if necessary
    if len(cleaned_dataset) == 0:
        cleaned_dataset = generate_fallback_samples()
    else:
        print(f"Successfully extracted {len(cleaned_dataset)} real tasks.")

    # 3. Add custom EU task (always)
    cleaned_dataset.append({
        "id": "eu_gdpr_rtbf",
        "type": "reasoning",
        "doc_name": "EU_GDPR_RightToErasure",
        "is_discovery": False,
        "input": {
            "rule": "GDPR Art 17: Right to Erasure applies UNLESS processing is necessary for compliance with a legal obligation.",
            "facts": "User requests deletion. Company refuses because they must keep records for 5 years under Tax Law.",
            "question": "Is the refusal compliant?",
            "options": ["Yes", "No"]
        },
        "target": "Yes"
    })

    # 4. Save
    if not os.path.exists("data"):
        os.makedirs("data")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(cleaned_dataset, f, indent=2)

    print(f"Final Status: Saved {len(cleaned_dataset)} reasoning items to '{OUTPUT_FILE}'")

if __name__ == "__main__":
    ingest()