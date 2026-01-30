import os
import json
import time
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
import project_secrets

# --- CONFIGURATION ---
INPUT_FOLDER = "data/tenders" 
OUTPUT_FILE = "data/real_tenders.json"

def ingest():
    print("🚀 Starting REAL Azure Document Ingestion...")

    # 1. Validation
    if not os.path.exists(INPUT_FOLDER):
        print(f"   ❌ Error: Folder '{INPUT_FOLDER}' not found.")
        return

    pdf_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"   ⚠️ No PDFs found in '{INPUT_FOLDER}'. Please add files.")
        return

    # 2. Connect to Azure
    print("   🔌 Connecting to Azure Document Intelligence...")
    try:
        client = DocumentIntelligenceClient(
            endpoint=project_secrets.DOC_INTEL_ENDPOINT,
            credential=AzureKeyCredential(project_secrets.DOC_INTEL_KEY)
        )
    except Exception as e:
        print(f"   ❌ Connection Failed: {e}")
        return

    dataset = []

    # 3. Process Files
    print(f"   📄 Found {len(pdf_files)} PDFs. Processing...")
    
    for filename in pdf_files:
        file_path = os.path.join(INPUT_FOLDER, filename)
        print(f"      -> Parsing: {filename}...", end=" ")
        
        try:
            with open(file_path, "rb") as f:
                # --- THE FIX IS HERE ---
                # We use 'body=f' instead of 'analyze_request=f'
                poller = client.begin_analyze_document(
                    model_id="prebuilt-read", 
                    body=f,
                    content_type="application/pdf"
                )
                result: AnalyzeResult = poller.result()
                
                # Extract full text
                full_text = result.content
                
                # Create a Task Record
                record = {
                    "id": f"tender_{filename.replace('.', '_').replace(' ', '_')}",
                    "type": "extraction",
                    "doc_name": filename,
                    "is_discovery": True,
                    "query_intent": "Extract all indemnity, liability, and termination clauses.",
                    "input": full_text,
                    "target": "N/A (Discovery Task)"
                }
                dataset.append(record)
                print("✅ Done.")
                
        except Exception as e:
            print(f"❌ Failed: {e}")

    # 4. Save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(dataset, f, indent=2)
        
    print(f"\n🎉 Success! Saved {len(dataset)} processed tenders to '{OUTPUT_FILE}'")

if __name__ == "__main__":
    ingest()