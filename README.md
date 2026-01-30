# Router-eval: Intelligent Model Routing System

An advanced intelligent routing system for legal document analysis that dynamically selects between specialized extraction and reasoning models based on task characteristics.

## Overview

Router-eval implements multiple routing strategies to optimize cost-accuracy trade-offs when processing legal documents. The system routes queries to either:
- **Maverick (17B)**: Lightweight extraction model for factual information retrieval
- **Llama-3 (70B)**: Powerful reasoning model for complex legal analysis

## Routing Methods

The system implements five core routing strategies plus two baselines:

### 1. **Keyword Router**
Analyzes query text for reasoning-oriented keywords (analyze, evaluate, compliant, valid, breach, etc.). Selects the reasoning model when keywords are detected, otherwise uses the extraction model.

### 2. **Gen 3 ML Router (Random Forest)**
A supervised machine learning router trained on sentence embeddings from historical performance data. Uses a Random Forest classifier to predict which model will perform best for each query based on semantic features.

### 3. **Length Cascade**
A document-length heuristic that routes large documents (>15k characters) to the extraction model (better for batch processing), and smaller documents to the reasoning model.

### 4. **Confidence Cascade**
Inspects the extraction model's output quality. Falls back to the reasoning model if the output is empty, too short (<5 chars), or contains failure phrases ("unable to", "cannot find", "sorry").

### 5. **Oracle (Benchmark)**
Theoretical upper-bound that always selects the model with the higher ground-truth score for evaluation purposes.

### Baselines
- **Pure Maverick**: Always uses the extraction model
- **Pure Llama-3 (70B)**: Always uses the reasoning model

## Project Structure

```
Router-eval/
├── .gitignore                    # Git ignore rules for secrets and outputs
├── app.py                        # Main application entry point
├── architecture_lab.py           # Streamlit visualization and ML router training
├── intelligent_router.py         # Compact rule-based routing implementation
├── ingest_legalbench.py         # LegalBench dataset ingestion pipeline
├── ingest_tenders.py            # Legal tender document ingestion
├── core_logic.py                # Core routing and classification logic
├── run_experiment.py            # Experiment execution script
├── run_hybrid_system.py          # Hybrid system runner
├── project_secrets.py           # Configuration (excluded from git)
├── requirements.txt             # Python dependencies
├── documentation.tex            # LaTeX documentation
├── data/                        # Data directory
│   ├── legalbench_data.json    # LegalBench dataset
│   ├── real_tenders.json       # Real legal tenders
│   └── tenders/               # Tender documents
└── outputs/                    # Generated results (excluded from git)
```

## Installation

### Prerequisites
- Python 3.10+
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone https://github.com/DT-Service-Consulting/AI_Corporate.git
cd Router-eval
```

2. Create a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate     # macOS/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run the Architecture Lab (Visualization)
```bash
streamlit run architecture_lab.py
```
Opens an interactive dashboard comparing all routing strategies with performance metrics and cost analysis.

### Ingest Legal Data
```bash
python ingest_legalbench.py
```
Downloads and processes the LegalBench dataset for training and evaluation.

### Run Experiments
```bash
python run_experiment.py
```
Executes the full evaluation pipeline comparing routing methods.

### Run Hybrid System
```bash
python run_hybrid_system.py
```
Runs the complete hybrid routing system with both baselines and intelligent routers.

## Key Features

- **Multiple routing strategies**: Choose the best approach for your use case
- **Cost-accuracy optimization**: Dynamically balance model cost vs. quality
- **ML-based routing**: Train supervised routers on your data
- **Comprehensive evaluation**: Compare routing methods side-by-side
- **Interactive visualization**: Streamlit dashboard with performance metrics

## Performance

The routing system evaluates accuracy and cost trade-offs:
- Each routing method is benchmarked against ground-truth scores
- Cost is estimated based on model usage percentage (Maverick=cheaper, 70B=more expensive)
- The Oracle provides theoretical maximum performance

## Configuration

Edit these constants in the relevant files:

**architecture_lab.py:**
```python
BASELINE_FILE = "results.json"
SEMANTIC_MODEL_NAME = 'all-MiniLM-L6-v2'
```

**intelligent_router.py:**
- Modify `reasoning_keywords` list for custom keyword routing
- Adjust `route_by_length()` threshold (currently 15k chars)

## Dependencies

See `requirements.txt` for full list. Key packages:
- `pandas`: Data manipulation
- `scikit-learn`: ML router (Random Forest)
- `sentence-transformers`: Semantic embeddings
- `streamlit`: Interactive visualization
- `plotly`: Charts and graphs

## Git Workflow

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feature: Add routing strategy for legal documents"

# Push to main
git push origin main
```

## Security

The `.gitignore` excludes:
- Personal configuration files (`project_secrets.py`)
- Generated outputs and results
- Python cache (`__pycache__/`)
- Virtual environments
- IDE settings
