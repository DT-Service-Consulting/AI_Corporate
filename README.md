# Router-eval: LAMMR — Legal Adaptive Multi-Model Router

Hybrid routing system for legal AI workloads. Dynamically assigns requests to specialized extraction or reasoning models using deterministic rules, a learned classifier, and an optional fluid-collaboration layer (FC-LAMMR).

Models:
- **Maverick (17B)** — extraction model, lower cost
- **Llama-3.3 (70B)** — reasoning model, higher capability

---

## Repository Structure

```
Router-eval/
├── core_logic.py                  # Shared model calls, scoring, cost estimation
├── intelligent_router.py          # Rule-based router (keyword / length heuristics)
├── run_experiment.py              # Baseline generation (both models on all tasks)
├── run_hybrid_system.py           # Hybrid rule+learning routing run
├── evaluate_research_ready.py     # Multi-seed bootstrap evaluation + significance tests
├── run_component_ablations.py     # Extraction chunking / NOT_FOUND ablations
├── run_split_robustness.py        # Dynamic stratified split robustness runner
├── ingest_legalbench.py           # LegalBench ingestion pipeline
├── ingest_tenders.py              # Tender document ingestion
├── mine_extraction_tasks.py       # Clause-level extraction QA mining
├── app.py                         # Streamlit dashboard
├── architecture_lab.py            # ML router training + visualization
├── fc_lammr/                      # FC-LAMMR package (fluid collaboration layer)
│   ├── run_fc_lammr_hybrid_test.py   # FC-LAMMR evaluation runner
│   ├── rescore_fc_lammr_results.py   # Offline parser-correction sensitivity analysis
│   ├── fc_lammr_router.py
│   ├── pattern_recognition_layer.py
│   ├── fluid_rerouting_layer.py
│   ├── evaluation_layer.py
│   ├── tom_inference_layer.py
│   └── utils/
├── paper/                         # Unpublished paper drafts (gitignored)
├── docs/                          # Analysis notes (gitignored)
├── data/                          # Datasets (gitignored — contains real contract data)
├── outputs/                       # Run outputs (gitignored)
│   ├── fc_lammr/                  # FC-LAMMR hybrid results
│   ├── research_eval*/            # evaluate_research_ready.py outputs
│   └── component_ablations/
├── results/                       # Evaluation artifacts (gitignored)
│   ├── checkpoints/               # FC-LAMMR run checkpoints
│   ├── fc_lammr/                  # FC-LAMMR summaries and logs
│   └── errors/
├── project_secrets.py             # Azure credentials — gitignored, never commit
├── requirements.txt
└── .gitignore
```

---

## Prerequisites

- Python 3.10+
- Azure AI Foundry access (Maverick + Llama-3.3-70B deployments)
- Azure Document Intelligence (optional, for PDF ingestion)

---

## Installation

```bash
git clone <repo-url>
cd Router-eval
python -m venv venv
# Windows
.\venv\Scripts\Activate.ps1
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

---

## Configuration

Create `project_secrets.py` at the repo root (gitignored):

```python
# Azure AI Foundry
AZURE_LLAMA_ENDPOINT = "https://<your-resource>.openai.azure.com/"
AZURE_LLAMA_KEY = "<your-key>"
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"

# Deployment names
EXTRACTION_DEPLOYMENT_NAME = "<maverick-deployment>"
REASONING_DEPLOYMENT_NAME  = "<llama70b-deployment>"
TOMIL_DEPLOYMENT_NAME      = "<tomil-deployment>"   # FC-LAMMR only

# Document Intelligence (optional — only needed for PDF ingestion)
DOC_INTEL_ENDPOINT = "https://<your-resource>.cognitiveservices.azure.com/"
DOC_INTEL_KEY      = "<your-key>"
```

---

## Data Setup

Data files are gitignored. Obtain or regenerate locally before running experiments.

### Ingest LegalBench reasoning tasks
```bash
python ingest_legalbench.py
# writes: data/legalbench_data.json
```

### Ingest tender documents
```bash
python ingest_tenders.py
# reads:  data/tenders/*.pdf
# writes: data/real_tenders.json
```

### Mine clause-level extraction tasks
```bash
python mine_extraction_tasks.py \
  --input data/real_tenders.json \
  --output data/real_tenders_extraction_qa.json \
  --target-min 50
# Generates extraction QA with hard negatives (NOT_FOUND) and long-doc stress variants.
# run_experiment.py and run_hybrid_system.py auto-detect this file when present.
```

---

## Running Experiments

### Step 1 — Generate baselines

Evaluates both models on all tasks. Produces `results.json` + split manifest.

```bash
python run_experiment.py \
  --seed 42 \
  --test-size 0.2 \
  --val-size 0.2 \
  --rate-limit-s 0.5
# outputs: results.json, split_manifest.json, run_metadata.json
```

### Step 2 — Run hybrid routing

Applies rule+learning routing over the test split.

```bash
python run_hybrid_system.py \
  --split-filter test \
  --router-variant full
# output: hybrid_system_results.json
```

Ablation variants:
```bash
python run_hybrid_system.py --split-filter test --router-variant no_keyword       --output hybrid_no_keyword.json
python run_hybrid_system.py --split-filter test --router-variant no_length        --output hybrid_no_length.json
python run_hybrid_system.py --split-filter test --router-variant no_reasoning_override --output hybrid_no_override.json
```

### Step 3 — Publication-ready evaluation

Multi-seed bootstrap evaluation, oracle-gap analysis, and paired significance tests.

```bash
python evaluate_research_ready.py \
  --results results.json \
  --output-dir outputs/research_eval \
  --seeds 42,43,44,45,46 \
  --bootstrap-samples 4000 \
  --ci-level 0.95 \
  --task-balance-power 1.5 \
  --uncertainty-threshold 0.7 \
  --reference-method always_maverick
```

Outputs written to `outputs/research_eval/`:
| File | Contents |
|------|----------|
| `method_summary.csv` | mean/std per method across seeds |
| `method_by_seed.csv` | per-seed results |
| `task_breakdown.csv` | extraction vs reasoning split |
| `oracle_gap_summary.csv` | regret-to-oracle + CI |
| `paired_significance.csv` | paired bootstrap p-values |
| `instance_level_scores.csv` | per-instance scores |
| `task_imbalance_report.json` | class imbalance warning |
| `error_cases_seed_*.json` | top regret cases |

Recursive routing policy:
```bash
python evaluate_research_ready.py \
  --results results.json \
  --output-dir outputs/research_eval \
  --seeds 42,43,44,45,46 \
  --recursive-max-depth 3 \
  --recursive-low-confidence 0.62 \
  --recursive-high-confidence 0.78
```

### Step 4 (optional) — Robustness and ablation runs

```bash
# Dynamic stratified split robustness
python run_split_robustness.py --execute

# Extraction chunking / NOT_FOUND normalization ablations
python run_component_ablations.py --execute
```

---

## FC-LAMMR (Fluid Collaboration Layer)

FC-LAMMR adds a pattern-recognition and fluid-rerouting layer on top of the base router.

### Run FC-LAMMR evaluation

```bash
python -m fc_lammr.run_fc_lammr_hybrid_test \
  --split-filter test \
  --split-manifest split_manifest.json \
  --output outputs/fc_lammr/fc_lammr_hybrid_results.json \
  --task-sleep 1.0 \
  --max-reasoning-calls 400 \
  --checkpoint-interval 25
```

Key flags:
| Flag | Default | Notes |
|------|---------|-------|
| `--task-sleep` | `0.5` | Seconds between tasks. Use `1.0` on live runs to avoid 429s |
| `--max-reasoning-calls` | unlimited | Hard cap on 70B calls; use `400` as conservative budget |
| `--checkpoint-interval` | `50` | Tasks between checkpoint writes |
| `--resume-from` | — | Path to checkpoint JSON to resume an interrupted run |
| `--prl-threshold` | `0.82` | Pattern recognition confidence threshold |
| `--cold-prl-threshold` | `0.91` | Threshold when pattern library is cold |
| `--reroute-threshold` | `0.65` | Fluid rerouting trigger threshold |

Resume an interrupted run:
```bash
python -m fc_lammr.run_fc_lammr_hybrid_test \
  --resume-from results/checkpoints/fc_lammr_checkpoint_<N>_<timestamp>.json \
  --output outputs/fc_lammr/fc_lammr_hybrid_results.json
```

### Offline parser-correction sensitivity analysis

Re-scores completed FC-LAMMR results with corrected answer-format parsing (raw outputs unchanged).

```bash
python -m fc_lammr.rescore_fc_lammr_results \
  --input  outputs/fc_lammr/fc_lammr_hybrid_results_all.json \
  --output outputs/fc_lammr/fc_lammr_hybrid_results_all_parser_corrected.json \
  --summary-json results/fc_lammr/fc_lammr_parser_corrected_summary.json \
  --summary-md  results/fc_lammr/fc_lammr_parser_corrected_summary.md
```

### Run FC-LAMMR tests

```bash
python -m pytest fc_lammr/test_fc_lammr.py -v
```

---

## Streamlit Dashboard

```bash
streamlit run architecture_lab.py
```

Requires `results.json` and `hybrid_system_results.json` at the repo root (generated by Steps 1–2).

---

## Scoring

- **Extraction tasks** — overlap-oriented metrics (Jaccard / F2 family)
- **Reasoning tasks** — correctness-based scoring aligned with LegalBench criteria
- **Zero-score rate** — fraction of tasks scoring exactly 0 (complete failure); reported per policy and task family
- **Cost** — estimated from token counts × per-token Azure rate; secondary indicator only

---

## Security

`project_secrets.py` is gitignored. Never commit it. Rotate keys if accidentally exposed.

Files excluded from git (see `.gitignore`):
- `project_secrets.py` — Azure API credentials
- `paper/` — unpublished research
- `data/` — real contract data and benchmark datasets
- `outputs/`, `results/` — generated artifacts
- `fc_lammr/pattern_library.json`, `fc_lammr/risk_register.json` — proprietary internals
- LaTeX build artifacts (`*.aux`, `*.bbl`, etc.)
