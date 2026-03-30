import collections
import difflib
import os
import re
import string

from openai import AzureOpenAI

# --- CONFIGURATION: MODELS ---
# These must match your Azure AI Foundry deployment names exactly.
MODELS_TO_TEST = [
    "gpt-4.1-2",
    "gpt-4o-mini-2",
]
REASONING_MODEL_NAME = "gpt-4.1-2"
EXTRACTION_MODEL_NAME = "gpt-4o-mini-2"

MODEL_LABELS = {
    REASONING_MODEL_NAME: "GPT-4.1",
    EXTRACTION_MODEL_NAME: "GPT-4o Mini",
}

# Approximate per-1K-token rates used for local experiment cost estimation.
# Update these if your Azure pricing or region differs.
MODEL_COST_PER_1K_TOKENS = {
    REASONING_MODEL_NAME: 0.010,
    EXTRACTION_MODEL_NAME: 0.00075,
}


def format_model_label(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name)


# --- CONFIGURATION: SECRETS ---
try:
    import project_secrets

    ENDPOINT = project_secrets.AZURE_LLAMA_ENDPOINT
    KEY = project_secrets.AZURE_LLAMA_KEY
    API_VERSION = getattr(
        project_secrets,
        "AZURE_OPENAI_API_VERSION",
        os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )
except ImportError:
    print("CRITICAL ERROR: 'project_secrets.py' not found.")
    ENDPOINT = None
    KEY = None
    API_VERSION = None

# --- CLIENT INIT ---
client = None
if ENDPOINT and KEY:
    try:
        client = AzureOpenAI(
            api_key=KEY,
            api_version=API_VERSION,
            azure_endpoint=ENDPOINT,
        )
    except Exception as e:
        print(f"Error initializing Azure Client: {e}")


NOT_FOUND_TOKEN = "NOT_FOUND"


def normalize_text(text):
    """
    Normalize text for overlap metrics:
    - casing/whitespace cleanup
    - punctuation removal
    - section symbol normalization
    """
    if not text:
        return ""
    text = str(text)
    text = text.replace("§", " section ")
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("`", "'")
    text = re.sub(r"\b(sec\.|cl\.)\b", "section", text, flags=re.IGNORECASE)
    text = text.lower()
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()


def canonicalize_missing(text: str) -> str:
    t = normalize_text(text)
    markers = {
        "not found",
        "notfound",
        "no related clause",
        "no clause found",
        "no relevant clause",
        "none",
        "na",
        "n a",
    }
    if t in markers:
        return NOT_FOUND_TOKEN
    return str(text)


def postprocess_extraction_output(text: str) -> str:
    if not text:
        return ""
    out = str(text).strip()
    out = re.sub(r"^```[a-zA-Z]*\s*", "", out).strip()
    out = re.sub(r"\s*```$", "", out).strip()
    out = re.sub(r"^(answer|final answer|output)\s*:\s*", "", out, flags=re.IGNORECASE).strip()
    out = re.sub(r"^\s*['\"]|['\"]\s*$", "", out).strip()
    out = re.sub(r"\s+", " ", out).strip()
    canon = canonicalize_missing(out)
    if canon == NOT_FOUND_TOKEN:
        return NOT_FOUND_TOKEN
    return out


def get_jaccard(gt, pred):
    """Calculates Jaccard similarity on token sets."""
    gt_clean = normalize_text(gt)
    pred_clean = normalize_text(pred)

    gt_words = set(gt_clean.split())
    pred_words = set(pred_clean.split())

    if len(gt_words) == 0 and len(pred_words) == 0:
        return 1.0

    intersection = len(gt_words.intersection(pred_words))
    union = len(gt_words.union(pred_words))

    return intersection / union if union > 0 else 0.0


def calculate_advanced_metrics(prediction, ground_truth, canonicalize_missing_enabled: bool = True):
    """
    Calculates Precision, Recall, F1, and F2 (Beta=2),
    with canonical NOT_FOUND handling.
    """
    pred_raw = postprocess_extraction_output(prediction)
    gt_raw = canonicalize_missing(ground_truth) if canonicalize_missing_enabled else str(ground_truth)
    if not canonicalize_missing_enabled:
        pred_raw = str(prediction or "").strip()

    if gt_raw == NOT_FOUND_TOKEN and pred_raw == NOT_FOUND_TOKEN:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "f2": 1.0,
            "jaccard": 1.0,
            "is_lazy": False,
        }

    if gt_raw == NOT_FOUND_TOKEN and pred_raw != NOT_FOUND_TOKEN:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "f2": 0.0,
            "jaccard": 0.0,
            "is_lazy": False,
        }

    pred_tokens = normalize_text(pred_raw).split()
    truth_tokens = normalize_text(gt_raw).split()

    is_lazy = pred_raw == NOT_FOUND_TOKEN and len(truth_tokens) > 0

    common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())

    precision = 0.0 if len(pred_tokens) == 0 else num_same / len(pred_tokens)
    recall = 0.0 if len(truth_tokens) == 0 else num_same / len(truth_tokens)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (2 * precision * recall) / (precision + recall)

    if (4 * precision) + recall == 0:
        f2 = 0.0
    else:
        f2 = (5 * precision * recall) / ((4 * precision) + recall)

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "f2": round(f2, 3),
        "jaccard": round(get_jaccard(gt_raw, pred_raw), 3),
        "is_lazy": is_lazy,
    }


# --- LLM FUNCTIONS ---
def call_azure_llm(prompt, model_name, system_prompt=None):
    """Sends a request to the specific model deployed on Azure OpenAI."""
    if not client:
        return "Error: No Client"
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt or "You are a careful legal AI assistant.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            model=model_name,
            temperature=0.0,
            max_tokens=1000,
        )
        return str(response.choices[0].message.content or "").strip()
    except Exception as e:
        return f"Azure Error: {e}"


def build_extraction_prompt(requirement_text: str, document_text: str) -> str:
    return f"""
    ### ROLE
    You are a high-precision Legal Extraction AI.

    ### TASK
    Extract the exact span from DOCUMENT that satisfies REQUIREMENT.

    ### REQUIREMENT
    "{requirement_text}"

    ### DOCUMENT
    "{document_text}"

    ### STRICT OUTPUT RULES
    1. Return only the exact copied span from DOCUMENT. No paraphrase.
    2. Do not add explanations, headers, or formatting.
    3. If the span is missing, output exactly: {NOT_FOUND_TOKEN}
    4. Output one span only.
    """


def chunk_document(text: str, chunk_size: int = 2600, overlap: int = 300):
    src = str(text or "")
    if len(src) <= chunk_size:
        return [src]
    chunks = []
    start = 0
    while start < len(src):
        end = min(len(src), start + chunk_size)
        chunks.append(src[start:end])
        if end >= len(src):
            break
        start = max(0, end - overlap)
    return chunks


def rank_chunks_by_query(chunks, requirement_text: str):
    q_tokens = set(normalize_text(requirement_text).split())
    ranked = []
    for idx, ch in enumerate(chunks):
        c_tokens = set(normalize_text(ch).split())
        overlap = len(q_tokens.intersection(c_tokens))
        density = overlap / (len(q_tokens) + 1e-9)
        ranked.append((density, overlap, -len(ch), idx))
    ranked.sort(reverse=True)
    return [chunks[r[3]] for r in ranked]


def pick_best_candidate(candidates, requirement_text: str):
    cleaned = [postprocess_extraction_output(c) for c in candidates if c]
    cleaned = [c for c in cleaned if c and c != NOT_FOUND_TOKEN]
    if not cleaned:
        return NOT_FOUND_TOKEN
    q_tokens = set(normalize_text(requirement_text).split())
    scored = []
    for c in cleaned:
        c_tokens = set(normalize_text(c).split())
        overlap = len(q_tokens.intersection(c_tokens))
        scored.append((overlap, len(c), c))
    scored.sort(reverse=True)
    return scored[0][2]


def evaluate_single_extraction(
    input_text,
    requirement_text,
    model_name,
    expected_text=None,
    chunk_size: int = 2600,
    chunk_overlap: int = 300,
    top_k_chunks: int = 3,
    long_doc_threshold: int = 6000,
    enable_chunking: bool = True,
    canonicalize_missing_enabled: bool = True,
):
    """
    Extraction evaluator with:
    - strict exact-span output prompt
    - canonical NOT_FOUND
    - long-document chunk retrieval + merge
    """
    if expected_text is None:
        expected_text = requirement_text

    doc_text = str(input_text or "")
    req_text = str(requirement_text or "")

    if enable_chunking and len(doc_text) > long_doc_threshold:
        chunks = chunk_document(doc_text, chunk_size=chunk_size, overlap=chunk_overlap)
        ranked_chunks = rank_chunks_by_query(chunks, req_text)[: max(1, int(top_k_chunks))]
        candidate_outputs = []
        for chunk in ranked_chunks:
            prompt = build_extraction_prompt(req_text, chunk)
            candidate_outputs.append(
                call_azure_llm(
                    prompt,
                    model_name,
                    system_prompt=f"You are a legal AI. Return exact copied span only. If missing, return {NOT_FOUND_TOKEN}.",
                )
            )
        model_output = pick_best_candidate(candidate_outputs, req_text)
    else:
        prompt = build_extraction_prompt(req_text, doc_text)
        model_output = call_azure_llm(
            prompt,
            model_name,
            system_prompt=f"You are a legal AI. Return exact copied span only. If missing, return {NOT_FOUND_TOKEN}.",
        )

    model_output = postprocess_extraction_output(model_output) if canonicalize_missing_enabled else str(model_output or "").strip()
    expected_text = canonicalize_missing(str(expected_text)) if canonicalize_missing_enabled else str(expected_text)

    metrics = calculate_advanced_metrics(
        model_output,
        expected_text,
        canonicalize_missing_enabled=canonicalize_missing_enabled,
    )

    return {
        "model_used": model_name,
        "model_output": model_output,
        "ground_truth": expected_text,
        "metrics": metrics,
    }


def evaluate_reasoning(rule, fact_pattern, correct_answer_obj, model_name):
    """
    Universal Reasoning Evaluator (v3.0)
    - Mode A (IRAC): User provides a RULE + FACTS -> Model applies logic.
    - Mode B (Knowledge QA): User provides TEXT -> Model classifies/answers based on internal legal knowledge.
    """

    # --- 1. PROMPT ENGINEERING ---
    if rule and len(rule.strip()) > 5:
        # MODE A: Strict IRAC Logic (Rule is provided)
        # Used for: Hearsay, Citations, Statutory Interpretation
        prompt = f"""
        ### ROLE
        You are a precision Legal Logic AI. Your task is to apply the provided RULE to the FACTS to answer the QUESTION.

        ### RULE
        {rule}

        ### FACTS
        {fact_pattern}

        ### QUESTION
        {correct_answer_obj['question']}

        ### INSTRUCTIONS
        1.  Think step-by-step: Does the fact pattern satisfy the conditions of the rule?
        2.  Answer strictly based on the provided RULE, even if it conflicts with real-world law.
        3.  Final Answer must be one of: {correct_answer_obj.get('options', ['Yes', 'No'])}
        4.  Format output as: "Reasoning... \nANSWER: <Option>"
        """
    else:
        # MODE B: General Legal Knowledge / Classification (No Rule provided)
        # Used for: Unfair Terms (EU), Privacy Policies (GDPR), Contract QA
        prompt = f"""
        ### ROLE
        You are a Senior Legal Consultant. Analyze the provided TEXT and answer the QUESTION based on general legal principles (e.g., GDPR, Consumer Protection, Contract Law).

        ### TEXT / SCENARIO
        {fact_pattern}

        ### QUESTION
        {correct_answer_obj['question']}

        ### INSTRUCTIONS
        1.  Analyze the legal implications of the text.
        2.  If the question asks for a classification (e.g., Unfair/Fair, Compliant/Non-Compliant), use standard legal definitions.
        3.  Final Answer must be one of: {correct_answer_obj.get('options', ['Yes', 'No', 'Unfair', 'Fair'])}
        4.  Format output as: "Reasoning... \nANSWER: <Option>"
        """

    # --- 2. MODEL CALL ---
    raw_output = call_azure_llm(
        prompt,
        model_name,
        system_prompt="You are a precision legal reasoning assistant. Follow the prompt carefully and give the final answer in the requested ANSWER format.",
    )

    # --- 3. SCORING (Robust Parsing) ---
    # We look for the answer at the end of the text or explicitly tagged
    parsed_answer = "UNKNOWN"
    ground_truth = str(correct_answer_obj['answer']).lower().strip()

    # Strategy 1: strict "ANSWER:" tag
    if "ANSWER:" in raw_output:
        parsed_answer = raw_output.split("ANSWER:")[1].strip().lower()

    # Strategy 2: Heuristic check (if model output is short or clearly contains the keyword)
    if parsed_answer == "UNKNOWN":
        for opt in correct_answer_obj.get('options', []):
            if opt.lower() in raw_output.lower():
                # We count it if the option appears in the last 50 chars (conclusion)
                # or if the entire output is very short (just the answer)
                if len(raw_output) < 50 or opt.lower() in raw_output.lower()[-100:]:
                    parsed_answer = opt.lower()
                    break

    # Exact Match Scoring
    # We check if the parsed answer *contains* the ground truth (handles "Yes." vs "Yes")
    is_correct = (ground_truth in parsed_answer) or (parsed_answer in ground_truth)

    # Fallback: If both are simple Yes/No, ensure strict equality
    if ground_truth in ['yes', 'no'] and parsed_answer in ['yes', 'no']:
        is_correct = (ground_truth == parsed_answer)

    return {
        "model_used": model_name,
        "task_type": "reasoning",
        "model_output": raw_output,
        "parsed_answer": parsed_answer,
        "ground_truth": correct_answer_obj['answer'],
        "metrics": {
            "accuracy": 1.0 if is_correct else 0.0
        }
    }
