import difflib
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
import string
import collections

# --- CONFIGURATION: MODELS ---
# These must match your Azure AI Foundry deployment names exactly.
MODELS_TO_TEST = [
    "Llama-3.3-70B-Instruct",
    "Llama-4-Maverick-17B-128E-Instruct-FP8"
]

# --- CONFIGURATION: SECRETS ---
try:
    import project_secrets
    ENDPOINT = project_secrets.AZURE_LLAMA_ENDPOINT
    KEY = project_secrets.AZURE_LLAMA_KEY
except ImportError:
    print("⚠️ CRITICAL ERROR: 'project_secrets.py' not found.")
    ENDPOINT = None
    KEY = None

# --- CLIENT INIT ---
client = None
if ENDPOINT and KEY:
    try:
        client = ChatCompletionsClient(
            endpoint=ENDPOINT,
            credential=AzureKeyCredential(KEY)
        )
    except Exception as e:
        print(f"⚠️ Error initializing Azure Client: {e}")

def normalize_text(text):
    """
    Removes quotes, punctuation, and extra whitespace for fair comparison.
    Converts "month"" -> "month"
    """
    if not text:
        return ""
    # Remove all punctuation (including quotes)
    remove_tokens = [".", ",", ";", ":", "'", '"', "(", ")"]
    text = text.lower()
    for token in remove_tokens:
        text = text.replace(token, "")
    return text.strip()

# --- METRIC FUNCTIONS ---
def get_jaccard(gt, pred):
    """
    Calculates Intersection over Union (IoU) of words.
    Score: 0.0 (No overlap) to 1.0 (Perfect match).
    """
    """
    Calculates Jaccard Similarity on token sets.
    """
    gt_clean = normalize_text(gt)
    pred_clean = normalize_text(pred)
    
    gt_words = set(gt_clean.split())
    pred_words = set(pred_clean.split())
    
    if len(gt_words) == 0 and len(pred_words) == 0:
        return 1.0
    
    intersection = len(gt_words.intersection(pred_words))
    union = len(gt_words.union(pred_words))
    
    return intersection / union if union > 0 else 0.0

def calculate_advanced_metrics(prediction, ground_truth):
    """
    Calculates Precision, Recall, F1, and F2 (Beta=2).
    """
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(ground_truth).split()
    
    # 1. Laziness Detection
    # If model says "no related clause" but there IS a ground truth -> FAIL
    is_lazy = False
    if "no related clause" in prediction.lower() and len(truth_tokens) > 0:
        is_lazy = True

    # 2. Token Overlap
    common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())
    
    if len(pred_tokens) == 0:
        precision = 0.0
    else:
        precision = num_same / len(pred_tokens)
        
    if len(truth_tokens) == 0:
        recall = 0.0
    else:
        recall = num_same / len(truth_tokens)
    
    # 3. F1 Score (Balanced)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (2 * precision * recall) / (precision + recall)
        
    # 4. F2 Score (Recall-Weighted, crucial for Law)
    # Formula: (5 * P * R) / (4 * P + R)
    if (4 * precision) + recall == 0:
        f2 = 0.0
    else:
        f2 = (5 * precision * recall) / ((4 * precision) + recall)
        
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "f2": round(f2, 3),
        "jaccard": round(get_jaccard(ground_truth, prediction), 3),
        "is_lazy": is_lazy
    }

# --- LLM FUNCTIONS ---
def call_azure_llm(prompt, model_name):
    """
    Sends a request to the specific model deployed on Azure.
    """
    if not client: return "Error: No Client"
    try:
        response = client.complete(
            messages=[
                SystemMessage(content="You are a legal AI. Extract relevant text exactly. If nothing is found, say 'No related clause'."),
                UserMessage(content=prompt),
            ],
            model=model_name,
            temperature=0.0,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Azure Error: {e}"

def evaluate_single_extraction(input_text, ground_truth, model_name):
    """
    Standardized Extraction Pipeline (v3.1)
    - Input: Document Text + "Query" (Ground Truth target)
    - Output: Metrics (F2, Precision, Recall, Laziness)
    """
    
    # --- 1. UNIVERSAL PROMPT (v3.1) ---
    # Engineered for Llama 3, Phi-4, and GPT-4 consistency.
    prompt = f"""
    ### ROLE
    You are a high-precision Legal Audit AI. Your job is to extract text from contracts with 100% fidelity.

    ### TASK
    Find and extract the specific text segment from the DOCUMENT below that matches the QUERY.

    ### QUERY / REQUIREMENT
    "{ground_truth}"

    ### DOCUMENT
    "{input_text}"

    ### STRICT OUTPUT RULES
    1.  **Extraction Only:** Return ONLY the exact text found in the document. Do not change a single word or punctuation mark.
    2.  **No Conversational Filler:** Do NOT say "Here is the text" or "The clause is:". Just output the text.
    3.  **Negative Constraint:** If the document does NOT contain text matching the requirement, output exactly: "No related clause"
    4.  **Scope:** Do not extract the whole paragraph if only one sentence matches. Be precise.

    ### FINAL ANSWER
    """

    # --- 2. MODEL CALL ---
    model_output = call_azure_llm(prompt, model_name)

    # --- 3. SCORING (ContractEval Logic) ---
    metrics = calculate_advanced_metrics(model_output, ground_truth)
    
    return {
        "model_used": model_name,
        "model_output": model_output,
        "ground_truth": ground_truth,
        "metrics": metrics 
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
    raw_output = call_azure_llm(prompt, model_name)
    
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