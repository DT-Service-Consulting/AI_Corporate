import re

class IntelligentRouter:
    def __init__(self):
        # CONFIGURATION: The Specialist Models
        self.EXTRACTION_MODEL = "Llama-4-Maverick-17B-128E-Instruct-FP8"
        self.REASONING_MODEL = "Llama-3.3-70B-Instruct"
        
        # KEYWORDS: Signals for specific tasks
        
        # 1. Extraction Triggers (High Recall needed)
        self.extraction_keywords = [
            "extract", "find", "locate", "identify", "quote", 
            "clause", "provision", "term", "section", "where is",
            "retrieve", "copy", "highlight", "list"
        ]
        
        # 2. Reasoning Triggers (High Logic needed)
        # Note: These are "Weighted" - if found, they usually override extraction.
        self.reasoning_keywords = [
            "analyze", "evaluate", "assess", "compliant", "valid", 
            "violation", "breach", "why", "reason", "interpret",
            "implication", "consequence", "hearsay", "gdpr", "unfair",
            "compare", "contrast", "risk", "red flag"
        ]

    def classify_intent(self, user_query, document_text=""):
        """Determine whether the query requires extraction (high recall) or reasoning (high intelligence)."""
        query_lower = user_query.lower()
        
        # Rule 1: Strong-signal check (priority: reasoning)
        # If the query contains reasoning-oriented keywords, prefer the reasoning model (70B).
        if any(word in query_lower for word in self.reasoning_keywords):
            return "reasoning"
            
        # RULE 2: The Extraction Check
        if any(word in query_lower for word in self.extraction_keywords):
            return "extraction"
            
        # Rule 3: Document-length heuristic
        # For very large documents (>15k chars), prefer the extraction model (MoE) which handles large inputs well.
        if len(document_text) > 15000:
            return "extraction"
            
        # Rule 4: Default fallback
        # When intent is ambiguous, prefer the reasoning model.
        return "reasoning"

    def route(self, user_query, document_text=""):
        """
        Returns the specific model name and the task type.
        """
        task_type = self.classify_intent(user_query, document_text)
        
        if task_type == "extraction":
            return self.EXTRACTION_MODEL, "extraction"
        else:
            return self.REASONING_MODEL, "reasoning"

# Sanity check (run this file directly to verify router behavior)
if __name__ == "__main__":
    router = IntelligentRouter()
    print("--- Router Diagnostic ---")
    print(f"1. 'Find the indemnity clause' -> {router.route('Find the indemnity clause')}")
    print(f"2. 'Is this indemnity clause valid under NY Law?' -> {router.route('Is this indemnity clause valid under NY Law?')}")
    print(f"3. 'Extract all dates' (Large document) -> {router.route('Extract all dates', 'a'*20000)}")