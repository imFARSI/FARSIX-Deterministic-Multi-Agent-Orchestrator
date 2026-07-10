import re
from typing import Any

def is_guardrails_block(text: str) -> bool:
    return text.startswith("GUARDRAILS_BLOCK")

def check_content_safety(text: str) -> bool:
    """Check input text for prompt injections or safety bypass attempts."""
    text_lower = text.lower()
    unsafe_patterns = [
        "ignore all previous instructions",
        "bypass safety",
        "disregard rules",
        "jailbreak",
        "system prompt",
    ]
    for pattern in unsafe_patterns:
        if pattern in text_lower:
            return False  # Not safe
    return True  # Safe

def check_for_safety_violations(text: str) -> bool:
    """Check agent output for safety rule violations."""
    if text.startswith("GUARDRAILS_BLOCK"):
        return False
        
    unsafe_patterns = [
        r"\b(ignore all|disregard|override)\b.{0,20}\b(safety|rules|instructions)\b",
    ]
    for pattern in unsafe_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True  # Violation found
    return False  # No violations

def check_logical_consistency(text: str) -> bool:
    """Check agent output for logical consistency."""
    if text.startswith("GUARDRAILS_BLOCK"):
        return True
        
    # Example: If it mentions risk_score=0 but also immediate_action_required=True
    text_lower = text.lower()
    if "risk_score" in text_lower and "immediate_action_required" in text_lower:
        if re.search(r'risk_score["\s:]*0', text_lower) and re.search(r'immediate_action_required["\s:]*true', text_lower):
            return False  # Inconsistent
    return True  # Consistent

def detect_hallucinations(text: str) -> bool:
    """Check agent output for common hallucination or refusal markers."""
    if text.startswith("GUARDRAILS_BLOCK"):
        return False
        
    hallucination_patterns = [
        r"\b(100%|0%) (certainty|confidence|probability)\b",
        r"\bI cannot\b.*\breport\b",
        r"\bAs an AI language model\b",
        r"\bI don't have access\b",
        r"\bI am a language model\b"
    ]
    for pattern in hallucination_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True  # Hallucination detected
    return False  # No hallucination

def check_output_completeness(text: str) -> bool:
    """Ensure the text output is sufficiently long and structured."""
    if not isinstance(text, str):
        return False
    if text.startswith("GUARDRAILS_BLOCK"):
        return True
    if len(text.strip()) < 50:
        return False
        
    # Basic check for structure/sections if it's a longer text
    if len(text.strip()) > 200:
        has_structure = any(
            marker in text
            for marker in ["##", "**", "1.", "- ", "•", "Risk", "Action", "Priority", "Summary", "Findings"]
        )
        if not has_structure:
            return False
            
    return True
