import re

def detect_intent(message: str) -> str:
    m = message.lower()

    if re.search(r"\b(pay|payment|bill|billing)\b", m):
        return "billing"

    if re.search(r"\b(find|locate)\b.*\bagent\b|\bagent\b", m):
        return "agent"

    if re.search(r"\b(contact|phone|call|representative|support)\b", m):
        return "contact"

    if "claim" in m or re.search(r"\b(accident|damage|loss)\b", m):
        if re.search(r"\b(status|track|check)\b", m):
            return "claims_status"
        if re.search(r"\b(upload|document|photo|file)\b", m):
            return "claims_upload"
        if re.search(r"\b(file|report|start|submit)\b", m):
            return "claims_file"
        return "claims_file"

    if re.search(r"\b(quote|coverage|policy|auto|home|renters|business)\b", m):
        return "product_info"

    return "general"
