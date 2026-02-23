import re
from typing import Tuple

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")
LONG_NUM_RE = re.compile(r"\b\d{8,}\b")  # conservative: 8+ digits

def redact_pii(text: str) -> Tuple[str, bool]:
    changed = False
    t = text

    t2 = EMAIL_RE.sub("[REDACTED_EMAIL]", t)
    changed |= (t2 != t)
    t = t2

    t2 = PHONE_RE.sub("[REDACTED_PHONE]", t)
    changed |= (t2 != t)
    t = t2

    t2 = LONG_NUM_RE.sub("[REDACTED_NUMBER]", t)
    changed |= (t2 != t)
    t = t2

    return t, changed

def safety_preamble() -> str:
    return "Note: Please don’t share sensitive personal information (SSN, full policy/claim numbers, banking details) in chat.\n"

def account_boundary() -> str:
    return "I can’t access personal Travelers accounts or internal claim systems. I can help you find the right official page and steps.\n"
