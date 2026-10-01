"""
Responsible-AI guardrails applied before and after the agents run.

Input side : schema validation, prompt-injection screening, PII redaction,
             length limits.
Output side: grounding check (every itinerary item must exist in the
             knowledge base), budget/consistency checks, mandatory disclaimer.
"""

import re
from datetime import date, datetime

from core.data import CITIES

MAX_TEXT = 1200

INJECTION_PATTERNS = [
    r"ignore (all|any|the|previous|prior|above).{0,20}(instructions|rules|prompt)",
    r"disregard (the|all|previous).{0,20}(instructions|rules)",
    r"(reveal|show|print).{0,20}(system prompt|api key|secret)",
    r"you are now", r"act as (?!a traveller)", r"jailbreak", r"developer mode",
]
PII_PATTERNS = {
    "email": r"[\w.+-]+@[\w-]+\.[\w.]+",
    "card": r"(?<!\d)(?:\d[ -]?){15}\d(?!\d)",
    "passport/ID": r"\b[A-Z][0-9]{7}\b|(?<!\d)\d{4}\s\d{4}\s\d{4}(?!\d)",   # Indian passport / Aadhaar-like
    "phone": r"(?<![\d-])(?:\+\d{1,3}[\s-]?)?(?:\d[\s-]?){9}\d(?![\d-])",  # 10+ digits, never ISO dates
}

DISCLAIMER = ("Prices, schedules and hotel listings in this prototype are simulated for "
              "demonstration. Weather comes from Open-Meteo when reachable, otherwise from "
              "climate averages. Always verify visas, prices and opening hours with official "
              "sources before booking.")


def screen_text(text: str) -> tuple[str, list[str]]:
    """Return (safe_text, warnings). Redacts PII and neutralises injection attempts."""
    warnings = []
    text = (text or "")[:MAX_TEXT]
    for label, pat in PII_PATTERNS.items():
        if re.search(pat, text):
            text = re.sub(pat, f"[{label} removed]", text)
            warnings.append(f"Removed {label} from the request (not needed for planning).")
    for pat in INJECTION_PATTERNS:
        if re.search(pat, text, re.I):
            text = re.sub(pat, "[removed]", text, flags=re.I)
            warnings.append("Ignored an instruction that tried to override the system rules.")
    return text, warnings


def validate_spec(spec: dict, today: date | None = None) -> list[str]:
    """Return a list of blocking errors (empty list = valid)."""
    today = today or date.today()
    errors = []
    o, d = spec.get("origin"), spec.get("destination")
    if o not in CITIES or not CITIES[o]["origin"]:
        errors.append("Please choose a supported origin city.")
    if d not in CITIES or not CITIES[d]["dest"]:
        errors.append("Please choose a supported destination.")
    if o and o == d:
        errors.append("Origin and destination must be different.")
    try:
        start = spec["start_date"]
        start = start if isinstance(start, date) else datetime.strptime(str(start), "%Y-%m-%d").date()
        if start < today:
            errors.append("Start date is in the past.")
        if (start - today).days > 330:
            errors.append("Start date must be within the next 11 months.")
    except (KeyError, ValueError, TypeError):
        errors.append("Please provide a valid start date.")
    n = spec.get("nights") or 0
    if not 1 <= int(n) <= 21:
        errors.append("Trip length must be between 1 and 21 nights.")
    t = spec.get("travellers") or 0
    if not 1 <= int(t) <= 9:
        errors.append("Travellers must be between 1 and 9.")
    b = spec.get("budget_inr") or 0
    if int(b) < 3000:
        errors.append("Budget must be at least INR 3,000.")
    return errors


def grounding_check(itinerary: dict, kb_names: set[str]) -> dict:
    """Share of itinerary activities that exist in the knowledge base."""
    total, grounded, missing = 0, 0, []
    for day in itinerary.get("days", []):
        for item in day.get("items", []):
            if item.get("kind") == "free":
                continue
            total += 1
            if item.get("activity", "").strip().lower() in kb_names:
                grounded += 1
            else:
                missing.append(item.get("activity"))
    return {"total": total, "grounded": grounded,
            "score": round(grounded / total, 3) if total else 1.0, "ungrounded": missing}
