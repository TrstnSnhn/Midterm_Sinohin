"""
safety.py — Content-filtering and safety-refusal module.

Provides a lightweight keyword-based guard that rejects queries containing
unsafe, explicit, or clearly irrelevant content.  The blocked-word list is
intentionally minimal for an exam setting; a production system would use a
classifier or moderation API.

Author : Sinohin
Course : 6INTELSY – Intelligent Systems, Midterm Exam
"""

import logging
import re
from typing import List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Blocked-keyword list (lowercase).  Extend as needed.
# ---------------------------------------------------------------------------
BLOCKED_KEYWORDS: List[str] = [
    "weapon",
    "bomb",
    "explosive",
    "explicit",
    "porn",
    "pornography",
    "hate",
    "self-harm",
    "suicide",
    "kill",
    "murder",
    "drug",
    "narcotic",
    "terrorism",
    "racist",
    "slur",
]


def is_safe_request(text: str) -> bool:
    """Return *True* if the request text passes the safety filter.

    The check is case-insensitive and uses whole-word boundary matching so
    that benign words like "therapist" or "skilled" are not accidentally
    blocked.

    Parameters
    ----------
    text : str
        The raw user input (word or file path).

    Returns
    -------
    bool
        ``True`` when the input is considered safe; ``False`` otherwise.
    """
    if not text or not text.strip():
        logger.warning("Empty input received — treated as safe (no-op).")
        return True

    lowered = text.lower()

    for keyword in BLOCKED_KEYWORDS:
        # \b ensures whole-word matching to reduce false positives
        if re.search(rf"\b{re.escape(keyword)}\b", lowered):
            logger.info("Blocked keyword detected: '%s' in input.", keyword)
            return False

    return True


def get_refusal_message() -> str:
    """Return a polite refusal string shown to the user."""
    return (
        "Sorry, I can't help with that request. "
        "Please enter a valid English word or image path."
    )