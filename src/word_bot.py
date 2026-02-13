"""
word_bot.py — Word-dictionary module (offline track).

Uses NLTK WordNet to look up definitions, parts of speech, example
sentences, and synonyms for a given English word.  Pronunciation is
not available through WordNet, so the field is returned as "N/A".

If an online AI client is configured (see ``ai_clients.py``), the module
can optionally delegate to an LLM for richer, disambiguated results.

Author : Sinohin
Course : 6INTELSY – Intelligent Systems, Midterm Exam
"""

import json
import logging
from typing import Any, Dict, List, Optional

from nltk.corpus import wordnet as wn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Map WordNet single-char POS tags to human-readable strings.
_POS_MAP = {
    "n": "noun",
    "v": "verb",
    "a": "adjective",
    "r": "adverb",
    "s": "adjective satellite",
}


def _pos_readable(wn_pos: str) -> str:
    """Convert a WordNet POS character to a readable label."""
    return _POS_MAP.get(wn_pos, "unknown")


def _collect_synonyms(synset, word: str, limit: int = 3) -> List[str]:
    """Gather unique synonyms from a synset, excluding the query word."""
    synonyms = sorted(
        {
            lemma.name().replace("_", " ")
            for lemma in synset.lemmas()
            if lemma.name().lower() != word.lower()
        }
    )
    return synonyms[:limit]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def define_word(word: str) -> str:
    """Look up *word* in WordNet and return a formatted dictionary entry.

    Parameters
    ----------
    word : str
        A single English word (e.g. "serendipity", "cat").

    Returns
    -------
    str
        A human-readable, structured string containing the word's
        definition, part of speech, pronunciation, examples, and synonyms.
        If the word is not found, a helpful fallback message is returned.
    """
    word = word.strip().lower()

    if not word:
        return _format_not_found(word)

    synsets = wn.synsets(word)

    if not synsets:
        logger.info("No synsets found for '%s'.", word)
        return _format_not_found(word)

    # Use the first (most frequent / common) synset as the primary sense.
    primary = synsets[0]

    pos = _pos_readable(primary.pos())
    definition = primary.definition()

    # Collect up to 2 examples; fall back to other synsets if needed.
    examples = primary.examples()[:2]
    if not examples:
        for ss in synsets[1:4]:
            examples = ss.examples()[:2]
            if examples:
                break

    # Collect synonyms across the first few synsets for variety.
    synonyms: List[str] = _collect_synonyms(primary, word)
    if len(synonyms) < 2:
        for ss in synsets[1:4]:
            synonyms += _collect_synonyms(ss, word)
        # De-duplicate while preserving order
        seen = set()
        unique: List[str] = []
        for s in synonyms:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        synonyms = unique[:3]

    # Pronunciation is not available in WordNet — mark as N/A.
    pronunciation = "N/A"

    return _format_entry(word, pos, pronunciation, definition, examples, synonyms)


def define_word_json(word: str) -> Dict[str, Any]:
    """Return the dictionary entry as a Python dict (JSON-serialisable).

    Useful for the optional Web UI or automated tests that prefer
    structured data over formatted text.
    """
    word = word.strip().lower()
    synsets = wn.synsets(word)

    if not synsets:
        return {
            "word": word,
            "part_of_speech": "unknown",
            "pronunciation": "N/A",
            "definition": "No entry found. Check spelling or try a simpler term.",
            "examples": [],
            "synonyms": [],
        }

    primary = synsets[0]
    examples = primary.examples()[:2]
    if not examples:
        for ss in synsets[1:4]:
            examples = ss.examples()[:2]
            if examples:
                break

    synonyms = _collect_synonyms(primary, word)
    if len(synonyms) < 2:
        for ss in synsets[1:4]:
            synonyms += _collect_synonyms(ss, word)
        seen = set()
        unique: List[str] = []
        for s in synonyms:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        synonyms = unique[:3]

    return {
        "word": word,
        "part_of_speech": _pos_readable(primary.pos()),
        "pronunciation": "N/A",
        "definition": primary.definition(),
        "examples": examples,
        "synonyms": synonyms,
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_entry(
    word: str,
    pos: str,
    pronunciation: str,
    definition: str,
    examples: List[str],
    synonyms: List[str],
) -> str:
    """Build a neatly formatted plain-text dictionary entry."""
    examples_text = (
        "\n".join(f"  - {ex}" for ex in examples)
        if examples
        else "  - (no example available)"
    )
    synonyms_text = ", ".join(synonyms) if synonyms else "(none)"

    return (
        f"[word]\n  {word}\n"
        f"[part_of_speech]\n  {pos}\n"
        f"[pronunciation]\n  {pronunciation}\n"
        f"[definition]\n  {definition}\n"
        f"[examples]\n{examples_text}\n"
        f"[synonyms]\n  {synonyms_text}"
    )


def _format_not_found(word: str) -> str:
    """Return a friendly 'not found' message."""
    return (
        f"[word]\n  {word}\n"
        f"[part_of_speech]\n  unknown\n"
        f"[pronunciation]\n  N/A\n"
        f"[definition]\n  No entry found. Check spelling or try a simpler term.\n"
        f"[examples]\n  - (none)\n"
        f"[synonyms]\n  (none)"
    )