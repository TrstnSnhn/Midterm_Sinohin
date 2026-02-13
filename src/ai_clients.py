"""
ai_clients.py — Optional wrapper for online AI inference APIs.

This module provides helper functions to call an external LLM or vision
API (e.g. Azure OpenAI) for richer word definitions and image descriptions.
By default the chatbot uses the **offline track** (NLTK WordNet + ResNet-50),
so this file is only activated when the environment variable
``USE_ONLINE_AI`` is set to ``"true"``.

API keys are read exclusively from environment variables — they are
**never** hard-coded.

Author : Sinohin, Angelo Tristan D.
Course : 6INTELSY – Intelligent Systems, Midterm Exam
"""

import base64
import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
USE_ONLINE_AI: bool = os.getenv("USE_ONLINE_AI", "false").lower() == "true"
API_KEY: Optional[str] = os.getenv("AZURE_OPENAI_API_KEY")
API_ENDPOINT: Optional[str] = os.getenv("AZURE_OPENAI_ENDPOINT")
API_MODEL: str = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o-mini")

# Prompt files (loaded lazily)
_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def _load_prompt(filename: str) -> str:
    """Read a prompt template from the prompts/ directory."""
    path = os.path.join(_PROMPTS_DIR, filename)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read().strip()
    except FileNotFoundError:
        logger.warning("Prompt file not found: %s", path)
        return ""


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def is_online_mode() -> bool:
    """Return True if the online AI track is configured and available."""
    return USE_ONLINE_AI and bool(API_KEY) and bool(API_ENDPOINT)


def define_word_online(word: str) -> Optional[Dict[str, Any]]:
    """Call an LLM to produce a rich dictionary entry for *word*.

    Returns ``None`` if the online mode is not configured so the caller
    can fall back to the offline track.

    The prompt used is loaded from ``prompts/word_prompts.txt``.
    """
    if not is_online_mode():
        return None

    system_prompt = _load_prompt("word_prompts.txt")
    if not system_prompt:
        logger.error("Word prompt file is empty — falling back to offline.")
        return None

    try:
        import requests  # Only needed when online mode is active.

        headers = {
            "Content-Type": "application/json",
            "api-key": API_KEY,
        }
        payload = {
            "model": API_MODEL,
            "temperature": 0.2,
            "max_tokens": 512,
            "top_p": 1.0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": word},
            ],
        }
        response = requests.post(
            f"{API_ENDPOINT}/openai/deployments/{API_MODEL}/chat/completions"
            "?api-version=2024-02-01",
            headers=headers,
            json=payload,
            timeout=10,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return json.loads(content)

    except Exception as exc:
        logger.exception("Online word lookup failed: %s", exc)
        return None


def describe_image_online(image_path: str) -> Optional[Dict[str, Any]]:
    """Call a vision model to describe the image at *image_path*.

    Returns ``None`` if the online mode is not configured so the caller
    can fall back to the offline track.

    The prompt used is loaded from ``prompts/image_prompts.txt``.
    """
    if not is_online_mode():
        return None

    system_prompt = _load_prompt("image_prompts.txt")
    if not system_prompt:
        logger.error("Image prompt file is empty — falling back to offline.")
        return None

    try:
        import requests

        with open(image_path, "rb") as img_file:
            b64_image = base64.b64encode(img_file.read()).decode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "api-key": API_KEY,
        }
        payload = {
            "model": API_MODEL,
            "temperature": 0.2,
            "max_tokens": 512,
            "top_p": 1.0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_image}"
                            },
                        },
                        {
                            "type": "text",
                            "text": "Describe this image.",
                        },
                    ],
                },
            ],
        }
        response = requests.post(
            f"{API_ENDPOINT}/openai/deployments/{API_MODEL}/chat/completions"
            "?api-version=2024-02-01",
            headers=headers,
            json=payload,
            timeout=15,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return json.loads(content)

    except Exception as exc:
        logger.exception("Online image description failed: %s", exc)
        return None