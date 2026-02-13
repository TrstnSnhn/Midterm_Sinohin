"""
image_bot.py — Image-dictionary module (offline track).

Uses a pre-trained ResNet-50 model (ImageNet weights) to classify an
uploaded image and return a structured dictionary-style entry consisting
of a *label*, a short *description*, and a contextual *meaning*.

For a richer experience an online vision API can be used instead (see
``ai_clients.py``).  The offline approach is the default so the chatbot
works without network access.

Author : Sinohin
Course : 6INTELSY – Intelligent Systems, Midterm Exam
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch
from PIL import Image
from torchvision import models

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level cache for the model and label list (loaded once on first use)
# ---------------------------------------------------------------------------
_model: Optional[torch.nn.Module] = None
_labels: Optional[list] = None
_preprocess = None


def _load_model() -> None:
    """Lazily load ResNet-50 with default ImageNet weights.

    The model and its associated class labels are cached at module level
    so subsequent calls avoid redundant I/O.
    """
    global _model, _labels, _preprocess

    if _model is not None:
        return  # Already loaded.

    logger.info("Loading ResNet-50 (ImageNet weights) …")
    weights = models.ResNet50_Weights.DEFAULT
    _model = models.resnet50(weights=weights)
    _model.eval()

    _labels = weights.meta["categories"]
    _preprocess = weights.transforms()
    logger.info("Model loaded — %d classes available.", len(_labels))


# ---------------------------------------------------------------------------
# Label → description / meaning templates
# ---------------------------------------------------------------------------
# A small hand-curated mapping for common ImageNet classes to provide
# richer descriptions.  For labels not in the map a generic template is
# used.

_LABEL_INFO: Dict[str, Tuple[str, str]] = {
    "tabby": (
        "A tabby cat with distinctive striped or spotted fur markings.",
        "A domestic cat pattern; one of the most common coat types in household cats.",
    ),
    "tiger cat": (
        "A tiger cat (mackerel tabby) with bold striped markings.",
        "A domestic cat breed known for tiger-like stripes on its coat.",
    ),
    "Siamese cat": (
        "A Siamese cat with a light body and dark points on the ears, face, and paws.",
        "A domestic cat breed known for its distinct coloration and vocal behavior.",
    ),
    "Egyptian cat": (
        "An Egyptian Mau cat with a spotted coat and elegant build.",
        "One of the oldest domesticated cat breeds, originating from Egypt.",
    ),
    "Persian cat": (
        "A Persian cat with long, luxurious fur and a flat face.",
        "A popular domestic breed known for its gentle temperament and long coat.",
    ),
    "golden retriever": (
        "A golden retriever with a friendly expression and golden coat.",
        "A popular dog breed valued for its intelligence and gentle nature.",
    ),
    "Labrador retriever": (
        "A Labrador retriever with a short, dense coat.",
        "One of the most popular dog breeds, often used as service and guide dogs.",
    ),
    "German shepherd": (
        "A German shepherd dog with an alert posture.",
        "A working dog breed known for its intelligence, loyalty, and versatility.",
    ),
    "cup": (
        "A cup — a small, open container used for drinking.",
        "A common household item for holding beverages, typically with a handle.",
    ),
    "coffee mug": (
        "A coffee mug — a sturdy cup designed for hot beverages.",
        "A household drinkware item, larger than a teacup, often ceramic.",
    ),
    "wooden spoon": (
        "A wooden spoon used for cooking and stirring.",
        "A kitchen utensil carved from wood, commonly used when cooking.",
    ),
    "chair": (
        "A chair — a piece of furniture for sitting.",
        "A common furniture item with a seat, back, and often four legs.",
    ),
    "folding chair": (
        "A folding chair — a portable, collapsible seating option.",
        "A lightweight chair designed to fold flat for easy storage and transport.",
    ),
    "rocking chair": (
        "A rocking chair with curved legs that allows back-and-forth motion.",
        "A type of chair often associated with relaxation and front porches.",
    ),
    "desk": (
        "A desk — a flat-surfaced table used for reading, writing, or working.",
        "A furniture piece commonly found in offices and study areas.",
    ),
    "laptop": (
        "A laptop computer — a portable personal computer.",
        "An electronic device used for computing, communication, and entertainment.",
    ),
    "cellular telephone": (
        "A mobile phone — a wireless communication device.",
        "A widely-used electronic device for calls, messaging, and internet access.",
    ),
    "monitor": (
        "A computer monitor — an electronic display screen.",
        "An output device used to visually display information from a computer.",
    ),
}


def _get_label_info(label: str) -> Tuple[str, str]:
    """Return (description, meaning) for a given ImageNet label.

    Falls back to a generic template when the label is not in the
    hand-curated mapping.
    """
    if label in _LABEL_INFO:
        return _LABEL_INFO[label]

    # Generic template
    description = f"An image likely depicting: {label}."
    meaning = (
        f"'{label.title()}' is a concept or object recognized by the image "
        "classifier. Consult a reference for more context."
    )
    return description, meaning


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def describe_image(image_path: str) -> str:
    """Classify the image at *image_path* and return a formatted entry.

    Parameters
    ----------
    image_path : str
        Path to a JPG or PNG image file.

    Returns
    -------
    str
        A structured plain-text string with ``[label]``, ``[description]``,
        and ``[meaning]`` sections.  If the file cannot be read, an error
        entry is returned instead.
    """
    # --- Validate path ---------------------------------------------------
    if not os.path.isfile(image_path):
        logger.error("File not found: %s", image_path)
        return _format_error(f"File not found: {image_path}")

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    _, ext = os.path.splitext(image_path)
    if ext.lower() not in valid_extensions:
        logger.error("Unsupported file type: %s", ext)
        return _format_error(
            f"Unsupported image format '{ext}'. Please use JPG or PNG."
        )

    # --- Load model & image ----------------------------------------------
    try:
        _load_model()
        img = Image.open(image_path).convert("RGB")
    except Exception as exc:
        logger.exception("Failed to open image: %s", image_path)
        return _format_error(f"Could not read image: {exc}")

    # --- Inference -------------------------------------------------------
    try:
        inp = _preprocess(img).unsqueeze(0)
        with torch.no_grad():
            logits = _model(inp)
            probs = logits.softmax(dim=1)
            confidence = probs.max().item()
            pred_idx = probs.argmax(dim=1).item()

        label = (
            _labels[pred_idx]
            if _labels and 0 <= pred_idx < len(_labels)
            else "unknown object"
        )

        # If confidence is very low, warn the user.
        if confidence < 0.15:
            label = "unknown object"
            description = (
                "The classifier could not identify this image with high confidence."
            )
            meaning = "Try uploading a clearer photo or a different angle."
        else:
            description, meaning = _get_label_info(label)

        logger.info(
            "Prediction: %s (confidence=%.2f%%)", label, confidence * 100
        )

    except Exception as exc:
        logger.exception("Inference failed for %s", image_path)
        return _format_error(f"Classification error: {exc}")

    return _format_entry(label, description, meaning, confidence)


def describe_image_json(image_path: str) -> Dict[str, Any]:
    """Return the image description as a Python dict (JSON-serialisable)."""
    # Re-use the text path and parse — keeps logic DRY for the exam.
    text = describe_image(image_path)
    result: Dict[str, Any] = {}
    current_key = None
    for line in text.splitlines():
        if line.startswith("[") and line.endswith("]"):
            current_key = line.strip("[]")
            result[current_key] = ""
        elif current_key:
            result[current_key] = (result[current_key] + " " + line.strip()).strip()
    return result


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_entry(
    label: str, description: str, meaning: str, confidence: float
) -> str:
    """Build a neatly formatted plain-text image dictionary entry."""
    return (
        f"[label]\n  {label}\n"
        f"[description]\n  {description}\n"
        f"[meaning]\n  {meaning}\n"
        f"[confidence]\n  {confidence:.2%}"
    )


def _format_error(message: str) -> str:
    """Return an error entry when the image cannot be processed."""
    return (
        f"[label]\n  unknown\n"
        f"[description]\n  {message}\n"
        f"[meaning]\n  N/A"
    )