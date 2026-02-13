"""
main.py — Entry point for the Multimodal Dictionary Chatbot (CLI).

Provides a simple command-line interface with the following commands:

    define <word>       Look up a word definition
    describe <image>    Describe an image file (JPG/PNG)
    help                Show available commands
    exit                Quit the chatbot

The chatbot delegates to ``word_bot`` and ``image_bot`` for the actual
processing and uses ``safety`` to filter unsafe requests.

Author : Sinohin
Course : 6INTELSY – Intelligent Systems, Midterm Exam
"""

import logging
import os
import sys

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so `src.*` imports work when
# running as `python -m src.main` or `python src/main.py`.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.word_bot import define_word
from src.image_bot import describe_image
from src.safety import is_safe_request, get_refusal_message

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NLTK data bootstrap (download WordNet corpus if not present)
# ---------------------------------------------------------------------------

def _ensure_nltk_data() -> None:
    """Download required NLTK data files if they are missing."""
    import nltk

    for resource in ("wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            logger.info("Downloading NLTK resource: %s", resource)
            nltk.download(resource, quiet=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

HELP_TEXT = """
╔════════════════════════════════════════════════════╗
║   Multimodal Dictionary Chatbot — Commands        ║
╠════════════════════════════════════════════════════╣
║  define <word>      Look up a word definition     ║
║  describe <image>   Describe an image (JPG/PNG)   ║
║  help               Show this help message        ║
║  exit               Quit the chatbot              ║
╚════════════════════════════════════════════════════╝
"""

WELCOME = """
╔════════════════════════════════════════════════════╗
║     Multimodal Dictionary Chatbot                 ║
║     Words & Pictures → Meanings                   ║
╠════════════════════════════════════════════════════╣
║  Type 'help' for a list of commands.              ║
╚════════════════════════════════════════════════════╝
"""


def main() -> None:
    """Run the interactive CLI loop."""
    _ensure_nltk_data()

    print(WELCOME)

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not line:
            continue

        # ---- Built-in commands ------------------------------------------
        if line.lower() == "exit":
            print("Bye!")
            break

        if line.lower() == "help":
            print(HELP_TEXT)
            continue

        # ---- Parse <command> <argument> ---------------------------------
        parts = line.split(maxsplit=1)

        if len(parts) < 2:
            print("Usage: define <word> | describe <image>")
            print("Type 'help' for more info.")
            continue

        cmd, arg = parts[0].lower(), parts[1]

        # ---- Safety check -----------------------------------------------
        if not is_safe_request(arg):
            print(get_refusal_message())
            continue

        # ---- Dispatch ---------------------------------------------------
        if cmd == "define":
            logger.info("Looking up word: %s", arg)
            result = define_word(arg)
            print(result)

        elif cmd == "describe":
            logger.info("Describing image: %s", arg)
            result = describe_image(arg)
            print(result)

        else:
            print(f"Unknown command: '{cmd}'. Type 'help' for options.")


# ---------------------------------------------------------------------------
# Allow running as  python src/main.py  OR  python -m src.main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()