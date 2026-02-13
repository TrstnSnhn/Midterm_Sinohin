# Multimodal Dictionary Chatbot (Words & Pictures → Meanings)

**Author:** Sinohin  
**Course:** 6INTELSY — Intelligent Systems  
**Exam:** Midterm Examination, 2nd Semester, AY 2025-2026  
**University:** Holy Angel University — School of Computing

---

## Overview

A Python chatbot that functions as a dictionary for both **words** and **pictures**:

- **Word mode:** Type any English word and receive its definition, part of speech, pronunciation, example sentences, and synonyms.
- **Image mode:** Provide a path to an image file (JPG/PNG) and receive a label, short description, and contextual meaning.

---

## How to Run

### Prerequisites

- Python 3.10+
- Install dependencies:

```bash
pip install -r requirements.txt
```

- NLTK data (auto-downloads on first run, or manually):

```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### CLI Mode (Default)

```bash
python -m src.main
```

Once running, use the following commands:

| Command              | Description                        |
|----------------------|------------------------------------|
| `define <word>`      | Look up a word definition          |
| `describe <image>`   | Describe an image file (JPG/PNG)   |
| `help`               | Show available commands            |
| `exit`               | Quit the chatbot                   |

**Example session:**

```
> define serendipity
[word]
  serendipity
[part_of_speech]
  noun
[pronunciation]
  N/A
[definition]
  the occurrence and development of events by chance in a happy or beneficial way
[examples]
  - a fortunate stroke of serendipity
[synonyms]
  fortune, luck

> describe tests/assets/cat.jpg
[label]
  tabby
[description]
  A tabby cat with distinctive striped or spotted fur markings.
[meaning]
  A domestic cat pattern; one of the most common coat types in household cats.
[confidence]
  87.34%
```

### Web UI Mode (Optional — Streamlit)

```bash
pip install streamlit
streamlit run streamlit_app.py
```

This opens a browser-based interface with a text box for words and a file uploader for images.

### Running on Google Colab

1. Upload the `src/` folder to Colab.
2. Run:
   ```python
   !pip install nltk torch torchvision pillow
   import nltk
   nltk.download('wordnet')
   nltk.download('omw-1.4')
   ```
3. Test:
   ```python
   from src.word_bot import define_word
   print(define_word("cat"))
   ```

---

## Offline vs Online Configuration

### Offline Track (Default)

| Component   | Tool                          | Notes                                      |
|-------------|-------------------------------|---------------------------------------------|
| Words       | NLTK WordNet                  | Definitions, POS, examples, synonyms        |
| Images      | torchvision ResNet-50         | ImageNet pre-trained, 1000-class classifier |
| Pronunciation | N/A                        | Not available offline via WordNet            |

No API keys or internet access required.

### Online Track (Optional)

Set the following environment variables to enable AI-powered responses:

```bash
export USE_ONLINE_AI=true
export AZURE_OPENAI_API_KEY=your-key-here
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
export AZURE_OPENAI_MODEL=gpt-4o-mini
```

On Windows (PowerShell):

```powershell
$env:USE_ONLINE_AI = "true"
$env:AZURE_OPENAI_API_KEY = "your-key-here"
$env:AZURE_OPENAI_ENDPOINT = "https://your-resource.openai.azure.com"
$env:AZURE_OPENAI_MODEL = "gpt-4o-mini"
```

When enabled, `ai_clients.py` routes word/image queries through the Azure OpenAI API using prompts stored in `src/prompts/`. If the API call fails, the system falls back to the offline track automatically.

---

## Model and Dataset Notes

### Word Dictionary

- **Data source:** NLTK WordNet 3.0 (via `nltk.corpus.wordnet`)
- **Coverage:** ~155,000 English words organized into synonym sets (synsets)
- **Sense selection:** The first (most frequent) synset is used as the primary sense
- **Synonyms:** Collected from lemma names across the top synsets, excluding the query word

### Image Classifier

- **Model:** ResNet-50 (torchvision `ResNet50_Weights.DEFAULT`)
- **Dataset:** Pre-trained on ImageNet (ILSVRC 2012) — 1,000 object classes
- **Inference:** Single forward pass, softmax confidence scoring
- **Label enrichment:** A hand-curated dictionary maps common ImageNet labels to richer descriptions and meanings
- **Low-confidence handling:** If confidence < 15%, the label defaults to "unknown object"

---

## Project Structure

```
MIDTERM_SINOHIN/
├── src/
│   ├── __init__.py            # Package init
│   ├── main.py                # CLI entry point
│   ├── word_bot.py            # Word dictionary (NLTK WordNet)
│   ├── image_bot.py           # Image classifier (ResNet-50)
│   ├── ai_clients.py          # Optional online AI wrappers
│   ├── safety.py              # Content filter / safety refusals
│   └── prompts/
│       ├── word_prompts.txt   # LLM prompt for word lookups
│       └── image_prompts.txt  # LLM prompt for image descriptions
├── test/
│   ├── test_words.py          # Unit tests for word lookups
│   ├── test_images.py         # Unit tests for image classification
│   └── assets/                # Test images (cat.jpg, mug.jpg, etc.)
├── requirements.txt           # Python dependencies
├── Readme.md                  # This file
└── prompt_justification.md    # Prompt design rationale (graded)
```

---

## AI Tools Disclosure

- **Claude (Anthropic):** Used to assist with code structure, documentation, and prompt design. All code was reviewed, understood, and adapted by the student.
- **NLTK WordNet:** Offline dictionary corpus — no API calls.
- **torchvision ResNet-50:** Pre-trained model weights downloaded from PyTorch Hub — no API calls during inference.
- **Azure OpenAI (optional):** Only activated if environment variables are set. Prompts are stored in `src/prompts/` and documented in `prompt_justification.md`.

---

## Running Tests

```bash
python -m pytest test/ -v
```

Or individually:

```bash
python -m pytest test/test_words.py -v
python -m pytest test/test_images.py -v
```
