# PROMPT_JUSTIFICATION.md

**Author:** Sinohin, Angelo Tristan D.
**Course:** 6INTELSY — Intelligent Systems, Midterm Examination  
**University:** Holy Angel University — School of Computing

---

## 1. Objective & Constraints

### What the Model Must Do

The chatbot serves as a **multimodal dictionary** with two core tasks:

1. **Word Lookup:** Given a single English word, return a structured entry containing: definition, part of speech, pronunciation, 1–2 example sentences, and 2–3 synonyms.
2. **Image Description:** Given an uploaded image (JPG/PNG), return a structured entry containing: a primary label, a 1–2 sentence description, and a short contextual meaning.

### Input / Output Formats

| Task   | Input                  | Output Format                                                        |
|--------|------------------------|----------------------------------------------------------------------|
| Word   | Single English word    | JSON with keys: `word`, `part_of_speech`, `pronunciation`, `definition`, `examples`, `synonyms` |
| Image  | JPG/PNG file path      | JSON with keys: `label`, `description`, `meaning`                    |

### Safety Requirements

- The model must **refuse** unsafe, explicit, violent, or irrelevant content.
- If the word is unknown or misspelled, the model should respond gracefully with a helpful fallback.
- If the image cannot be classified, the model should return "unknown object" rather than hallucinate.

---

## 2. Prompt Design

### Word Prompt (`src/prompts/word_prompts.txt`)

#### Role

> "You are a precise, neutral dictionary assistant."

**Rationale:** The role establishes the model as a factual reference tool, not a creative or conversational agent. The word "precise" discourages verbose or speculative answers. "Neutral" prevents opinionated or biased definitions.

#### Instructions

The prompt specifies exactly which fields to return:
- `word` — the queried word (lowercase)
- `part_of_speech` — noun, verb, adjective, etc.
- `pronunciation` — IPA transcription if available
- `definition` — clear and accurate, 1–2 sentences
- `examples` — 1–2 natural example sentences
- `synonyms` — 2–3 synonyms

**Rationale:** Enumerating the exact fields ensures the model produces complete, structured output every time. Without explicit field listing, models tend to omit fields inconsistently.

#### Constraints

1. Prefer the most common sense of the word.
2. Keep responses factual — no subjective opinions.
3. Handle unknown/misspelled words gracefully with closest matches.
4. Refuse unsafe or irrelevant queries with a specific error JSON.
5. Output **only** valid JSON.

**Rationale:** Constraint #1 prevents the model from choosing obscure senses. Constraint #5 (strict JSON) makes parsing reliable and prevents prose leaking into structured output.

#### Style & Tone

Concise, neutral, plain English. Definitions should be accessible to a general audience without being oversimplified.

#### Output Schema

```json
{
  "word": "string",
  "part_of_speech": "string",
  "pronunciation": "string",
  "definition": "string",
  "examples": ["string"],
  "synonyms": ["string"]
}
```

#### Few-shot Example

```
Input: "serendipity"
Output:
{
  "word": "serendipity",
  "part_of_speech": "noun",
  "pronunciation": "/ˌsɛr.ənˈdɪp.ɪ.ti/",
  "definition": "The occurrence of events by chance in a happy or beneficial way.",
  "examples": [
    "Finding that rare book at the garage sale was pure serendipity.",
    "Their meeting was a wonderful case of serendipity."
  ],
  "synonyms": ["luck", "fortune", "chance"]
}
```

**Rationale:** Including a concrete example anchors the model's expected output format and quality level, reducing formatting variance across queries.

---

### Image Prompt (`src/prompts/image_prompts.txt`)

#### Role

> "You are a vision description assistant."

**Rationale:** A narrower role than "general assistant" — it signals to the model that it should focus on visual analysis, not conversation.

#### Instructions

1. Identify the single most probable label (primary concept).
2. Produce a 1–2 sentence description.
3. Produce a short "meaning" explaining category, function, or significance.

**Rationale:** Separating "description" (what it looks like) from "meaning" (what it is / why it matters) ensures the model provides both perceptual and semantic information.

#### Constraints

1. Be precise and factual — no speculation.
2. If uncertain, use "unknown object" as the label.
3. Refuse unsafe image content with an error JSON.
4. Output only valid JSON.

**Rationale:** Constraint #2 is critical — it prevents hallucinated labels when the model is unsure, which is a common failure mode in vision models.

#### Output Schema

```json
{
  "label": "string",
  "description": "string",
  "meaning": "string"
}
```

#### Few-shot Example

```
Output:
{
  "label": "Siamese cat",
  "description": "A Siamese cat with a light-colored body and dark points on its ears, face, and paws.",
  "meaning": "A domestic cat breed originating from Thailand, known for its distinctive coloration and vocal personality."
}
```

---

## 3. Parameters & Tools

### Model Configuration (Online Track)

| Parameter      | Value        | Rationale                                                                |
|----------------|--------------|--------------------------------------------------------------------------|
| Model          | gpt-4o-mini  | Balances capability with speed and cost for a dictionary use case        |
| Temperature    | 0.2          | Low temperature for **consistency** — dictionary entries should not vary significantly between calls |
| Max Tokens     | 512          | Sufficient for the structured JSON output; prevents runaway responses    |
| Top-p          | 1.0          | Combined with low temperature, allows the model full vocabulary access while still being deterministic |

**Why temperature = 0.2?**  
Dictionary definitions are factual — there is little benefit to high creativity. A low temperature ensures that repeated queries for the same word yield near-identical results, which is the expected behavior of a dictionary.

**Why max_tokens = 512?**  
The JSON output for a word entry is typically 150–250 tokens. Setting 512 provides headroom for longer definitions or multiple examples without allowing the model to produce pages of unnecessary text.

### Offline Track Tools

| Component       | Tool                        | Rationale                                                    |
|-----------------|-----------------------------|--------------------------------------------------------------|
| Word lookup     | NLTK WordNet                | Well-established lexical database; no API needed             |
| Image classify  | torchvision ResNet-50       | Pre-trained on ImageNet; fast inference; no API needed        |
| Safety filter   | Keyword-based (regex)       | Lightweight, fast, no dependencies; whole-word matching avoids false positives |

---

## 4. Iteration Log

### Iteration 1 → Iteration 2 (Word Prompt)

**Problem:** Initial prompt did not specify JSON-only output. The model sometimes returned a mix of prose and JSON, making parsing unreliable.

**Change:** Added explicit constraint: *"Output ONLY valid JSON with the keys: word, part_of_speech, pronunciation, definition, examples, synonyms."*

**Result:** Model consistently returned parseable JSON after this change.

---

### Iteration 2 → Iteration 3 (Word Prompt)

**Problem:** When given misspelled words like "serendipitee", the model hallucinated a definition instead of flagging the error.

**Change:** Added constraint: *"If the word is unknown, misspelled, or ambiguous, respond with the closest match and a brief note."*

**Result:** Model now suggests "serendipity" as the closest match instead of inventing a definition for the misspelled input.

---

### Iteration 1 → Iteration 2 (Image Prompt)

**Problem:** For ambiguous or blurry images, the model would confidently assign a label even when uncertain (hallucination).

**Change:** Added constraint: *"If uncertain, say 'unknown object'."* Also added a confidence threshold (< 15%) in the offline code that forces the label to "unknown object".

**Result:** Reduced false-positive labels for unclear images.

---

### Iteration 2 → Iteration 3 (Safety)

**Problem:** The initial keyword blocklist used simple substring matching, which caused false positives (e.g., "therapist" was blocked because it contains "rapist").

**Change:** Switched to regex whole-word boundary matching (`\b` anchors) in `safety.py`.

**Result:** Legitimate words like "therapist", "grape", and "skilled" are no longer falsely blocked.

---

## 5. Risk & Mitigation

| Risk                        | Mitigation                                                                  |
|-----------------------------|-----------------------------------------------------------------------------|
| **Ambiguous words**         | WordNet selects the most frequent synset (sense #1). The online prompt instructs the model to prefer the most common sense. |
| **Unknown / misspelled words** | Offline: returns "No entry found" with guidance. Online: prompt instructs model to suggest closest match. |
| **Unsafe requests**         | `safety.py` blocks queries containing dangerous keywords using whole-word regex matching. The online prompt includes a refusal instruction. |
| **Image hallucination**     | Offline: confidence threshold (< 15%) → "unknown object". Online: prompt explicitly says "If uncertain, say unknown object." |
| **Unreadable images**       | `image_bot.py` validates file existence and extension before inference. PIL errors are caught and return a descriptive error message. |
| **API failures (online)**   | `ai_clients.py` wraps all API calls in try/except. On failure, the system falls back to the offline track automatically. |
| **Model inconsistency**     | Low temperature (0.2) and explicit JSON schema reduce output variance. |

---

## 6. Final Prompts (Exact Text)

### Word Prompt (`src/prompts/word_prompts.txt`)

```
You are a precise, neutral dictionary assistant. When given a single English word, return a concise entry with the following fields:

- word: the queried word (lowercase)
- part_of_speech: noun, verb, adjective, adverb, etc.
- pronunciation: IPA phonetic transcription if available, otherwise "N/A"
- definition: a clear, accurate definition in 1–2 sentences
- examples: 1–2 simple, natural example sentences demonstrating usage
- synonyms: 2–3 synonyms (if available)

Rules:
1. Prefer the most common sense of the word.
2. Keep responses factual — avoid subjective opinions or commentary.
3. If the word is unknown, misspelled, or ambiguous, respond with the closest match and a brief note.
4. Do NOT respond to unsafe, explicit, or irrelevant queries — reply with: {"error": "Unsafe or irrelevant request."}
5. Output ONLY valid JSON with the keys: word, part_of_speech, pronunciation, definition, examples, synonyms.

Example input: "serendipity"
Example output:
{
  "word": "serendipity",
  "part_of_speech": "noun",
  "pronunciation": "/ˌsɛr.ənˈdɪp.ɪ.ti/",
  "definition": "The occurrence of events by chance in a happy or beneficial way.",
  "examples": [
    "Finding that rare book at the garage sale was pure serendipity.",
    "Their meeting was a wonderful case of serendipity."
  ],
  "synonyms": ["luck", "fortune", "chance"]
}
```

### Image Prompt (`src/prompts/image_prompts.txt`)

```
You are a vision description assistant. Given an image, your task is to identify and describe the primary subject.

Return a structured response with these fields:
- label: the single most probable label (primary concept/object in the image)
- description: a concise 1–2 sentence description of what the image shows
- meaning: a short explanation of the object's category, common use, function, or cultural significance

Rules:
1. Be precise and factual — avoid speculation or creative embellishment.
2. If uncertain about the subject, use "unknown object" as the label.
3. Do NOT describe or respond to images containing explicit, violent, or unsafe content — reply with: {"error": "Unsafe content detected."}
4. Output ONLY valid JSON with the keys: label, description, meaning.

Example output:
{
  "label": "Siamese cat",
  "description": "A Siamese cat with a light-colored body and dark points on its ears, face, and paws.",
  "meaning": "A domestic cat breed originating from Thailand, known for its distinctive coloration and vocal personality."
}
```
