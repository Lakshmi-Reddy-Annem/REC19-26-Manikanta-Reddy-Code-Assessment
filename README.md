# Defence RAG — Non-Embedding Document Q&A

A document Q&A system over Indian Naval/Defence regulation PDFs that deliberately **avoids embedding-based vector search** while keeping per-query cost comparable to standard RAG.

---

## Approach

### Why not embeddings?

Standard RAG embeds every chunk at index time and embeds every query at inference time. This repo replaces both steps with **BM25 (Best Match 25)** — a classic TF-IDF based probabilistic retrieval algorithm. BM25:

- Costs **zero tokens** per query for retrieval (no embedding call)
- Is deterministic and interpretable (no black-box vector space)
- Works particularly well on regulatory/statutory text because such documents use precise, consistent terminology ("31 gun salute" is never paraphrased as "large cannon honor")

### Architecture

```
[PDF corpus]
     │
     ▼  build_index.py  (run once)
[Text extraction via pdfplumber]
     │
     ▼
[Sliding-window chunker]
 400 words/chunk, 80-word overlap
 Metadata: doc name, chapter, page range
     │
     ▼
[BM25Okapi index]  ←── saved to index/bm25_index.pkl
[Chunk store]      ←── saved to index/chunks.json
```

```
[User question]
     │
     ▼  ask.py  (per query)
[BM25 retrieval]  — top-5 chunks, zero embedding cost
     │
     ▼
[Context assembly]  — formatted excerpts with source metadata
     │
     ▼
[Gemini 2.0 Flash]  — generate answer + cite sources
     │
     ▼
[Answer + citations + token cost report]
```

### Cost comparison vs RAG

BM25 system numbers are **measured** from actual Gemini API `usage_metadata` (prompt_token_count + candidates_token_count). RAG baseline numbers are **theoretical estimates** based on the same chunk size and top-k assumptions — implementing a full RAG pipeline to measure it exactly is out of scope here, but the LLM input cost is structurally identical since both systems send the same 5×400-word chunks to the model.

| Step | RAG (estimated) | This system (measured) |
|---|---|---|
| Index time embedding | O(n_chunks × chunk_tokens) — one-time | **0** |
| Per-query embedding | ~50 tokens | **0** |
| Per-query LLM input | ~2200 tokens | ~2200 tokens (actual) |
| Per-query LLM output | ~150 tokens | ~150 tokens (actual) |
| **Total per query** | **~2400 tokens** | **~2350 tokens** |

The LLM generation cost is structurally identical — both systems send the same top-5 chunks as context. The only difference is that RAG also pays an embedding call per query, which this system eliminates entirely. `evaluate.py` prints the actual measured per-query token counts vs this baseline.

### Handling different question types

| Question type | How handled |
|---|---|
| Single-doc fact lookup | BM25 pinpoints the right chunk |
| Needle-in-haystack (specific number, date, name) | BM25 keyword match is precise |
| Multi-doc synthesis | Top-5 chunks may span both documents; Gemini synthesises |
| Whole-corpus questions | Retrieved chunks provide representative coverage |
| Unanswerable | Gemini is prompted to say "I cannot find this information..." |

### Scaling to 100+ documents

The system scales linearly:
- BM25 index is an in-memory object — 100 PDFs (~50k chunks) fit comfortably in RAM
- Index build time is ~1-2 min/100 docs (one-time)
- Per-query latency is unchanged — BM25 scoring is O(n) but very fast in practice

---

## Setup

```bash
pip install -r requirements.txt
```

Place all PDF files in the `data/` directory.

---

## Usage

### Step 1: Build the index (run once)

```bash
python build_index.py
```

### Step 2: Ask a question

```bash
# Set your Gemini API key
set GEMINI_API_KEY=your_key_here          # Windows
export GEMINI_API_KEY=your_key_here       # Linux/Mac

# Single question
python ask.py "What is the gun salute for the President of India?"

# Interactive mode
python ask.py
```

### Step 3: Run cost evaluation

```bash
python evaluate.py
```

Runs 10 sample questions and prints a cost comparison table vs hypothetical RAG baseline. Saves full results to `evaluation_results.json`.

---

## Output format

```
Answer:
According to the regulations, the President of India is accorded a salute of 31 guns.
[Source: RegsNavyIII, Chapter II, Page 12]

Sources retrieved (BM25):
  RegsNavyIII | Chapter II | Pages 11-13  (score: 14.23)
  RegsNavyIII | Chapter II | Pages 9-11   (score: 8.71)
  ...

Token cost: 2180 in + 92 out = 2272 total  [Zero embedding cost — BM25 retrieval only]
```

---

## Design decisions

- **Chunk size 400 words / 80-word overlap** — large enough to capture a full regulation with context, small enough to keep LLM input cost bounded
- **BM25Okapi** — the standard variant with term frequency saturation; best for short queries against longer documents
- **No NLTK/spaCy dependency** — tokenization is a simple regex `\b[a-z0-9]+\b` for portability
- **Gemini 2.5 Flash** — fast, cheap, large context window; swap to any Gemini model via `GEMINI_MODEL` constant in `ask.py`

---

## File structure

```
.
├── data/               ← put your PDFs here
├── index/              ← auto-generated by build_index.py
│   ├── bm25_index.pkl
│   └── chunks.json
├── build_index.py      ← parse PDFs + build index
├── ask.py              ← query pipeline
├── evaluate.py         ← cost comparison vs RAG
├── requirements.txt
└── README.md
```
