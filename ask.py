"""
Query the document corpus: BM25 retrieval → Gemini answer generation.
No embedding-based retrieval. Cost tracks input/output tokens only.

Usage:
    # Set your API key first:
    #   Windows:  set GEMINI_API_KEY=your_key
    #   Linux/Mac: export GEMINI_API_KEY=your_key

    python ask.py "What is the gun salute for the President of India?"
    python ask.py          # interactive mode
"""

import json
import os
import pickle
import re
import sys
from pathlib import Path

from google import genai
from google.genai import types
from rank_bm25 import BM25Okapi

INDEX_DIR = Path("index")
TOP_K = 5
GEMINI_MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = (
    "You are a precise assistant for Indian Naval and Defence regulations. "
    "Answer questions based ONLY on the provided document excerpts. "
    "Rules:\n"
    "1. If the answer is in the excerpts: state it directly and cite the source as "
    "[Source: <document>, <chapter>, Page <N>].\n"
    "2. If only partial information is present: share what you found and note the gap.\n"
    "3. If the answer is NOT in the excerpts: respond with exactly "
    "\"I cannot find this information in the provided documents.\"\n"
    "Keep answers concise and factual."
)


def tokenize(text: str) -> list:
    return re.findall(r'\b[a-z0-9]+\b', text.lower())


def load_index():
    idx_path = INDEX_DIR / "bm25_index.pkl"
    chunks_path = INDEX_DIR / "chunks.json"
    if not idx_path.exists() or not chunks_path.exists():
        print("[!] Index not found. Run 'python build_index.py' first.")
        sys.exit(1)
    with open(idx_path, "rb") as f:
        bm25 = pickle.load(f)
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return bm25, chunks


def retrieve(query: str, bm25: BM25Okapi, chunks: list, top_k: int = TOP_K) -> list:
    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    results = []
    for idx in top_idx:
        c = dict(chunks[idx])
        c["score"] = round(float(scores[idx]), 4)
        results.append(c)
    return results


def build_context(retrieved: list) -> str:
    parts = []
    for i, c in enumerate(retrieved, 1):
        pages = (
            f"Page {c['start_page']}"
            if c["start_page"] == c["end_page"]
            else f"Pages {c['start_page']}-{c['end_page']}"
        )
        header = f"[{c['doc']} | {c.get('chapter', '')} | {pages}]"
        parts.append(f"--- Excerpt {i} {header} ---\n{c['text']}")
    return "\n\n".join(parts)


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def answer(query: str, bm25: BM25Okapi, chunks: list, client: genai.Client) -> dict:
    retrieved = retrieve(query, bm25, chunks)
    context = build_context(retrieved)

    prompt = (
        f"Here are relevant excerpts from Indian Naval regulations:\n\n"
        f"{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
        ),
    )
    reply = response.text.strip()

    # Use actual API token counts when available; fall back to heuristic
    usage = getattr(response, "usage_metadata", None)
    if usage and getattr(usage, "prompt_token_count", None):
        input_tok = usage.prompt_token_count
        output_tok = usage.candidates_token_count or approx_tokens(reply)
    else:
        input_tok = approx_tokens(SYSTEM_PROMPT + prompt)
        output_tok = approx_tokens(reply)

    return {
        "question": query,
        "answer": reply,
        "sources": [
            {
                "doc": c["doc"],
                "chapter": c.get("chapter", ""),
                "pages": f"{c['start_page']}-{c['end_page']}",
                "bm25_score": c["score"],
            }
            for c in retrieved
        ],
        "tokens": {
            "input": input_tok,
            "output": output_tok,
            "total": input_tok + output_tok,
            "model": GEMINI_MODEL,
            "source": "api" if (usage and getattr(usage, "prompt_token_count", None)) else "estimated",
            "note": "Zero embedding cost — BM25 retrieval only",
        },
    }


def print_result(result: dict):
    print(f"\nAnswer:\n{result['answer']}\n")
    print("Sources retrieved (BM25):")
    for s in result["sources"]:
        print(f"  {s['doc']} | {s['chapter']} | Pages {s['pages']}  (score: {s['bm25_score']})")
    t = result["tokens"]
    print(f"\nToken cost: {t['input']} in + {t['output']} out = {t['total']} total  [{t['source']}]  [{t['note']}]\n")


def main():
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("[!] GEMINI_API_KEY not set.")
        print("    Windows:   set GEMINI_API_KEY=your_key")
        print("    Linux/Mac: export GEMINI_API_KEY=your_key")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    print("Loading index...", end=" ", flush=True)
    bm25, chunks = load_index()
    print(f"{len(chunks)} chunks ready.")

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        result = answer(query, bm25, chunks, client)
        print_result(result)
    else:
        print("Interactive mode — type 'quit' to exit.\n")
        while True:
            try:
                query = input("Question: ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if query.lower() in ("quit", "exit", "q", ""):
                break
            result = answer(query, bm25, chunks, client)
            print_result(result)


if __name__ == "__main__":
    main()
