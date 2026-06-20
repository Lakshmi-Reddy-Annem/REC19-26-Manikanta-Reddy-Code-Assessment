"""
Cost comparison: BM25-based system vs hypothetical embedding-RAG baseline.

Usage:
    python evaluate.py
"""

import json
import os
import sys
import time
from pathlib import Path

from google import genai

sys.path.insert(0, str(Path(__file__).parent))
from ask import load_index, answer, GEMINI_MODEL, SYSTEM_PROMPT

EMBEDDING_TOKENS_PER_QUERY = 50
EMBEDDING_TOKENS_PER_CHUNK = 400
TOP_K = 5

SAMPLE_QUESTIONS = [
    # Single-doc fact lookup
    "What is the gun salute accorded to the President of India?",
    # Eligibility / multi-part
    "What are the eligibility requirements for enrolment in the Indian Naval Auxiliary Service?",
    # Promotion policy
    "How is promotion handled for Branch Officers?",
    # Uniform regulations
    "What uniform is prescribed for officers of the Indian Naval Auxiliary Service?",
    # Pay entitlements
    "What are the pay entitlements for sailors when mobilised?",
    # Acting rank conditions
    "Who can grant acting rank and under what conditions?",
    # Retirement age
    "What is the retirement age for officers in the Indian Naval Auxiliary Service?",
    # Compulsory retirement
    "Describe the conditions under which compulsory retirement can be imposed.",
    # Privileges
    "What privileges are granted to Indian Naval Reserve officers?",
    # Regulatory reference
    "What does Regulation 1 of Part III state about short title and commencement?",
    # Unanswerable — not in any naval regulation document
    "What is the procurement budget allocated for aircraft carrier construction in FY2024-25?",
]


def rag_cost_estimate(num_chunks: int) -> dict:
    index_cost = num_chunks * EMBEDDING_TOKENS_PER_CHUNK
    per_query_llm_input = TOP_K * EMBEDDING_TOKENS_PER_CHUNK + 200
    per_query_llm_output = 150
    return {
        "one_time_index_tokens": index_cost,
        "per_query_embed_tokens": EMBEDDING_TOKENS_PER_QUERY,
        "per_query_llm_input_tokens": per_query_llm_input,
        "per_query_llm_output_tokens": per_query_llm_output,
        "per_query_total_tokens": EMBEDDING_TOKENS_PER_QUERY + per_query_llm_input + per_query_llm_output,
    }


def run_evaluation():
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("[!] Set GEMINI_API_KEY first.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    print("Loading index...", end=" ", flush=True)
    bm25, chunks = load_index()
    print(f"{len(chunks)} chunks\n")

    results = []
    total_input = 0
    total_output = 0

    print("=" * 70)
    for i, q in enumerate(SAMPLE_QUESTIONS, 1):
        print(f"[{i}/{len(SAMPLE_QUESTIONS)}] {q[:65]}...")
        r = answer(q, bm25, chunks, client)
        results.append(r)
        if i < len(SAMPLE_QUESTIONS):
            time.sleep(13)  # stay within 5 RPM free tier limit
        total_input += r["tokens"]["input"]
        total_output += r["tokens"]["output"]
        unanswerable = "cannot find" in r["answer"].lower()
        label = "[UNANSWERABLE]" if unanswerable else r["answer"][:80]
        print(f"  -> {label}...")
        print(f"     tokens: {r['tokens']['input']} in + {r['tokens']['output']} out")

    print("=" * 70)
    avg_in = total_input // len(results)
    avg_out = total_output // len(results)
    rag = rag_cost_estimate(len(chunks))

    print(f"\n{'-'*50}")
    print(f"  BM25 System (this repo)")
    print(f"{'-'*50}")
    print(f"  Embedding cost at index time  : 0 tokens  (BM25, no embeddings)")
    print(f"  Embedding cost per query      : 0 tokens")
    print(f"  Avg LLM input tokens/query    : {avg_in}")
    print(f"  Avg LLM output tokens/query   : {avg_out}")
    print(f"  Avg total tokens/query        : {avg_in + avg_out}")

    print(f"\n{'-'*50}")
    print(f"  Hypothetical RAG Baseline")
    print(f"{'-'*50}")
    print(f"  Embedding tokens at index time: {rag['one_time_index_tokens']:,}  (one-time)")
    print(f"  Embedding tokens per query    : {rag['per_query_embed_tokens']}")
    print(f"  Avg LLM input tokens/query    : {rag['per_query_llm_input_tokens']}")
    print(f"  Avg LLM output tokens/query   : {rag['per_query_llm_output_tokens']}")
    print(f"  Avg total tokens/query        : {rag['per_query_total_tokens']}")

    print(f"\n{'-'*50}")
    print(f"  Comparison")
    print(f"{'-'*50}")
    our_total = avg_in + avg_out
    rag_total = rag["per_query_total_tokens"]
    ratio = our_total / rag_total if rag_total else 1
    print(f"  Cost ratio (ours / RAG)       : {ratio:.2f}x  ({'cheaper' if ratio < 1 else 'comparable'})")
    print(f"  Embedding cost saved          : 100% per query")

    out_path = Path("evaluation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "per_question": results,
            "summary": {
                "our_avg_tokens_per_query": our_total,
                "rag_avg_tokens_per_query": rag_total,
                "ratio": round(ratio, 3),
                "embedding_cost_saved": "100%",
            },
        }, f, indent=2, ensure_ascii=False)

    print(f"\nFull results → '{out_path}'")


if __name__ == "__main__":
    run_evaluation()
