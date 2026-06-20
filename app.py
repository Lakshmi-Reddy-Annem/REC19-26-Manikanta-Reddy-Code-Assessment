"""
Simple Gradio UI for the BM25 document Q&A system.

Usage:
    set GEMINI_API_KEY=your_key      (Windows)
    export GEMINI_API_KEY=your_key   (Linux/Mac)
    python app.py
"""

import os
import sys
from pathlib import Path

import gradio as gr
from google import genai

sys.path.insert(0, str(Path(__file__).parent))
from ask import load_index, answer

api_key = os.environ.get("GEMINI_API_KEY", "").strip()
if not api_key:
    print("[!] GEMINI_API_KEY not set. Set it before running app.py.")
    sys.exit(1)

client = genai.Client(api_key=api_key)

print("Loading index...", end=" ", flush=True)
bm25, chunks = load_index()
print(f"{len(chunks)} chunks ready.")


def query(question: str):
    if not question.strip():
        return "", "", ""

    result = answer(question.strip(), bm25, chunks, client)

    sources_md = "\n".join(
        f"- **{s['doc']}** | {s['chapter']} | Pages {s['pages']}  *(BM25 score: {s['bm25_score']})*"
        for s in result["sources"]
    )

    t = result["tokens"]
    cost_line = (
        f"`{t['input']} in + {t['output']} out = {t['total']} total tokens`  "
        f"— zero embedding cost (BM25 retrieval)"
    )

    return result["answer"], sources_md, cost_line


with gr.Blocks(title="Defence Doc Q&A") as demo:
    gr.Markdown("## Defence Document Q&A\nBM25 retrieval + Gemini — no embeddings")

    with gr.Row():
        question_box = gr.Textbox(
            label="Your question",
            placeholder="e.g. What is the gun salute for the President of India?",
            scale=5,
        )
        ask_btn = gr.Button("Ask", variant="primary", scale=1)

    answer_box = gr.Markdown(label="Answer")

    with gr.Accordion("Sources retrieved (BM25)", open=False):
        sources_box = gr.Markdown()

    cost_box = gr.Markdown(label="Token cost")

    ask_btn.click(query, inputs=question_box, outputs=[answer_box, sources_box, cost_box])
    question_box.submit(query, inputs=question_box, outputs=[answer_box, sources_box, cost_box])

    gr.Examples(
        examples=[
            "What is the gun salute accorded to the President of India?",
            "What are the eligibility requirements for enrolment in the Indian Naval Auxiliary Service?",
            "Who can grant acting rank and under what conditions?",
            "What is the procurement budget for aircraft carrier construction in FY2024-25?",
        ],
        inputs=question_box,
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
