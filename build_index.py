"""
Build the BM25 index from all PDFs in the data/ folder.
Run this once before querying.

Usage:
    python build_index.py
"""

import json
import pickle
import re
from pathlib import Path

import pdfplumber
from rank_bm25 import BM25Okapi

DATA_DIR = Path("data")
INDEX_DIR = Path("index")
INDEX_DIR.mkdir(exist_ok=True)

CHUNK_SIZE = 400   # words per chunk
OVERLAP = 80       # word overlap between chunks


def tokenize(text: str) -> list:
    return re.findall(r'\b[a-z0-9]+\b', text.lower())


def extract_pages(pdf_path: Path) -> list:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({
                    "doc": pdf_path.stem,
                    "page": page_num,
                    "text": text.strip(),
                })
    return pages


def detect_chapter(text: str) -> str:
    m = re.search(r'CHAPTER\s+([IVXLCDM]+|[0-9]+)', text, re.IGNORECASE)
    return f"Chapter {m.group(1)}" if m else ""


def pages_to_chunks(pages: list) -> list:
    """Sliding-window chunker over concatenated page words."""
    # Flatten to a word list keeping source metadata per word
    word_meta = []
    current_chapter = ""
    for p in pages:
        ch = detect_chapter(p["text"])
        if ch:
            current_chapter = ch
        for word in p["text"].split():
            word_meta.append((word, p["doc"], p["page"], current_chapter))

    chunks = []
    i = 0
    while i < len(word_meta):
        window = word_meta[i: i + CHUNK_SIZE]
        if not window:
            break
        text = " ".join(w[0] for w in window)
        doc = window[0][1]
        start_page = window[0][2]
        end_page = window[-1][2]
        chapter = next((w[3] for w in window if w[3]), "")
        chunks.append({
            "id": len(chunks),
            "doc": doc,
            "chapter": chapter,
            "start_page": start_page,
            "end_page": end_page,
            "text": text,
        })
        i += CHUNK_SIZE - OVERLAP

    return chunks


def build_index():
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"[!] No PDFs found in '{DATA_DIR}/'. Add your PDFs there and re-run.")
        return

    print(f"Found {len(pdf_files)} PDF(s):")
    all_chunks = []

    for pdf_path in pdf_files:
        print(f"  Parsing {pdf_path.name} ...", end=" ", flush=True)
        pages = extract_pages(pdf_path)
        chunks = pages_to_chunks(pages)
        all_chunks.extend(chunks)
        print(f"{len(pages)} pages -> {len(chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")
    print("Building BM25 index ...", end=" ", flush=True)

    tokenized = [tokenize(c["text"]) for c in all_chunks]
    bm25 = BM25Okapi(tokenized)

    with open(INDEX_DIR / "bm25_index.pkl", "wb") as f:
        pickle.dump(bm25, f)

    with open(INDEX_DIR / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print("done.")
    print(f"\nIndex written to '{INDEX_DIR}/'.")
    print("Next step: python ask.py")


if __name__ == "__main__":
    build_index()
