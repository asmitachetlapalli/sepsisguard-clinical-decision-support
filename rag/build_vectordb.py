#!/usr/bin/env python3
"""
Build ChromaDB vector store from sepsis clinical guidelines.
Chunks the guidelines text and embeds using sentence-transformers.
"""

from __future__ import annotations

import re
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions


GUIDELINES_PATH = Path(__file__).parent / "sepsis_guidelines.txt"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "sepsis_guidelines"


def chunk_guidelines(text: str, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """Split guidelines into sections by headers, then sub-chunk if too long."""
    sections = re.split(r"\n(?=###?\s)", text)
    chunks = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Extract header if present
        header_match = re.match(r"(###?\s.+)", section)
        header = header_match.group(1) if header_match else ""

        if len(section) <= chunk_size:
            chunks.append({"text": section, "header": header})
        else:
            # Split long sections into overlapping chunks
            words = section.split()
            for i in range(0, len(words), chunk_size // 5):
                chunk_words = words[i : i + chunk_size // 5 + overlap // 5]
                chunk_text = " ".join(chunk_words)
                if chunk_text.strip():
                    chunks.append({"text": chunk_text, "header": header})

    return chunks


def build_db():
    print("Building ChromaDB vector store...")

    # Read guidelines
    text = GUIDELINES_PATH.read_text()
    chunks = chunk_guidelines(text)
    print(f"Created {len(chunks)} chunks from guidelines")

    # Set up ChromaDB with sentence-transformers embeddings
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Delete existing collection if it exists
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"description": "Sepsis clinical guidelines for RAG"},
    )

    # Add chunks
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    documents = [c["text"] for c in chunks]
    metadatas = [{"header": c["header"], "source": "Surviving Sepsis Campaign 2021"} for c in chunks]

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Added {len(chunks)} chunks to collection '{COLLECTION_NAME}'")
    print(f"ChromaDB persisted to {CHROMA_DIR}")

    # Test query
    results = collection.query(query_texts=["What is the first-line vasopressor for septic shock?"], n_results=3)
    print("\nTest query: 'What is the first-line vasopressor for septic shock?'")
    for i, doc in enumerate(results["documents"][0]):
        print(f"\n  Result {i+1}: {doc[:150]}...")

    print("\nDone!")


if __name__ == "__main__":
    build_db()
