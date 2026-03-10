#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) pipeline for SepsisGuard.

Indexes the Surviving Sepsis Campaign 2021 guidelines into a vector store
and retrieves relevant clinical recommendations for a given patient context.

Components:
  1. Document loader & chunker (PDF/text → overlapping chunks)
  2. Embedding generator (sentence-transformers)
  3. Vector store (Weaviate or local FAISS fallback)
  4. Retriever (top-k similarity search with metadata filtering)

Usage:
    # Index guidelines
    python rag/rag_pipeline.py --mode index --input-path rag/guidelines/ssc_2021.txt

    # Query
    python rag/rag_pipeline.py --mode query --query "lactate elevated treatment"

    # Interactive
    python rag/rag_pipeline.py --mode interactive
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

# Optional imports (graceful fallback)
try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    import weaviate
    HAS_WEAVIATE = True
except ImportError:
    HAS_WEAVIATE = False


# Data structures

@dataclass
class Chunk:
    """A single chunk of clinical guideline text with metadata."""
    text: str
    chunk_id: str
    source: str
    section: str = ""
    recommendation_number: str = ""
    strength: str = ""            # "strong" / "weak" / "BPS"
    page: int = 0
    embedding: Optional[list[float]] = field(default=None, repr=False)


@dataclass
class RetrievalResult:
    """A retrieved chunk with its similarity score."""
    chunk: Chunk
    score: float


# 1. Document Loading & Chunking

class GuidelineChunker:
    """
    Splits clinical guideline text into overlapping chunks with metadata.

    Strategy:
      - Split by double newlines (paragraph boundaries)
      - Merge small paragraphs until ~chunk_size tokens
      - Overlap by ~overlap tokens for context continuity
      - Detect recommendation numbers (e.g., "Recommendation 14:")
        and section headers for metadata tagging
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _approx_tokens(self, text: str) -> int:
        """Rough token count (words ≈ 0.75 tokens)."""
        return int(len(text.split()) * 1.3)

    def _extract_metadata(self, text: str) -> dict:
        """Extract recommendation number and strength from chunk text."""
        meta: dict = {"recommendation_number": "", "strength": "", "section": ""}

        # Look for recommendation patterns
        import re
        rec_match = re.search(r"[Rr]ecommendation\s+(\d+)", text)
        if rec_match:
            meta["recommendation_number"] = rec_match.group(1)

        # Strength of recommendation
        text_lower = text.lower()
        if "strong recommendation" in text_lower:
            meta["strength"] = "strong"
        elif "weak recommendation" in text_lower or "conditional" in text_lower:
            meta["strength"] = "weak"
        elif "best practice" in text_lower or "bps" in text_lower:
            meta["strength"] = "BPS"

        # Section detection
        section_keywords = {
            "resuscitation": "Initial Resuscitation",
            "antimicrobial": "Antimicrobial Therapy",
            "hemodynamic": "Hemodynamic Management",
            "ventilation": "Ventilation",
            "vasopressor": "Vasopressors",
            "corticosteroid": "Corticosteroids",
            "fluid": "Fluid Therapy",
            "blood": "Blood Products",
            "glucose": "Glucose Control",
            "renal": "Renal Replacement",
            "nutrition": "Nutrition",
            "sedation": "Sedation & Analgesia",
        }
        for keyword, section_name in section_keywords.items():
            if keyword in text_lower:
                meta["section"] = section_name
                break

        return meta

    def chunk_text(self, text: str, source: str = "SSC-2021") -> list[Chunk]:
        """Split text into overlapping chunks with metadata."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        chunks: list[Chunk] = []
        current_text = ""
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self._approx_tokens(para)

            if current_tokens + para_tokens > self.chunk_size and current_text:
                # Save current chunk
                chunk = self._create_chunk(current_text, source, len(chunks))
                chunks.append(chunk)

                # Overlap: keep last portion
                words = current_text.split()
                overlap_words = words[-min(self.overlap, len(words)):]
                current_text = " ".join(overlap_words) + "\n\n" + para
                current_tokens = self._approx_tokens(current_text)
            else:
                current_text = (current_text + "\n\n" + para).strip()
                current_tokens += para_tokens

        # Final chunk
        if current_text.strip():
            chunks.append(self._create_chunk(current_text, source, len(chunks)))

        return chunks

    def _create_chunk(self, text: str, source: str, index: int) -> Chunk:
        """Create a Chunk with auto-extracted metadata."""
        meta = self._extract_metadata(text)
        chunk_id = hashlib.md5(text.encode()).hexdigest()[:12]
        return Chunk(
            text=text.strip(),
            chunk_id=f"{source}_{index:04d}_{chunk_id}",
            source=source,
            section=meta["section"],
            recommendation_number=meta["recommendation_number"],
            strength=meta["strength"],
        )

    def chunk_file(self, filepath: str | Path) -> list[Chunk]:
        """Load a text file and chunk it."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        text = path.read_text(encoding="utf-8")
        return self.chunk_text(text, source=path.stem)


# 2. Embedding Generation

class EmbeddingModel:
    """
    Generate embeddings using sentence-transformers.
    Falls back to a simple TF-IDF approach if sentence-transformers is unavailable.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.dim = 384  # default for MiniLM

        if HAS_SBERT:
            print(f"  Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.dim = self.model.get_sentence_embedding_dimension()
        else:
            print("  Warning: sentence-transformers not installed.")
            print("  Using fallback TF-IDF embeddings (install sentence-transformers for production).")
            self._build_fallback()

    def _build_fallback(self) -> None:
        """Simple hash-based embedding fallback for development."""
        self.dim = 128

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts into embedding vectors."""
        if self.model:
            return self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)

        # Fallback: deterministic hash-based pseudo-embeddings
        embeddings = []
        for text in texts:
            words = text.lower().split()
            vec = np.zeros(self.dim, dtype=np.float32)
            for w in words:
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                idx = h % self.dim
                vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            embeddings.append(vec)
        return np.array(embeddings)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string."""
        return self.encode([query])[0]


# 3. Vector Store

class LocalVectorStore:
    """
    Local FAISS-based vector store (production uses Weaviate).
    Falls back to brute-force numpy search if FAISS is unavailable.
    """

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.chunks: list[Chunk] = []
        self.index = None

        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(embedding_dim)  # inner product (cosine after normalization)
            print(f"  FAISS index initialized (dim={embedding_dim})")
        else:
            self._embeddings: list[np.ndarray] = []
            print(f"  Using numpy brute-force search (dim={embedding_dim})")

    def add_chunks(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Add chunks with their embeddings to the store."""
        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings_norm = embeddings / norms

        if HAS_FAISS and self.index is not None:
            self.index.add(embeddings_norm.astype(np.float32))
        else:
            self._embeddings.extend(embeddings_norm)

        for chunk, emb in zip(chunks, embeddings_norm.tolist()):
            chunk.embedding = emb
            self.chunks.append(chunk)

        print(f"  Indexed {len(chunks)} chunks (total: {len(self.chunks)})")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve top-k most similar chunks."""
        qe = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        qe = qe.astype(np.float32).reshape(1, -1)

        if HAS_FAISS and self.index is not None:
            scores, indices = self.index.search(qe, min(top_k, len(self.chunks)))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.chunks):
                    results.append(RetrievalResult(chunk=self.chunks[idx], score=float(score)))
            return results
        else:
            # Numpy fallback
            if not self._embeddings:
                return []
            emb_matrix = np.array(self._embeddings, dtype=np.float32)
            scores = emb_matrix @ qe.T
            scores = scores.squeeze()
            top_indices = np.argsort(scores)[::-1][:top_k]
            return [
                RetrievalResult(chunk=self.chunks[i], score=float(scores[i]))
                for i in top_indices
            ]

    def save(self, path: str | Path) -> None:
        """Save vector store to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save chunks (without embeddings to save space)
        chunks_data = []
        for c in self.chunks:
            d = asdict(c)
            d.pop("embedding", None)
            chunks_data.append(d)

        with open(path / "chunks.json", "w") as f:
            json.dump(chunks_data, f, indent=2)

        # Save embeddings
        embeddings = np.array([c.embedding for c in self.chunks], dtype=np.float32)
        np.save(path / "embeddings.npy", embeddings)

        print(f"  Vector store saved to {path} ({len(self.chunks)} chunks)")

    def load(self, path: str | Path) -> None:
        """Load vector store from disk."""
        path = Path(path)

        with open(path / "chunks.json") as f:
            chunks_data = json.load(f)

        embeddings = np.load(path / "embeddings.npy")

        self.chunks = []
        for d, emb in zip(chunks_data, embeddings):
            chunk = Chunk(**{k: v for k, v in d.items() if k != "embedding"})
            chunk.embedding = emb.tolist()
            self.chunks.append(chunk)

        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.index.add(embeddings)
        else:
            self._embeddings = list(embeddings)

        print(f"  Loaded {len(self.chunks)} chunks from {path}")


# 4. RAG Pipeline (orchestrator)

class SepsisRAGPipeline:
    """
    End-to-end RAG pipeline for clinical recommendation retrieval.

    Usage:
        pipeline = SepsisRAGPipeline()
        pipeline.index_guidelines("rag/guidelines/ssc_2021.txt")
        results = pipeline.query("patient with elevated lactate and hypotension")
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ):
        self.chunker = GuidelineChunker(chunk_size=chunk_size, overlap=chunk_overlap)
        self.embedder = EmbeddingModel(model_name=embedding_model)
        self.store = LocalVectorStore(embedding_dim=self.embedder.dim)

    def index_guidelines(self, filepath: str | Path) -> int:
        """Load, chunk, embed, and index a guidelines document."""
        print(f"\nIndexing: {filepath}")

        # Chunk
        chunks = self.chunker.chunk_file(filepath)
        print(f"  Created {len(chunks)} chunks")

        # Embed
        texts = [c.text for c in chunks]
        embeddings = self.embedder.encode(texts)
        print(f"  Generated embeddings: shape={embeddings.shape}")

        # Index
        self.store.add_chunks(chunks, embeddings)
        return len(chunks)

    def index_text(self, text: str, source: str = "guidelines") -> int:
        """Index raw text directly (useful for testing)."""
        chunks = self.chunker.chunk_text(text, source=source)
        texts = [c.text for c in chunks]
        embeddings = self.embedder.encode(texts)
        self.store.add_chunks(chunks, embeddings)
        return len(chunks)

    def query(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[RetrievalResult]:
        """Retrieve relevant guideline chunks for a clinical query."""
        query_emb = self.embedder.encode_query(query)
        results = self.store.search(query_emb, top_k=top_k)
        return [r for r in results if r.score >= min_score]

    def format_context(self, results: list[RetrievalResult]) -> str:
        """Format retrieved chunks into a context string for the LLM."""
        if not results:
            return "No relevant guidelines found."

        parts = []
        for i, r in enumerate(results, 1):
            meta = []
            if r.chunk.section:
                meta.append(f"Section: {r.chunk.section}")
            if r.chunk.recommendation_number:
                meta.append(f"Rec #{r.chunk.recommendation_number}")
            if r.chunk.strength:
                meta.append(f"Strength: {r.chunk.strength}")
            meta_str = " | ".join(meta) if meta else "General"

            parts.append(
                f"[Source {i}] ({meta_str}, relevance: {r.score:.3f})\n{r.chunk.text}"
            )

        return "\n\n---\n\n".join(parts)

    def save(self, path: str | Path) -> None:
        self.store.save(path)

    def load(self, path: str | Path) -> None:
        self.store.load(path)


# CLI

def demo_with_sample_guidelines() -> None:
    """Run a demo with sample guideline text (no external files needed)."""
    sample_guidelines = """
Recommendation 1: Initial Resuscitation
For patients with sepsis-induced hypoperfusion or septic shock, we recommend that
at least 30 mL/kg of IV crystalloid fluid be given within the first 3 hours of
resuscitation. Strong recommendation, low quality of evidence.

Recommendation 2: Vasopressors
For adults with septic shock, we recommend norepinephrine as the first-line
vasopressor over other agents. Strong recommendation, high quality of evidence.
If the target MAP cannot be achieved with norepinephrine alone, we suggest adding
vasopressin (up to 0.03 U/min) rather than further titrating norepinephrine.
Weak recommendation, moderate quality of evidence.

Recommendation 3: Antimicrobial Therapy
For adults with possible septic shock or high likelihood of sepsis, we recommend
administering antimicrobials immediately, ideally within 1 hour of recognition.
Strong recommendation, low quality of evidence.
Empiric broad-spectrum therapy with one or more antimicrobials should be initiated
to cover all likely pathogens.

Recommendation 4: Lactate Monitoring
We recommend guiding resuscitation to decrease serum lactate in patients with
elevated lactate levels, as a marker of tissue hypoperfusion.
Weak recommendation, low quality of evidence.
Serial lactate measurements (every 2-4 hours) should be performed until lactate
normalizes to guide adequacy of resuscitation.

Recommendation 5: Corticosteroids
For adults with septic shock and an ongoing requirement for vasopressor therapy,
we suggest using IV corticosteroids. Weak recommendation, moderate quality of evidence.
Typical regimen: hydrocortisone 200 mg per day given as 50 mg IV every 6 hours.

Recommendation 6: Blood Glucose Control
For adults with sepsis or septic shock, we recommend initiating insulin therapy
when two consecutive blood glucose levels are > 180 mg/dL, targeting an upper
blood glucose level <= 180 mg/dL rather than <= 110 mg/dL.
Strong recommendation, high quality of evidence.

Recommendation 7: Mechanical Ventilation
For adults with sepsis-induced ARDS, we recommend using a low tidal volume
ventilation strategy (6 mL/kg predicted body weight) over higher tidal volumes.
Strong recommendation, high quality of evidence.
Plateau pressures should be maintained < 30 cm H2O.

Recommendation 8: Fluid Management
After initial resuscitation, for adults with sepsis or septic shock, we suggest
using conservative fluid management over a liberal fluid strategy.
Weak recommendation, low quality of evidence.
Balanced crystalloids (e.g., Ringer's lactate) are preferred over normal saline
for large-volume resuscitation.

Recommendation 9: Renal Replacement Therapy
For adults with sepsis and acute kidney injury who require renal replacement
therapy, we suggest using either continuous or intermittent renal replacement therapy.
Weak recommendation, low quality of evidence.

Recommendation 10: Nutrition
For adults with sepsis or septic shock who can be fed enterally, we recommend
early initiation of enteral nutrition (within 72 hours) rather than no early
enteral nutrition or parenteral nutrition only.
Weak recommendation, moderate quality of evidence.
    """

    print("\n  Running demo with sample SSC 2021 guidelines...")
    pipeline = SepsisRAGPipeline(chunk_size=150, chunk_overlap=30)
    n_chunks = pipeline.index_text(sample_guidelines, source="SSC-2021-sample")
    print(f"  Indexed {n_chunks} chunks\n")

    # Demo queries
    queries = [
        "patient has elevated lactate level, what is the recommended treatment?",
        "blood pressure is low, need vasopressor recommendation",
        "patient has high blood glucose in septic shock",
        "when should antibiotics be administered for sepsis?",
    ]

    for q in queries:
        print(f"\n{'─' * 60}")
        print(f"  Query: {q}")
        print(f"{'─' * 60}")
        results = pipeline.query(q, top_k=3)
        for i, r in enumerate(results, 1):
            sec = r.chunk.section or "General"
            rec = r.chunk.recommendation_number or "—"
            print(f"\n  [{i}] Score: {r.score:.3f} | Section: {sec} | Rec: {rec}")
            preview = r.chunk.text[:200].replace("\n", " ")
            print(f"      {preview}...")

    # Save index
    project_root = Path(__file__).resolve().parent.parent
    index_dir = project_root / "rag" / "index"
    pipeline.save(index_dir)
    print(f"\n  Index saved to {index_dir}")

    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="SepsisGuard RAG Pipeline")
    parser.add_argument("--mode", choices=["index", "query", "interactive", "demo"],
                        default="demo")
    parser.add_argument("--input-path", type=str, default=None)
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    print("=" * 60)
    print("  SepsisGuard — RAG Pipeline")
    print("=" * 60)

    if args.mode == "demo":
        demo_with_sample_guidelines()

    elif args.mode == "index":
        if not args.input_path:
            print("Error: --input-path required for index mode")
            sys.exit(1)
        pipeline = SepsisRAGPipeline()
        n = pipeline.index_guidelines(args.input_path)
        project_root = Path(__file__).resolve().parent.parent
        pipeline.save(project_root / "rag" / "index")
        print(f"\nIndexed {n} chunks.")

    elif args.mode == "query":
        if not args.query:
            print("Error: --query required for query mode")
            sys.exit(1)
        pipeline = SepsisRAGPipeline()
        project_root = Path(__file__).resolve().parent.parent
        pipeline.load(project_root / "rag" / "index")
        results = pipeline.query(args.query, top_k=args.top_k)
        context = pipeline.format_context(results)
        print(f"\n{context}")

    elif args.mode == "interactive":
        pipeline = SepsisRAGPipeline()
        project_root = Path(__file__).resolve().parent.parent
        index_path = project_root / "rag" / "index"
        if index_path.exists():
            pipeline.load(index_path)
        else:
            print("No existing index found. Running demo indexing...")
            demo_with_sample_guidelines()

        print("\nInteractive mode. Type 'quit' to exit.\n")
        while True:
            query = input("Query> ").strip()
            if query.lower() in ("quit", "exit", "q"):
                break
            if not query:
                continue
            results = pipeline.query(query, top_k=args.top_k)
            print(pipeline.format_context(results))
            print()


if __name__ == "__main__":
    main()
