# src/rag_engine.py
import os
import warnings

# ── Suppress known harmless warnings BEFORE any torch/transformers imports ────
warnings.filterwarnings("ignore", message=".*position_ids.*")
warnings.filterwarnings("ignore", message=".*torch.classes.*")

# Fix: stop Streamlit's file watcher from crashing on torch internals
# This eliminates the repeated "torch.classes raised" spam in logs
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import streamlit as st

# ── Prefer the updated non-deprecated package ─────────────────────────────────
try:
    from langchain_huggingface import HuggingFaceEmbeddings   # pip install langchain-huggingface
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # fallback


class MedicalRAG:
    """
    Chunks OCR-extracted report text, embeds it with a lightweight
    sentence-transformer, stores vectors in FAISS (in-memory), and
    retrieves the most relevant chunks for any follow-up question.
    Runs entirely on CPU — no GPU required.
    """

    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # ~80 MB, fast on CPU

    def __init__(self):
        # Cache embedder in session_state so it loads only once per session
        if "rag_embedder" not in st.session_state:
            with st.spinner("🔧 Loading embedding model (one-time setup)…"):
                st.session_state.rag_embedder = HuggingFaceEmbeddings(
                    model_name=self.EMBED_MODEL,
                    model_kwargs={
                        "device": "cpu",           # explicit CPU
                        "trust_remote_code": False,
                    },
                    encode_kwargs={
                        "normalize_embeddings": True,
                        "batch_size": 32,          # smaller batch = less RAM spike
                    },
                )
        self.embedder = st.session_state.rag_embedder
        self.vector_store = None

    # ── Build FAISS index from report text ────────────────────────────────────
    def index_report(self, text: str) -> int:
        """
        Split the report into overlapping chunks, embed each one,
        and store in a FAISS vector index.
        Returns the number of chunks created.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,    # ~100 words per chunk
            chunk_overlap=60,  # overlap preserves context at boundaries
            separators=["\n\n", "\n", ".", " "],
        )
        chunks = splitter.split_text(text)

        if not chunks:
            return 0

        docs = [
            Document(page_content=chunk, metadata={"chunk_id": i})
            for i, chunk in enumerate(chunks)
        ]

        self.vector_store = FAISS.from_documents(docs, self.embedder)
        return len(chunks)

    # ── Retrieve top-k relevant chunks for a question ─────────────────────────
    def retrieve(self, question: str, k: int = 4) -> str:
        """
        Embed the question, run cosine similarity search, and return
        the top-k most relevant chunks as a single context string.
        """
        if not self.vector_store:
            return ""
        results = self.vector_store.similarity_search(question, k=k)
        return "\n\n---\n\n".join(r.page_content for r in results)

    # ── Retrieve with scores (useful for debugging / transparency) ────────────
    def retrieve_with_scores(self, question: str, k: int = 4) -> list:
        """Returns list of (chunk_text, score) tuples, best match first."""
        if not self.vector_store:
            return []
        results = self.vector_store.similarity_search_with_score(question, k=k)
        return [(doc.page_content, float(score)) for doc, score in results]

    # ── Clear index when a new report is uploaded ─────────────────────────────
    def clear(self):
        """Reset so the next report gets a fresh index."""
        self.vector_store = None

    @property
    def is_ready(self) -> bool:
        return self.vector_store is not None