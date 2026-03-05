from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import streamlit as st


class MedicalRAG:
    """
    Chunks the OCR-extracted report, embeds it, stores in FAISS,
    and retrieves relevant context for follow-up questions.
    """

    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small & fast, runs on CPU

    def __init__(self):
        # Cache the embedding model across reruns
        if "rag_embedder" not in st.session_state:
            with st.spinner("🔧 Loading embedding model (one-time)…"):
                st.session_state.rag_embedder = HuggingFaceEmbeddings(
                    model_name=self.EMBED_MODEL,
                    model_kwargs={"device": "cpu"},
                )
        self.embedder = st.session_state.rag_embedder
        self.vector_store = None

    # ── Build index from report text ──────────────────────────────────────────
    def index_report(self, text: str) -> int:
        """Chunk + embed the report. Returns number of chunks created."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=60,
            separators=["\n\n", "\n", ".", " "],
        )
        chunks = splitter.split_text(text)
        docs = [Document(page_content=c, metadata={"chunk": i})
                for i, c in enumerate(chunks)]

        self.vector_store = FAISS.from_documents(docs, self.embedder)
        return len(chunks)

    # ── Retrieve relevant context for a question ──────────────────────────────
    def retrieve(self, question: str, k: int = 4) -> str:
        """Return top-k relevant chunks joined as a single context string."""
        if not self.vector_store:
            return ""
        results = self.vector_store.similarity_search(question, k=k)
        return "\n\n---\n\n".join(r.page_content for r in results)

    # ── Convenience: is the index ready? ──────────────────────────────────────
    @property
    def is_ready(self) -> bool:
        return self.vector_store is not None