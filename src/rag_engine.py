# src/rag_engine.py
import os
import warnings

warnings.filterwarnings("ignore", message=".*position_ids.*")
warnings.filterwarnings("ignore", message=".*torch.classes.*")
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import streamlit as st

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


class MedicalRAG:
    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    _shared_embedder = None

    def __init__(self):
        self.embedder = self._get_embedder()
        self.vector_store = None

    @classmethod
    def _get_embedder(cls):
        if cls._shared_embedder is not None:
            return cls._shared_embedder

        try:
            from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
            ctx = get_script_run_ctx()
        except Exception:
            ctx = None

        if ctx is not None:
            if "rag_embedder" not in st.session_state:
                with st.spinner("Loading embedding model (one-time setup)…"):
                    st.session_state.rag_embedder = HuggingFaceEmbeddings(
                        model_name=cls.EMBED_MODEL,
                        model_kwargs={"device": "cpu", "trust_remote_code": False},
                        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
                    )
            cls._shared_embedder = st.session_state.rag_embedder
        else:
            cls._shared_embedder = HuggingFaceEmbeddings(
                model_name=cls.EMBED_MODEL,
                model_kwargs={"device": "cpu", "trust_remote_code": False},
                encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
            )

        return cls._shared_embedder

    def index_report(self, text: str) -> int:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=60,
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

    def create_index(self, texts: list[str]) -> int:
        """Index a list of text blocks (e.g. trial descriptions). Returns chunk count."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=80,
            separators=["\n\n", "\n", ".", " "],
        )
        all_docs = []
        doc_id = 0
        for text in texts:
            chunks = splitter.split_text(text)
            for chunk in chunks:
                all_docs.append(
                    Document(page_content=chunk, metadata={"chunk_id": doc_id})
                )
                doc_id += 1

        if not all_docs:
            return 0

        self.vector_store = FAISS.from_documents(all_docs, self.embedder)
        return len(all_docs)

    def retrieve(self, question: str, k: int = 4) -> list[str]:
        """Retrieve top-k relevant chunks as a list of strings."""
        if not self.vector_store:
            return []
        results = self.vector_store.similarity_search(question, k=k)
        return [r.page_content for r in results]

    def retrieve_as_text(self, question: str, k: int = 4) -> str:
        """Retrieve top-k relevant chunks as a single joined string."""
        chunks = self.retrieve(question, k)
        return "\n\n---\n\n".join(chunks)

    def retrieve_with_scores(self, question: str, k: int = 4) -> list:
        if not self.vector_store:
            return []
        results = self.vector_store.similarity_search_with_score(question, k=k)
        return [(doc.page_content, float(score)) for doc, score in results]

    def clear(self):
        self.vector_store = None

    @property
    def is_ready(self) -> bool:
        return self.vector_store is not None
