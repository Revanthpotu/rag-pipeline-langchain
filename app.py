"""Streamlit UI for the RAG Pipeline.

Provides a clean chat interface where users can:
1. Upload PDF documents for ingestion.
2. Ask natural-language questions about the uploaded content.
3. See which source documents were used to generate each answer.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import streamlit as st

from rag.config import OLLAMA_MODEL, TOP_K, UPLOAD_DIR
from rag.ingestion import clear_collection, get_ingested_sources, ingest_pdf
from rag.llm import generate_answer
from rag.retrieval import retrieve

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Pipeline",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container { max-width: 960px; }
    .source-box {
        background-color: rgba(100, 100, 100, 0.08);
        border-left: 3px solid #4A90D9;
        padding: 0.6rem 1rem;
        margin: 0.3rem 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Sidebar — Document Management
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📁 Document Manager")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files to add them to the knowledge base.",
    )

    if uploaded_files:
        if st.button("🔄 Ingest uploaded PDFs", use_container_width=True):
            upload_dir = Path(UPLOAD_DIR)
            upload_dir.mkdir(parents=True, exist_ok=True)

            progress = st.progress(0, text="Ingesting…")
            total = len(uploaded_files)

            for idx, uploaded_file in enumerate(uploaded_files):
                progress.progress(
                    (idx) / total,
                    text=f"Processing {uploaded_file.name}…",
                )
                try:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".pdf", dir=str(upload_dir)
                    ) as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_path = tmp.name

                    chunks = ingest_pdf(tmp_path, filename=uploaded_file.name)
                    st.success(f"✅ **{uploaded_file.name}** — {chunks} chunks ingested")
                except Exception as exc:
                    st.error(f"❌ **{uploaded_file.name}** — {exc}")

            progress.progress(1.0, text="Done!")

    st.divider()

    st.subheader("📚 Knowledge Base")
    sources = get_ingested_sources()
    if sources:
        for src in sources:
            st.markdown(f"- `{src}`")
    else:
        st.caption("No documents ingested yet.")

    st.divider()

    st.subheader("⚙️ Settings")
    st.caption(f"**LLM:** {OLLAMA_MODEL}")
    st.caption(f"**Top-K chunks:** {TOP_K}")

    if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
        clear_collection()
        st.session_state.messages = []
        st.success("Knowledge base cleared.")
        st.rerun()

# ---------------------------------------------------------------------------
# Main area — Chat Interface
# ---------------------------------------------------------------------------
st.title("📄 RAG Pipeline")
st.caption("Ask questions about your uploaded documents.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Sources used", expanded=False):
                for src in msg["sources"]:
                    st.markdown(
                        f'<div class="source-box">📄 <strong>{src["source"]}</strong> '
                        f'(chunk {src["chunk_index"]})</div>',
                        unsafe_allow_html=True,
                    )

if user_query := st.chat_input("Ask a question about your documents…"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer…"):
            sources_list = get_ingested_sources()
            if not sources_list:
                answer = (
                    "📭 **No documents in the knowledge base yet.**\n\n"
                    "Please upload one or more PDF files using the sidebar "
                    "and click **Ingest uploaded PDFs** first."
                )
                sources_meta: list[dict] = []
            else:
                query_result = retrieve(user_query)
                if not query_result.results:
                    answer = (
                        "I couldn't find any relevant information in the "
                        "uploaded documents for your question. Try rephrasing "
                        "or uploading additional documents."
                    )
                    sources_meta = []
                else:
                    answer = generate_answer(query_result.prompt)
                    sources_meta = query_result.sources

        st.markdown(answer)

        if sources_meta:
            with st.expander("📎 Sources used", expanded=False):
                for src in sources_meta:
                    st.markdown(
                        f'<div class="source-box">📄 <strong>{src["source"]}</strong> '
                        f'(chunk {src["chunk_index"]})</div>',
                        unsafe_allow_html=True,
                    )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources_meta}
    )
