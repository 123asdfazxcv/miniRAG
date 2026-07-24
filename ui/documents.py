"""
RAG -- Tab: Document Management

Three sub-tabs:
  1. Upload documents (PDF / TXT / paste / URL)
  2. Preset knowledge bases (demo + test docs)
  3. Indexed documents overview
"""

import re
import urllib.request
from pathlib import Path
from typing import Optional

import streamlit as st
from minirag import Document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_engine() -> None:
    """Create engine in session state if it does not exist."""
    if st.session_state.get("engine") is None:
        from minirag import RAGConfig, init
        from retrieval import EnhancedMiniRAG
        init()
        config = st.session_state.get("config", RAGConfig())
        strategy = st.session_state.get("strategy", "default")
        st.session_state.engine = EnhancedMiniRAG(config, strategy=strategy)


def _ensure_session_keys() -> None:
    """Initialise session-state keys used by this tab."""
    for key, default in [
        ("indexed_docs", []),
        ("chunk_count", 0),
        ("messages", []),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default


def _add_documents(docs: list[Document], source_name: str, source_type: str) -> None:
    """Index *docs* into the engine and update session state."""
    _ensure_engine()
    try:
        count = st.session_state.engine.ingest(docs)
        st.session_state.chunk_count = count
        st.session_state.indexed_docs = [{
            "source": source_name,
            "type": source_type,
            "chunks": count,
        }]
        st.success(f"Indexed: {source_name} ({count} chunks)")
    except Exception as exc:
        st.error(f"Index failed: {exc}")


def _load_demo_knowledge() -> list[Document]:
    """Read the LightSpeed Tech demo knowledge base from project root."""
    demo_path = Path(__file__).resolve().parent.parent / "demo_knowledge.txt"
    text = demo_path.read_text(encoding="utf-8")
    return [Document(content=text, metadata={"source": "demo_knowledge.txt", "type": "txt"})]


def _load_test_documents() -> list[Document]:
    """Return the built-in cat/car test documents from minirag."""
    from minirag import TEST_DOCUMENTS
    return list(TEST_DOCUMENTS)


def _fetch_url_text(url: str, timeout: int = 5) -> Optional[str]:
    """Fetch *url*, strip HTML tags, and return plain text if > 50 chars."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        text = re.sub(r"<[^>]+>", "", html)
        text = re.sub(r"\s+", " ", text).strip()
        return text if len(text) > 50 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Tab 1 -- Upload documents
# ---------------------------------------------------------------------------

def _render_upload_tab() -> None:
    st.subheader("Upload documents")

    source_type = st.radio(
        "Source type",
        options=["PDF file", "TXT file", "Paste text", "URL fetch"],
        horizontal=True,
        key="doc_source_type",
    )

    if source_type == "PDF file":
        _render_pdf_upload()
    elif source_type == "TXT file":
        _render_txt_upload()
    elif source_type == "Paste text":
        _render_paste_text()
    else:
        _render_url_fetch()


def _render_pdf_upload() -> None:
    uploaded = st.file_uploader("Choose a PDF file", type=["pdf"], key="pdf_upload")
    if uploaded is None:
        return

    if st.button("Index PDF", key="btn_index_pdf", use_container_width=True):
        try:
            import pdfplumber
            with pdfplumber.open(uploaded) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
            text = "\n".join(pages).strip()
            if not text:
                st.error("No extractable text found in the PDF.")
                return
            doc = Document(content=text, metadata={"source": uploaded.name, "type": "pdf"})
            _add_documents([doc], uploaded.name, "pdf")
        except Exception as exc:
            st.error(f"Failed to read PDF: {exc}")


def _render_txt_upload() -> None:
    uploaded = st.file_uploader("Choose a TXT file", type=["txt"], key="txt_upload")
    if uploaded is None:
        return

    if st.button("Index TXT", key="btn_index_txt", use_container_width=True):
        try:
            text = uploaded.read().decode("utf-8")
            if not text.strip():
                st.error("File is empty.")
                return
            doc = Document(content=text, metadata={"source": uploaded.name, "type": "txt"})
            _add_documents([doc], uploaded.name, "txt")
        except Exception as exc:
            st.error(f"Failed to read TXT: {exc}")


def _render_paste_text() -> None:
    doc_name = st.text_input("Document name (optional)", key="paste_name", placeholder="my-doc")
    pasted = st.text_area("Paste document content", height=250, key="paste_content")

    if not pasted.strip():
        return

    if st.button("Index pasted text", key="btn_index_paste", use_container_width=True):
        source = doc_name.strip() or "pasted-text"
        doc = Document(content=pasted, metadata={"source": source, "type": "paste"})
        _add_documents([doc], source, "paste")


def _render_url_fetch() -> None:
    url = st.text_input("URL", key="url_input", placeholder="https://example.com/article")

    if not url.strip():
        return

    if st.button("Fetch & Index URL", key="btn_index_url", use_container_width=True):
        with st.spinner("Fetching..."):
            text = _fetch_url_text(url.strip())
        if text is None:
            st.error("Failed to fetch URL or returned text was too short (< 50 chars).")
            return
        st.info(f"Fetched {len(text)} characters.")
        doc = Document(content=text, metadata={"source": url.strip(), "type": "url"})
        _add_documents([doc], url.strip(), "url")


# ---------------------------------------------------------------------------
# Tab 2 -- Preset knowledge bases
# ---------------------------------------------------------------------------

def _render_presets_tab() -> None:
    st.subheader("Preset knowledge bases")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Load LightSpeed Tech KB", key="btn_preset_demo", use_container_width=True):
            with st.spinner("Loading demo knowledge base..."):
                docs = _load_demo_knowledge()
            _add_documents(docs, "demo_knowledge.txt", "preset")
            st.info(f"Loaded {len(docs)} document(s) from demo_knowledge.txt")

    with col2:
        if st.button("Load test docs (Cats/Cars)", key="btn_preset_test", use_container_width=True):
            with st.spinner("Loading test documents..."):
                docs = _load_test_documents()
            _add_documents(docs, "TEST_DOCUMENTS", "preset")
            st.info(f"Loaded {len(docs)} documents (cats + cars)")


# ---------------------------------------------------------------------------
# Tab 3 -- Indexed documents overview
# ---------------------------------------------------------------------------

def _render_overview_tab() -> None:
    st.subheader("Indexed documents")

    if not st.session_state.indexed_docs:
        st.info("No documents indexed yet. Upload or load a preset first.")
        return

    # -- metrics row --
    col_a, col_b = st.columns(2)
    col_a.metric("Documents", len(st.session_state.indexed_docs))
    col_b.metric("Chunks", st.session_state.chunk_count)

    # -- data table --
    st.dataframe(
        st.session_state.indexed_docs,
        use_container_width=True,
        column_config={
            "source": "Source",
            "type": "Type",
            "chunks": "Chunks",
        },
    )

    # -- reset --
    if st.button("Clear all indexes", key="btn_clear_all", use_container_width=True):
        st.session_state.engine = None
        st.session_state.indexed_docs = []
        st.session_state.chunk_count = 0
        st.session_state.messages = []
        st.success("All indexes cleared.")
        st.rerun()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_documents_tab() -> None:
    """Render the document-management tab with three sub-tabs."""
    _ensure_session_keys()

    tab_upload, tab_presets, tab_overview = st.tabs([
        "📤 上传文档",
        "📦 预设知识库",
        "📋 已索引文档",
    ])

    with tab_upload:
        _render_upload_tab()
    with tab_presets:
        _render_presets_tab()
    with tab_overview:
        _render_overview_tab()
