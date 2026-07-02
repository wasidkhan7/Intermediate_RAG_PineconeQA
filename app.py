# app/streamlit_app.py

import streamlit as st
from config import get_settings, validate_settings
from modules.embeddings import Embedder
from modules.pinecone_db import PineconeDB, VectorStoreConnectionError
from modules.loader import InvalidPDFError
from modules.pipeline import ingest_pdf, answer_query
from modules.utils import read_recent_logs

# ------------------------------------------------------------------ #
#  Page config — must be the first Streamlit call
# ------------------------------------------------------------------ #

st.set_page_config(
    page_title="RAG over PDFs · Pinecone",
    page_icon="📄",
    layout="wide",
)

# ------------------------------------------------------------------ #
#  Settings validation
# ------------------------------------------------------------------ #

settings = get_settings()
problems = validate_settings(settings)

st.title("📄 RAG System — PDF Question Answering")
st.caption(
    "Upload one or more PDFs, then ask questions answered "
    "strictly from their content."
)

if problems:
    st.error(
        "Configuration problem(s):\n\n" +
        "\n".join(f"- {p}" for p in problems) +
        "\n\nSet these in a `.env` file (see `.env.example`)."
    )
    st.stop()

# ------------------------------------------------------------------ #
#  Session state — persists across rerenders
# ------------------------------------------------------------------ #

if "documents" not in st.session_state:
    # { doc_name: { "namespace": str, "num_pages": int, "num_chunks": int } }
    st.session_state.documents = {}

if "query_history" not in st.session_state:
    # [ { "query": str, "answer": str, "grounded": bool } ]
    st.session_state.query_history = []

# ------------------------------------------------------------------ #
#  Cached resources — loaded once per session, not on every rerender
# ------------------------------------------------------------------ #

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder() -> Embedder:
    return Embedder(settings.embedding_model)


@st.cache_resource(show_spinner="Connecting to Pinecone...")
def load_db(_embedder: Embedder) -> PineconeDB:
    return PineconeDB(
        api_key=settings.pinecone_api_key,
        index_name=settings.pinecone_index_name,
        dimension=settings.embedding_dim,
        embeddings=_embedder.model,
        cloud=settings.pinecone_cloud,
        region=settings.pinecone_region,
    )


try:
    embedder = load_embedder()
    db = load_db(embedder)
except VectorStoreConnectionError as exc:
    st.error(f"❌ Could not connect to Pinecone: {exc}")
    st.stop()

# ------------------------------------------------------------------ #
#  Sidebar — all tunable settings
# ------------------------------------------------------------------ #

with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("Chunking")
    chunk_size = st.slider(
        "Chunk size (characters)",
        min_value=200, max_value=2000, value=1000, step=100
    )
    chunk_overlap = st.slider(
        "Chunk overlap (characters)",
        min_value=0, max_value=400, value=200, step=20
    )

    st.subheader("Retrieval")
    top_k = st.slider(
        "Top-K chunks to retrieve",
        min_value=1, max_value=15, value=5
    )
    similarity_threshold = st.slider(
        "Similarity threshold",
        min_value=0.0, max_value=1.0, value=0.3, step=0.05
    )

    st.subheader("Metadata filter")
    use_page_filter = st.checkbox("Filter by specific page number")
    page_filter = None
    if use_page_filter:
        page_filter = st.number_input(
            "Page number", min_value=1, step=1, value=1
        )

    st.divider()

    st.subheader("📚 Indexed documents")
    if st.session_state.documents:
        for name, meta in st.session_state.documents.items():
            st.markdown(
                f"**{name}**  \n"
                f"{meta['num_pages']} pages · {meta['num_chunks']} chunks"
            )
    else:
        st.caption("No documents indexed yet.")

# ------------------------------------------------------------------ #
#  Section 1 — PDF Upload
# ------------------------------------------------------------------ #

st.subheader("1️⃣ Upload PDF(s)")

uploaded_files = st.file_uploader(
    "Upload one or more PDF files (max 20 MB each)",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    if st.button("📥 Process & Index", type="primary"):
        for uf in uploaded_files:
            doc_name = uf.name

            if doc_name in st.session_state.documents:
                st.info(f"'{doc_name}' is already indexed — skipping.")
                continue

            try:
                with st.spinner(f"Processing '{doc_name}'..."):
                    result = ingest_pdf(
                        file_bytes=uf.read(),
                        doc_name=doc_name,
                        db=db,
                        max_pdf_size_mb=settings.max_pdf_size_mb,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )

                st.session_state.documents[doc_name] = {
                    "namespace": result.namespace,
                    "num_pages": result.num_pages,
                    "num_chunks": result.num_chunks,
                }
                st.success(
                    f"✅ '{doc_name}' indexed: "
                    f"{result.num_pages} pages → {result.num_chunks} chunks"
                )

            except InvalidPDFError as exc:
                st.error(f"❌ '{doc_name}': {exc}")
            except VectorStoreConnectionError as exc:
                st.error(f"❌ Pinecone error: {exc}")

st.divider()

# ------------------------------------------------------------------ #
#  Section 2 — Query
# ------------------------------------------------------------------ #

st.subheader("2️⃣ Ask a question")

if not st.session_state.documents:
    st.info("Upload and index at least one PDF before asking questions.")

else:
    selected_docs = st.multiselect(
        "Search within (select one or more documents)",
        options=list(st.session_state.documents.keys()),
        default=list(st.session_state.documents.keys()),
    )

    query = st.text_input(
        "Your question",
        placeholder="e.g. What dataset was used for training?",
    )

    if st.button("🔍 Ask", type="primary"):
        if not query.strip():
            st.warning("Please enter a non-empty question.")
        elif not selected_docs:
            st.warning("Select at least one document to search within.")
        else:
            namespaces = [
                st.session_state.documents[d]["namespace"]
                for d in selected_docs
            ]

            try:
                with st.spinner("Retrieving context and generating answer..."):
                    result = answer_query(
                        query=query,
                        doc_namespaces=namespaces,
                        doc_names=selected_docs,
                        db=db,
                        groq_api_key=settings.groq_api_key,
                        groq_model=settings.groq_model,
                        top_k=top_k,
                        similarity_threshold=similarity_threshold,
                        page_filter=int(page_filter) if page_filter else None,
                    )

                # Save to session history
                st.session_state.query_history.append({
                    "query": query,
                    "answer": result.answer,
                    "grounded": result.grounded,
                })

                # Display answer
                st.markdown("### Answer")
                if result.grounded:
                    st.success(result.answer)
                else:
                    st.warning(result.answer)

                # Display source attribution
                if result.used_chunks:
                    st.markdown("### 📎 Sources")
                    for i, chunk in enumerate(result.used_chunks, start=1):
                        with st.expander(
                            f"Source {i} · {chunk.doc_name} · "
                            f"Page {chunk.page_number} · "
                            f"Score {chunk.score:.2f}"
                        ):
                            st.write(chunk.text)

                            if chunk.score >= 0.7:
                                st.caption("🟢 High confidence")
                            elif chunk.score >= 0.45:
                                st.caption("🟡 Medium confidence")
                            else:
                                st.caption("🔴 Low confidence")

            except ValueError as exc:
                st.warning(str(exc))
            except VectorStoreConnectionError as exc:
                st.error(f"❌ Pinecone error: {exc}")
            except RuntimeError as exc:
                st.error(f"❌ {exc}")

st.divider()

# ------------------------------------------------------------------ #
#  Section 3 — Query history
# ------------------------------------------------------------------ #

with st.expander("🕘 Query history (this session)"):
    if not st.session_state.query_history:
        st.caption("No queries yet.")
    else:
        for item in reversed(st.session_state.query_history):
            tag = "✅" if item["grounded"] else "⚠️"
            st.markdown(f"{tag} **Q:** {item['query']}")
            st.markdown(f"**A:** {item['answer']}")
            st.markdown("---")

with st.expander("📜 Logged queries (persisted to disk)"):
    logs = read_recent_logs(limit=20)
    if not logs:
        st.caption("No logs yet.")
    else:
        st.dataframe(logs, use_container_width=True)