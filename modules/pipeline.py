# app/core/pipeline.py

from dataclasses import dataclass
from langchain_core.documents import Document
from .loader import load_pdf, InvalidPDFError
from .chunker import chunk_documents
from .embeddings import Embedder
from .pinecone_db import PineconeDB, RetrievedChunk
from .llm_generator import generate_answer, GeneratedAnswer
from .utils import log_query

@dataclass
class IngestResult:
    doc_name: str
    namespace: str
    num_pages: int
    num_chunks: int

def ingest_pdf(
    file_bytes: bytes,
    doc_name: str,
    db: PineconeDB,
    max_pdf_size_mb: int,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> IngestResult:

    # Step 1 — load and extract text page by page
    pages: list[Document] = load_pdf(
        file_bytes=file_bytes,
        doc_name=doc_name,
        max_size_mb=max_pdf_size_mb,
    )

    # Step 2 — split pages into overlapping chunks
    chunks: list[Document] = chunk_documents(
        docs=pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if not chunks:
        raise InvalidPDFError(
            "No usable text chunks could be produced from this PDF."
        )

    # Step 3 — upsert chunks into Pinecone
    # (embedding now happens inside db.upsert_chunks via LangChain)
    namespace = PineconeDB.namespace_for(doc_name)
    db.upsert_chunks(
        namespace=namespace,
        chunks=chunks,
    )

    return IngestResult(
        doc_name=doc_name,
        namespace=namespace,
        num_pages=len(pages),
        num_chunks=len(chunks),
    )


def answer_query(
    query: str,
    doc_namespaces: list[str],
    doc_names: list[str],
    db: PineconeDB,
    groq_api_key: str,
    groq_model: str,
    top_k: int = 5,
    similarity_threshold: float = 0.3,
    page_filter: int | None = None,
) -> GeneratedAnswer:

    if not query.strip():
        raise ValueError("Query must not be empty.")

    # Step 1 — retrieve from every selected document namespace
    all_matches: list[RetrievedChunk] = []
    for namespace in doc_namespaces:
        matches = db.query(
            namespace=namespace,
            query=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            page_filter=page_filter,
        )
        all_matches.extend(matches)

    # Step 2 — merge results from all documents, re-rank by score
    all_matches.sort(key=lambda m: m.score, reverse=True)
    top_matches = all_matches[:top_k]

    # Step 3 — generate grounded answer from retrieved context
    result = generate_answer(
        query=query,
        retrieved_chunks=top_matches,
        api_key=groq_api_key,
        model=groq_model,
    )

    # Step 4 — log the query to disk
    log_query(
        doc_names=doc_names,
        query=query,
        top_k=top_k,
        grounded=result.grounded,
        num_results=len(top_matches),
    )

    return result
