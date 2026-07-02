"""
modules/chunker.py
-------------------
Takes the page-level text produced by loader.py and splits it into
overlapping, token-sized chunks suitable for embedding. Each chunk
keeps track of which page and document it came from, plus a unique
chunk_id, so later steps (Pinecone metadata, source attribution) have
everything they need.
"""

# app/core/chunker.py

import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def _make_chunk_id(doc_name: str, page_number: int, chunk_index: int, text: str) -> str:
    raw = f"{doc_name}-{page_number}-{chunk_index}-{text[:30]}"
    short_hash = hashlib.sha1(raw.encode()).hexdigest()[:10]
    return f"{doc_name}_p{page_number}_c{chunk_index}_{short_hash}"


def chunk_documents(
    docs: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:

    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap")

    # Step 1 — create the splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    # Step 2 — split all documents
    chunks: list[Document] = splitter.split_documents(docs)

    # Step 3 — enrich metadata on each chunk
    for index, chunk in enumerate(chunks):
        doc_name = chunk.metadata.get("doc_name", "unknown")
        page_number = chunk.metadata.get("page_number", 1)
        chunk.metadata["chunk_id"] = _make_chunk_id(
            doc_name, page_number, index, chunk.page_content
        )
        chunk.metadata["chunk_index"] = index

    return chunks