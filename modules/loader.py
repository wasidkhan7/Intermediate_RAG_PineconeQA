

import re
import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


class InvalidPDFError(Exception):
    """Raised when a file isn't a valid/usable PDF."""
    pass


def validate_pdf_bytes(file_bytes: bytes, max_size_mb: int = 20) -> None:
    if not file_bytes:
        raise InvalidPDFError("The file is empty.")

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise InvalidPDFError(
            f"File is {size_mb:.1f} MB, exceeds the {max_size_mb} MB limit."
        )

    if file_bytes[:4] != b"%PDF":
        raise InvalidPDFError("This file is not a valid PDF.")


def clean_text(raw_text: str) -> str:
    text = raw_text.replace("\x00", " ")
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def load_pdf(
    file_bytes: bytes,
    doc_name: str,
    max_size_mb: int = 20,
) -> list[Document]:

    # Step 1 — validate before doing anything else
    validate_pdf_bytes(file_bytes, max_size_mb)

    # Step 2 — write bytes to a temp file (PyPDFLoader needs a file path)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    # Step 3 — load with LangChain
    try:
        loader = PyPDFLoader(tmp_path)
        raw_docs = loader.load()
    except Exception as exc:
        raise InvalidPDFError(f"Could not parse PDF: {exc}") from exc
    finally:
        os.unlink(tmp_path)  # always delete temp file, even if loading fails

    if not raw_docs:
        raise InvalidPDFError("This PDF has no pages.")

    # Step 4 — clean text and fix metadata on each page
    docs: list[Document] = []
    for doc in raw_docs:
        cleaned = clean_text(doc.page_content)
        if cleaned:
            docs.append(Document(
                page_content=cleaned,
                metadata={
                    "page_number": doc.metadata.get("page", 0) + 1,
                    "doc_name": doc_name,
                    "source": doc_name,
                }
            ))

    if not docs:
        raise InvalidPDFError(
            "No extractable text found — this PDF may be scanned images only."
        )

    return docs