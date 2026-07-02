# Five operations it needs to handle:

# Create the index (if it doesn't exist yet)
# Upsert vectors + metadata into a namespace
# Query vectors by cosine similarity
# List existing namespaces
# Delete a namespace (for when a user removes a document) 
# app/core/pinecone_db.py

from dataclasses import dataclass
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import PineconeApiException
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


class VectorStoreConnectionError(Exception):
    """Raised when Pinecone cannot be reached or the index is unusable."""


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    page_number: int
    doc_name: str
    score: float


class PineconeDB:
    def __init__(
        self,
        api_key: str,
        index_name: str,
        dimension: int,
        embeddings: HuggingFaceEmbeddings,
        cloud: str = "aws",
        region: str = "us-east-1",
    ):
        if not api_key:
            raise VectorStoreConnectionError("Pinecone API key is missing.")

        self.index_name = index_name
        self.dimension = dimension
        self.embeddings = embeddings

        try:
            self.pc = Pinecone(api_key=api_key)
            self._ensure_index(cloud, region)
            self.index = self.pc.Index(index_name)
        except PineconeApiException as exc:
            raise VectorStoreConnectionError(f"Pinecone connection failed: {exc}") from exc
        except Exception as exc:
            raise VectorStoreConnectionError(f"Pinecone connection failed: {exc}") from exc

    # ------------------------------------------------------------------ #
    #  Index management
    # ------------------------------------------------------------------ #

    def _ensure_index(self, cloud: str, region: str) -> None:
        existing = [i["name"] for i in self.pc.list_indexes()]
        if self.index_name not in existing:
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )

    @staticmethod
    def namespace_for(doc_name: str) -> str:
        safe = "".join(c if c.isalnum() else "_" for c in doc_name).lower()
        return f"doc_{safe}"[:80]

    # ------------------------------------------------------------------ #
    #  Write
    # ------------------------------------------------------------------ #

    def upsert_chunks(
        self,
        namespace: str,
        chunks: list[Document],
    ) -> int:
        if not chunks:
            return 0

        try:
            vector_store = PineconeVectorStore(
                index=self.index,
                embedding=self.embeddings,
                namespace=namespace,
            )

            # LangChain uses chunk_id from metadata as the Pinecone vector ID
            chunk_ids = [
                chunk.metadata.get("chunk_id", str(i))
                for i, chunk in enumerate(chunks)
            ]

            vector_store.add_documents(documents=chunks, ids=chunk_ids)

        except Exception as exc:
            raise VectorStoreConnectionError(f"Upsert failed: {exc}") from exc

        return len(chunks)

    # ------------------------------------------------------------------ #
    #  Read
    # ------------------------------------------------------------------ #

    def query(
        self,
        namespace: str,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        page_filter: int | None = None,
    ) -> list[RetrievedChunk]:

        metadata_filter = (
            {"page_number": {"$eq": page_filter}} if page_filter else None
        )

        try:
            vector_store = PineconeVectorStore(
                index=self.index,
                embedding=self.embeddings,
                namespace=namespace,
            )

            results = vector_store.similarity_search_with_score(
                query=query,
                k=top_k,
                filter=metadata_filter,
            )

        except Exception as exc:
            raise VectorStoreConnectionError(f"Query failed: {exc}") from exc

        matches: list[RetrievedChunk] = []
        for doc, score in results:
            if score < similarity_threshold:
                continue
            matches.append(
                RetrievedChunk(
                    chunk_id=doc.metadata.get("chunk_id", ""),
                    text=doc.page_content,
                    page_number=doc.metadata.get("page_number", -1),
                    doc_name=doc.metadata.get("doc_name", "unknown"),
                    score=score,
                )
            )

        return matches

    # ------------------------------------------------------------------ #
    #  Namespace management
    # ------------------------------------------------------------------ #

    def list_namespaces(self) -> list[str]:
        try:
            stats = self.index.describe_index_stats()
            return list(stats.get("namespaces", {}).keys())
        except Exception as exc:
            raise VectorStoreConnectionError(
                f"Could not fetch index stats: {exc}"
            ) from exc

    def delete_namespace(self, namespace: str) -> None:
        try:
            self.index.delete(namespace=namespace, delete_all=True)
        except Exception as exc:
            raise VectorStoreConnectionError(
                f"Could not delete namespace: {exc}"
            ) from exc