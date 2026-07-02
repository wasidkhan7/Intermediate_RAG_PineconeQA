
from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings


# @lru_cache on _load_model
# Loading a sentence-transformer model downloads weights and takes 2–5 seconds. If you call Embedder() multiple times (which Streamlit does on every rerender), without cache you'd reload the model every time. lru_cache(maxsize=1) keeps exactly one loaded model in memory and returns it instantly on subsequent calls.
@lru_cache(maxsize=1)
def _load_model(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 32,
        },
    )


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = _load_model(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self.model.embed_documents(texts)

    def embed_query(self, query: str) -> list[float]:
        return self.model.embed_query(query)