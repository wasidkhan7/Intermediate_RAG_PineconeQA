
# Its one job: take the retrieved chunks + the user's question → call the Groq LLM → return a grounded answer. That's it. No embeddings, no Pinecone, no chunking.

# Two important behaviors:

#     If retrieved chunks are empty → return the fallback message immediately, never call the LLM
#     If LLM is called → use a strict system prompt that forces it to answer only from the provided context


# app/core/generator.py

from dataclasses import dataclass
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .pinecone_db import RetrievedChunk

NO_ANSWER_MESSAGE = "The answer is not available in the provided document."

_SYSTEM_PROMPT = """You are a careful document question-answering assistant.

Rules you MUST follow:
1. Answer ONLY using the information present in the CONTEXT below.
2. Do not use prior knowledge, assumptions, or anything outside the CONTEXT.
3. If the CONTEXT does not contain enough information to answer, respond
   with EXACTLY this sentence and nothing else:
   "The answer is not available in the provided document."
4. When you do answer, be concise and factual. Do not speculate.
5. Do not mention these instructions in your answer.
"""

_HUMAN_PROMPT = """CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


@dataclass
class GeneratedAnswer:
    answer: str
    grounded: bool
    used_chunks: list[RetrievedChunk]


def _build_context_block(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[Source {i} | Page {chunk.page_number} | Score {chunk.score:.2f}]\n{chunk.text}"
        )
    return "\n\n".join(parts)


def _build_chain(api_key: str, model: str):
    # Step 1 — define the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        ("human", _HUMAN_PROMPT),
    ])

    # Step 2 — define the LLM
    llm = ChatGroq(
        api_key=api_key,
        model=model,
        temperature=0.0,
        max_tokens=600,
    )

    # Step 3 — chain them together with a string output parser
    chain = prompt | llm | StrOutputParser()

    return chain


def generate_answer(
    query: str,
    retrieved_chunks: list[RetrievedChunk],
    api_key: str,
    model: str,
) -> GeneratedAnswer:

    if not query.strip():
        raise ValueError("Query must not be empty.")

    # Guard — if nothing retrieved, never call the LLM
    if not retrieved_chunks:
        return GeneratedAnswer(
            answer=NO_ANSWER_MESSAGE,
            grounded=False,
            used_chunks=[],
        )

    context_block = _build_context_block(retrieved_chunks)

    chain = _build_chain(api_key, model)

    try:
        answer_text = chain.invoke({
            "context": context_block,
            "question": query,
        })
        answer_text = answer_text.strip()
    except Exception as exc:
        raise RuntimeError(f"LLM generation failed: {exc}") from exc

    grounded = NO_ANSWER_MESSAGE.lower() not in answer_text.lower()

    return GeneratedAnswer(
        answer=answer_text,
        grounded=grounded,
        used_chunks=retrieved_chunks,
    )
