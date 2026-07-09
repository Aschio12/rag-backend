from collections.abc import Generator

from app.llm import generate_answer, generate_answer_stream
from app.prompts import RAG_SYSTEM_PROMPT, build_rag_prompt
from app.vector_store import search as vector_search


def answer(query: str, top_k: int = 5) -> dict:
    results = vector_search(query, top_k=top_k)
    prompt = build_rag_prompt(query, results)

    answer_text = generate_answer(RAG_SYSTEM_PROMPT, prompt)

    return {
        "answer": answer_text,
        "sources": [
            {
                "doc_id": r["doc_id"],
                "text": r["text"],
                "score": r["score"],
            }
            for r in results
        ],
    }


def answer_stream(query: str, top_k: int = 5) -> Generator[str, None, None]:
    results = vector_search(query, top_k=top_k)
    prompt = build_rag_prompt(query, results)
    yield from generate_answer_stream(RAG_SYSTEM_PROMPT, prompt)
