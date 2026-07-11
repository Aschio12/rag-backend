from collections.abc import Generator

from app.llm import generate_answer, generate_answer_stream
from app.prompts import RAG_SYSTEM_PROMPT, build_rag_prompt
from app.vector_store import search as vector_search


def answer(query: str, top_k: int = 5, hybrid: bool = False, rerank: bool = False) -> dict:
    results = vector_search(query, top_k=top_k, hybrid=hybrid)

    if rerank and results:
        results = _rerank_results(query, results)

    prompt = build_rag_prompt(query, results)

    answer_text = generate_answer(RAG_SYSTEM_PROMPT, prompt)

    return {
        "answer": answer_text,
        "sources": [
            {
                "doc_id": r.get("doc_id", ""),
                "text": r.get("text", ""),
                "score": r.get("score", 0.0),
                "page_number": r.get("page_number", 0),
                "id": r.get("id", ""),
            }
            for r in results
        ],
    }


def answer_stream(query: str, top_k: int = 5, hybrid: bool = False, rerank: bool = False) -> Generator[str, None, None]:
    results = vector_search(query, top_k=top_k, hybrid=hybrid)

    if rerank and results:
        results = _rerank_results(query, results)

    prompt = build_rag_prompt(query, results)
    yield from generate_answer_stream(RAG_SYSTEM_PROMPT, prompt)


def _rerank_results(query: str, results: list[dict]) -> list[dict]:
    """Rerank results using LLM relevance scoring (lightweight cross-encoder)."""
    scored = []
    for r in results:
        text = r.get("text", "")[:500]  # Truncate for speed
        prompt = (
            f"On a scale of 0.0 to 1.0, how relevant is this text to the query?\n"
            f"Query: {query}\n"
            f"Text: {text}\n"
            f"Score (just the number):"
        )
        try:
            score_text = generate_answer(
                "You are a relevance judge. Return only a float between 0 and 1.",
                prompt,
                max_tokens=10,
            )
            score = float(score_text.strip())
            score = max(0.0, min(1.0, score))
        except Exception:
            score = r.get("score", 0.0)
        scored.append({**r, "rerank_score": score})

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored
