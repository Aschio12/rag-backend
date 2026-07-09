from app.llm import generate_answer
from app.prompts import RAG_SYSTEM_PROMPT, build_rag_prompt
from app.vector_store import search as vector_search


def answer(query: str, top_k: int = 5) -> dict:
    results = vector_search(query, top_k=top_k)
    prompt = build_rag_prompt(query, results)

    answer_text = generate_answer(prompt)

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
