from typing import Any

from app.llm import generate_answer
from app.vector_store import search as vector_search

from .base import BaseAgent


class ResearchAgent(BaseAgent):
    name = "research"

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        query = state.get("query", "")
        plan = state.get("plan", "")
        hybrid = state.get("hybrid", False)

        # Generate optimized search queries from the plan
        prompt = (
            f"Based on the user query and execution plan, generate 3-5 specific search queries "
            f"that will retrieve the most relevant information from a knowledge base.\n\n"
            f"User query: {query}\n"
            f"Plan: {plan}\n\n"
            f"Return each search query on a new line, numbered."
        )

        search_queries_text = generate_answer(
            "You are a research agent that generates optimal search queries.",
            prompt,
            max_tokens=256,
        )

        search_queries = [
            q.strip().split(". ", 1)[1] if ". " in q else q.strip()
            for q in search_queries_text.split("\n")
            if q.strip() and any(c.isalpha() for c in q)
        ]

        if not search_queries:
            search_queries = [query]

        # Execute searches
        all_results = []
        seen_texts = set()
        for sq in search_queries:
            results = vector_search(sq, top_k=3, hybrid=hybrid)
            for r in results:
                text = r.get("text", "")
                if text and text not in seen_texts:
                    seen_texts.add(text)
                    all_results.append(r)

        # Sort by score descending, take top results
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        state["research_results"] = all_results[:10]
        state["search_queries"] = search_queries
        state["current_step"] = "retrieval"
        self.log(f"Found {len(all_results)} unique results from {len(search_queries)} search queries")
        return state
