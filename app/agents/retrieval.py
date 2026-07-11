from typing import Any

from app.llm import generate_answer

from .base import BaseAgent


class RetrievalAgent(BaseAgent):
    name = "retrieval"

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        query = state.get("query", "")
        research_results = state.get("research_results", [])

        if not research_results:
            state["retrieved_context"] = ""
            state["current_step"] = "reasoning"
            self.log("No research results to retrieve from")
            return state

        # Format retrieved chunks
        formatted_chunks = []
        for i, r in enumerate(research_results):
            doc_id = r.get("doc_id", "unknown")[:8]
            page = r.get("page_number", 0)
            score = r.get("score", 0.0)
            text = r.get("text", "")
            page_str = f", Page {page}" if page else ""
            formatted_chunks.append(
                f"[{i + 1}] Source: {doc_id}{page_str} (relevance: {score:.2f})\n{text}"
            )

        context = "\n\n---\n\n".join(formatted_chunks)

        # Use LLM to select the most relevant passages
        selection_prompt = (
            f"Given the user query and the retrieved document passages below, "
            f"select the 3-5 most relevant passages that are needed to answer the query.\n\n"
            f"Query: {query}\n\n"
            f"Retrieved passages:\n{context}\n\n"
            f"Return the numbers of the most relevant passages (e.g., 1, 3, 5) and a brief note on why each is relevant."
        )

        selection = generate_answer(
            "You are a retrieval agent that selects the most relevant information.",
            selection_prompt,
            max_tokens=256,
        )

        state["retrieved_context"] = context
        state["retrieval_selection"] = selection
        state["current_step"] = "reasoning"
        self.log(f"Retrieved and selected relevant passages from {len(formatted_chunks)} chunks")
        return state
