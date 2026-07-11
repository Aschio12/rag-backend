from typing import Any

from app.llm import generate_answer

from .base import BaseAgent


class ReasoningAgent(BaseAgent):
    name = "reasoning"

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        query = state.get("query", "")
        context = state.get("retrieved_context", "")
        plan = state.get("plan", "")
        selection = state.get("retrieval_selection", "")

        if not context:
            state["draft_answer"] = (
                "I searched the knowledge base but couldn't find relevant information "
                "to answer your question. You may need to upload documents related to this topic."
            )
            state["current_step"] = "critic"
            self.log("No context available, using fallback response")
            return state

        reasoning_prompt = (
            f"User query: {query}\n\n"
            f"Execution plan:\n{plan}\n\n"
            f"Retrieved information:\n{context}\n\n"
            f"Relevance assessment:\n{selection}\n\n"
            f"Now, reason step by step to answer the query using the retrieved information.\n"
            f"1. First, identify the key facts needed\n"
            f"2. Then, analyze how they relate to the query\n"
            f"3. Finally, formulate a comprehensive answer\n\n"
            f"Show your reasoning process, then provide the final answer."
        )

        reasoning = generate_answer(
            "You are a reasoning agent that thinks step by step. "
            "Always cite sources with [Source: name, Page X] when using specific information.",
            reasoning_prompt,
            max_tokens=1024,
        )

        state["draft_answer"] = reasoning
        state["current_step"] = "critic"
        self.log("Reasoning completed and draft answer generated")
        return state
