from typing import Any

from app.llm import generate_answer

from .base import BaseAgent


class CriticAgent(BaseAgent):
    name = "critic"

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        query = state.get("query", "")
        draft = state.get("draft_answer", "")
        context = state.get("retrieved_context", "")

        if not draft:
            state["current_step"] = "verification"
            return state

        critic_prompt = (
            f"Review the following draft answer to a user query.\n\n"
            f"User query: {query}\n\n"
            f"Draft answer:\n{draft}\n\n"
            f"Retrieved context:\n{context}\n\n"
            f"Evaluate the draft on:\n"
            f"1. Accuracy - Does it correctly use the retrieved information?\n"
            f"2. Completeness - Does it fully address the query?\n"
            f"3. Clarity - Is it well-structured and easy to understand?\n"
            f"4. Citations - Are sources properly cited?\n\n"
            f"Provide specific feedback and suggestions for improvement."
        )

        critique = generate_answer(
            "You are a critic agent that evaluates answer quality and provides constructive feedback.",
            critic_prompt,
            max_tokens=512,
        )

        # Generate improved answer based on critique
        if any(word in critique.lower() for word in ["improve", "missing", "incorrect", "unclear", "add"]):
            improve_prompt = (
                f"Based on the following critique, improve the draft answer.\n\n"
                f"User query: {query}\n\n"
                f"Original draft:\n{draft}\n\n"
                f"Critique:\n{critique}\n\n"
                f"Please provide an improved version that addresses the critique."
            )

            improved = generate_answer(
                "You are an answer improvement specialist.",
                improve_prompt,
                max_tokens=1024,
            )
            state["draft_answer"] = improved
            state["was_critiqued"] = True
            self.log("Critique applied, answer improved")
        else:
            state["was_critiqued"] = False
            self.log("No critique needed, answer looks good")

        state["critique"] = critique
        state["current_step"] = "verification"
        return state
