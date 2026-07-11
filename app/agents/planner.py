from typing import Any

from app.llm import generate_answer

from .base import BaseAgent


class PlannerAgent(BaseAgent):
    name = "planner"

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        query = state.get("query", "")
        conversation_history = state.get("conversation_history", "")

        prompt = (
            f"Analyze the following user query and create a step-by-step plan to answer it.\n\n"
            f"Conversation history:\n{conversation_history}\n\n"
            f"User query: {query}\n\n"
            f"Break down what needs to be done. Consider:\n"
            f"1. What information is needed?\n"
            f"2. What search terms should be used?\n"
            f"3. What reasoning steps are required?\n"
            f"4. How should the final answer be structured?\n\n"
            f"Return a concise plan as bullet points."
        )

        plan = generate_answer(
            "You are a planning agent that creates execution plans for complex queries.",
            prompt,
            max_tokens=512,
        )

        state["plan"] = plan
        state["current_step"] = "research"
        self.log(f"Plan created: {plan[:100]}...")
        return state
