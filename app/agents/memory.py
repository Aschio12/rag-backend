from typing import Any

from .base import BaseAgent


# In-memory conversation store
_conversations: dict[str, list[dict]] = {}


class MemoryAgent(BaseAgent):
    name = "memory"

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        conversation_id = state.get("conversation_id", "")
        query = state.get("query", "")
        draft = state.get("draft_answer", "")

        # Initialize conversation memory if new
        if conversation_id and conversation_id not in _conversations:
            _conversations[conversation_id] = []

        # Store the current exchange
        if conversation_id:
            _conversations[conversation_id].append({
                "role": "user",
                "content": query,
            })
            if draft:
                _conversations[conversation_id].append({
                    "role": "assistant",
                    "content": draft,
                })
            # Keep last 10 messages for context
            _conversations[conversation_id] = _conversations[conversation_id][-10:]

        # Build conversation history string for the response
        history = _conversations.get(conversation_id, [])
        history_str = "\n".join(
            [f"{m['role']}: {m['content'][:200]}" for m in history[-4:-2]]  # Last exchange before current
        )

        state["conversation_history"] = history_str
        state["current_step"] = "complete"
        state["final_answer"] = draft or "I couldn't generate an answer. Please try rephrasing your question."
        state["sources"] = state.get("research_results", [])[:5]

        self.log(f"Memory updated for conversation {conversation_id[:8] if conversation_id else 'none'}")

        # Summarize key information for the final answer display
        state["summary"] = {
            "plan_executed": bool(state.get("plan")),
            "searches_performed": len(state.get("search_queries", [])),
            "sources_found": len(state.get("research_results", [])),
            "critique_applied": state.get("was_critiqued", False),
            "claims_verified": len(state.get("verification_results", [])),
        }

        return state


def get_conversation_memory(conversation_id: str) -> list[dict]:
    return _conversations.get(conversation_id, [])


def clear_conversation_memory(conversation_id: str):
    _conversations.pop(conversation_id, None)
