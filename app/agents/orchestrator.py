import json
import time
from collections.abc import AsyncGenerator
from typing import Any

from app.agents.critic import CriticAgent
from app.agents.memory import MemoryAgent
from app.agents.planner import PlannerAgent
from app.agents.reasoning import ReasoningAgent
from app.agents.research import ResearchAgent
from app.agents.retrieval import RetrievalAgent
from app.agents.verification import VerificationAgent


class AgentOrchestrator:
    """Runs the agent pipeline and yields SSE events for each step."""

    def __init__(self):
        self.agents = {
            "planning": PlannerAgent(),
            "searching": ResearchAgent(),
            "retrieving": RetrievalAgent(),
            "reasoning": ReasoningAgent(),
            "critiquing": CriticAgent(),
            "verifying": VerificationAgent(),
            "memory": MemoryAgent(),
        }
        self.steps = [
            ("planning", "Planning", "Analyzing query and creating execution plan..."),
            ("searching", "Searching", "Searching knowledge base for relevant information..."),
            ("retrieving", "Retrieving", "Retrieving and selecting relevant passages..."),
            ("reasoning", "Reasoning", "Reasoning over retrieved information..."),
            ("critiquing", "Critiquing", "Evaluating answer quality..."),
            ("verifying", "Verifying", "Verifying claims against sources..."),
            ("memory", "Generating response", "Finalizing response..."),
        ]

    async def run(
        self,
        query: str,
        conversation_id: str = "",
        hybrid: bool = False,
    ) -> AsyncGenerator[str, None]:
        state: dict[str, Any] = {
            "query": query,
            "conversation_id": conversation_id,
            "hybrid": hybrid,
            "conversation_history": "",
        }

        start_time = time.time()

        for agent_key, step_label, step_desc in self.steps:
            agent = self.agents[agent_key]

            # Emit step start
            yield self._format_event("step_start", {
                "step": agent_key,
                "label": step_label,
                "description": step_desc,
            })

            try:
                state = await agent.run(state)
                elapsed = time.time() - start_time
                yield self._format_event("step_complete", {
                    "step": agent_key,
                    "label": step_label,
                    "duration": round(elapsed, 2),
                })
            except Exception as e:
                yield self._format_event("step_error", {
                    "step": agent_key,
                    "label": step_label,
                    "error": str(e),
                })

        # Emit final answer
        total_time = time.time() - start_time
        yield self._format_event("complete", {
            "answer": state.get("final_answer", state.get("draft_answer", "")),
            "sources": state.get("sources", []),
            "summary": state.get("summary", {}),
            "total_time": round(total_time, 2),
            "plan": state.get("plan", ""),
            "critique": state.get("critique", ""),
            "verification": state.get("verification_results", []),
            "search_queries": state.get("search_queries", []),
        })

    def _format_event(self, event: str, data: dict) -> str:
        return f"data: {json.dumps({'event': event, **data})}\n\n"
