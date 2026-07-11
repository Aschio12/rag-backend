"""Unit tests for individual agent modules (without LLM calls).
These tests verify state management and data flow logic.
"""
import pytest

from app.agents.memory import MemoryAgent, clear_conversation_memory


class TestMemoryAgentDetailed:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        clear_conversation_memory("unit_test_conv")
        yield
        clear_conversation_memory("unit_test_conv")

    @pytest.mark.asyncio
    async def test_conversation_history_building(self):
        agent = MemoryAgent()
        # First exchange
        state = {
            "query": "Hello",
            "conversation_id": "unit_test_conv",
            "draft_answer": "Hi there!",
            "research_results": [],
            "plan": "plan text",
            "search_queries": ["q1"],
            "was_critiqued": False,
            "verification_results": [],
        }
        result = await agent.run(state)
        assert result["conversation_history"] == ""

        # Second exchange
        state2 = {
            "query": "How are you?",
            "conversation_id": "unit_test_conv",
            "draft_answer": "I'm doing well, thanks!",
            "research_results": [],
            "plan": "plan text",
            "search_queries": ["q2"],
            "was_critiqued": False,
            "verification_results": [],
        }
        result2 = await agent.run(state2)
        # Should contain previous exchange
        assert "user: Hello" in result2["conversation_history"]
        assert "assistant: Hi there!" in result2["conversation_history"]

    @pytest.mark.asyncio
    async def test_sources_from_research_results(self):
        agent = MemoryAgent()
        state = {
            "query": "Test",
            "conversation_id": "unit_test_conv",
            "draft_answer": "Answer.",
            "research_results": [
                {"doc_id": "doc1", "text": "text1", "score": 0.9},
                {"doc_id": "doc2", "text": "text2", "score": 0.8},
            ],
            "plan": "",
            "search_queries": [],
            "was_critiqued": False,
            "verification_results": [],
        }
        result = await agent.run(state)
        assert len(result["sources"]) == 2
        assert result["sources"][0]["doc_id"] == "doc1"

    @pytest.mark.asyncio
    async def test_summary_empty_plan(self):
        agent = MemoryAgent()
        state = {
            "query": "Test",
            "conversation_id": "",
            "draft_answer": "Answer.",
            "research_results": [],
        }
        result = await agent.run(state)
        assert result["summary"]["plan_executed"] is False
        assert result["summary"]["searches_performed"] == 0
        assert result["summary"]["sources_found"] == 0


class TestOrchestratorState:
    def test_agent_names(self):
        from app.agents.orchestrator import AgentOrchestrator
        orch = AgentOrchestrator()
        expected_names = {
            "planning", "searching", "retrieving",
            "reasoning", "critiquing", "verifying", "memory",
        }
        assert set(orch.agents.keys()) == expected_names

    def test_step_labels_match_agents(self):
        from app.agents.orchestrator import AgentOrchestrator
        orch = AgentOrchestrator()
        step_map = {s[0]: s[1] for s in orch.steps}
        assert step_map["planning"] == "Planning"
        assert step_map["searching"] == "Searching"
        assert step_map["retrieving"] == "Retrieving"
        assert step_map["reasoning"] == "Reasoning"
        assert step_map["critiquing"] == "Critiquing"
        assert step_map["verifying"] == "Verifying"
        assert step_map["memory"] == "Generating response"

    def test_format_event(self):
        from app.agents.orchestrator import AgentOrchestrator
        orch = AgentOrchestrator()
        event_str = orch._format_event("step_start", {"step": "test", "label": "Test"})
        assert event_str.startswith("data: ")
        import json
        data = json.loads(event_str[6:])
        assert data["event"] == "step_start"
        assert data["step"] == "test"
