"""Tests for the agent system modules."""
import pytest

from app.agents.memory import MemoryAgent, get_conversation_memory, clear_conversation_memory
from app.agents.orchestrator import AgentOrchestrator


class TestMemoryAgent:
    @pytest.fixture(autouse=True)
    def clean_memory(self):
        clear_conversation_memory("test_conv")
        yield
        clear_conversation_memory("test_conv")

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self):
        agent = MemoryAgent()
        state = {
            "query": "What is RAG?",
            "conversation_id": "test_conv",
            "draft_answer": "RAG stands for Retrieval Augmented Generation.",
            "research_results": [],
        }
        result = await agent.run(state)
        assert result["current_step"] == "complete"
        assert result["final_answer"] == "RAG stands for Retrieval Augmented Generation."

        memory = get_conversation_memory("test_conv")
        assert len(memory) == 2
        assert memory[0]["role"] == "user"
        assert memory[0]["content"] == "What is RAG?"
        assert memory[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_empty_conversation_id(self):
        agent = MemoryAgent()
        state = {
            "query": "Hello",
            "conversation_id": "",
            "draft_answer": "Hi there!",
            "research_results": [],
        }
        result = await agent.run(state)
        assert result["current_step"] == "complete"
        assert result["final_answer"] == "Hi there!"

    @pytest.mark.asyncio
    async def test_no_draft_answer(self):
        agent = MemoryAgent()
        state = {
            "query": "Test",
            "conversation_id": "test_conv",
            "draft_answer": "",
            "research_results": [],
        }
        result = await agent.run(state)
        assert "couldn't generate" in result["final_answer"]

    @pytest.mark.asyncio
    async def test_summary_stats(self):
        agent = MemoryAgent()
        state = {
            "query": "Test",
            "conversation_id": "test_conv",
            "draft_answer": "Answer.",
            "plan": "plan text",
            "search_queries": ["q1", "q2"],
            "research_results": [{"text": "r1"}, {"text": "r2"}, {"text": "r3"}],
            "was_critiqued": True,
            "verification_results": [{"claim": "c1"}],
        }
        result = await agent.run(state)
        assert result["summary"]["plan_executed"] is True
        assert result["summary"]["searches_performed"] == 2
        assert result["summary"]["sources_found"] == 3
        assert result["summary"]["critique_applied"] is True
        assert result["summary"]["claims_verified"] == 1


class TestOrchestrator:
    def test_agents_loaded(self):
        orch = AgentOrchestrator()
        assert len(orch.agents) == 7
        assert "planning" in orch.agents
        assert "searching" in orch.agents
        assert "retrieving" in orch.agents
        assert "reasoning" in orch.agents
        assert "critiquing" in orch.agents
        assert "verifying" in orch.agents
        assert "memory" in orch.agents

    def test_steps_order(self):
        orch = AgentOrchestrator()
        step_keys = [s[0] for s in orch.steps]
        assert step_keys == [
            "planning", "searching", "retrieving", "reasoning",
            "critiquing", "verifying", "memory",
        ]

    @pytest.mark.asyncio
    async def test_orchestrator_error_handling(self):
        """Test that orchestrator handles errors gracefully and returns a complete event."""
        orch = AgentOrchestrator()
        events = []
        async for event_str in orch.run(query="", conversation_id=""):
            import json
            data = json.loads(event_str[6:])
            events.append(data)

        # Should have at least start + error + complete
        assert len(events) >= 3
        assert events[-1]["event"] == "complete"
        assert "answer" in events[-1]

    @pytest.mark.asyncio
    async def test_orchestrator_streams_events(self):
        import os
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set, skipping integration test")

        orch = AgentOrchestrator()
        events = []
        async for event_str in orch.run(query="What is machine learning?", conversation_id="test_orch"):
            assert event_str.startswith("data: ")
            import json
            data = json.loads(event_str[6:])
            events.append(data)

        # Should have step_start, step_complete, step_error, or complete events
        assert len(events) >= 7  # At minimum one per agent

        # Last event should be 'complete'
        assert events[-1]["event"] == "complete"
        assert "answer" in events[-1]
