from typing import Any

from app.llm import generate_answer
from app.vector_store import search as vector_search

from .base import BaseAgent


class VerificationAgent(BaseAgent):
    name = "verification"

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        query = state.get("query", "")
        draft = state.get("draft_answer", "")

        if not draft:
            state["current_step"] = "memory"
            return state

        # Extract key claims from the draft
        extract_prompt = (
            f"Extract the key factual claims from the following answer. "
            f"List each claim as a separate line.\n\n"
            f"Answer:\n{draft}\n\n"
            f"Claims:"
        )

        claims_text = generate_answer(
            "You are a verification agent that extracts claims to verify.",
            extract_prompt,
            max_tokens=256,
        )

        claims = [c.strip() for c in claims_text.split("\n") if c.strip() and len(c) > 10]
        if not claims:
            claims = [draft[:200]]

        # Verify each claim against the knowledge base
        verification_results = []
        for claim in claims[:3]:  # Limit to 3 claims for speed
            search_results = vector_search(claim, top_k=2, hybrid=True)
            supported = any(r.get("score", 0) > 0.5 for r in search_results)

            verify_prompt = (
                f"Verify this claim against the retrieved evidence and determine if it's supported.\n\n"
                f"Claim: {claim}\n\n"
                f"Evidence:\n"
                + "\n".join([f"- {r.get('text', '')[:200]} (score: {r.get('score', 0):.2f})" for r in search_results])
                + f"\n\nIs this claim supported? Answer 'supported', 'partially supported', or 'unsupported' with a brief explanation."
            )

            verdict = generate_answer(
                "You are a verification agent that checks factual claims.",
                verify_prompt,
                max_tokens=128,
            )

            verification_results.append({
                "claim": claim,
                "verdict": verdict,
                "supported": any(w in verdict.lower() for w in ["supported", "partially"]),
            })

        state["verification_results"] = verification_results
        state["current_step"] = "memory"
        self.log(f"Verified {len(verification_results)} claims from the answer")
        return state
