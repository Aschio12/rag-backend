from collections.abc import Generator
from typing import Any

from openai import OpenAI

from app.config import settings

_client: OpenAI | None = None


def _is_mock_mode() -> bool:
    return not settings.openai_api_key or settings.openai_api_key.strip() == ""


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def _build_messages(system: str, prompt: str, image_data: str | None = None, image_type: str = "image/png") -> list[dict]:
    if image_data:
        return [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image_type};base64,{image_data}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]


_MOCK_RESPONSES: list[dict[str, Any]] = [
    {
        "keywords": ["planning agent"],
        "response": (
            "• Search for relevant information about the topic\n"
            "• Retrieve and filter the most relevant passages\n"
            "• Synthesize findings into a coherent answer\n"
            "• Review answer for accuracy and completeness\n"
            "• Verify key claims against sources"
        ),
    },
    {
        "keywords": ["research agent"],
        "response": (
            "1. What is the topic and its key concepts\n"
            "2. Main characteristics and applications\n"
            "3. Recent developments and examples"
        ),
    },
    {
        "keywords": ["retrieval agent"],
        "response": (
            "1, 2, 3 - These passages contain the most relevant information "
            "directly addressing the user's query with specific details."
        ),
    },
    {
        "keywords": ["reasoning agent"],
        "response": (
            "Based on the available information, here is a comprehensive answer:\n\n"
            "The topic refers to a concept that has several key aspects. "
            "First, it involves understanding the fundamental principles [Source: doc1, Page 1]. "
            "Second, practical applications demonstrate its importance in real-world scenarios [Source: doc2, Page 3]. "
            "Finally, ongoing developments continue to expand our understanding of this area.\n\n"
            "In summary, this is a multifaceted subject with significant relevance to the field."
        ),
    },
    {
        "keywords": ["critic agent"],
        "response": (
            "The answer covers the main points accurately. However, consider:\n"
            "1. Completeness - Could add more specific examples\n"
            "2. Clarity - Structure could be improved with clearer section breaks\n"
            "3. Citations - Ensure all claims have source references\n\n"
            "Overall, a solid draft that needs minor refinements."
        ),
    },
    {
        "keywords": ["answer improvement specialist"],
        "response": (
            "Here is an improved version of the answer:\n\n"
            "The topic encompasses several important dimensions: "
            "fundamental principles, practical applications, and ongoing developments. "
            "The core concept centers on understanding how these elements interact [Source: doc1, Page 1]. "
            "Real-world implementations demonstrate significant impact across multiple domains [Source: doc2, Page 3]. "
            "Current research continues to advance our knowledge and capabilities.\n\n"
            "This refined answer provides a more comprehensive view with better structure and clarity."
        ),
    },
    {
        "keywords": ["extracts claims"],
        "response": (
            "Claim 1: The concept has fundamental principles that define its scope\n"
            "Claim 2: Practical applications exist in real-world scenarios\n"
            "Claim 3: Ongoing developments continue to advance understanding"
        ),
    },
    {
        "keywords": ["checks factual claims"],
        "response": "supported - The claim is consistent with the retrieved evidence and can be verified from the available sources.",
    },
]


def _mock_answer(system_prompt: str, prompt: str) -> str:
    for entry in _MOCK_RESPONSES:
        if any(kw in system_prompt.lower() for kw in entry["keywords"]):
            return entry["response"]
    return (
        "This is a mock response for testing purposes. "
        "The system is working correctly through the agent pipeline."
    )


def generate_answer(
    system_prompt: str,
    prompt: str,
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    image_data: str | None = None,
    image_type: str = "image/png",
) -> str:
    if _is_mock_mode():
        return _mock_answer(system_prompt, prompt)
    response = _get_client().chat.completions.create(
        model=model or settings.openai_model,
        messages=_build_messages(system_prompt, prompt, image_data=image_data, image_type=image_type),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def generate_answer_stream(
    system_prompt: str,
    prompt: str,
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> Generator[str, None, None]:
    if _is_mock_mode():
        answer = _mock_answer(system_prompt, prompt)
        for i in range(0, len(answer), 3):
            yield answer[i:i + 3]
        return
    stream = _get_client().chat.completions.create(
        model=model or settings.openai_model,
        messages=_build_messages(system_prompt, prompt),
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content
