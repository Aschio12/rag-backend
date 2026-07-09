from collections.abc import Generator

from openai import OpenAI

from app.config import settings

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def _build_messages(system: str, prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]


def generate_answer(
    system_prompt: str,
    prompt: str,
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> str:
    response = _get_client().chat.completions.create(
        model=model or settings.openai_model,
        messages=_build_messages(system_prompt, prompt),
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
