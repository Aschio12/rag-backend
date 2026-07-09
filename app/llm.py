from openai import OpenAI

from app.config import settings

_client = OpenAI(api_key=settings.openai_api_key)


def generate_answer(
    prompt: str,
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> str:
    response = _client.chat.completions.create(
        model=model or settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""
