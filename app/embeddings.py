import numpy as np
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

_embedding_function = ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])


def embed_text(text: str) -> list[float]:
    return _embedding_function([text])[0]


def embed_texts(texts: list[str]) -> list[list[float]]:
    return _embedding_function(texts)
