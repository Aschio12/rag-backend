RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Follow these rules:
1. Answer using ONLY the information from the provided context.
2. If the context does not contain the answer, say "I cannot find this information in the provided documents."
3. Cite the source document names when referencing information.
4. Be concise and direct.
5. Do not make up information."""


def build_rag_prompt(query: str, context_chunks: list[dict]) -> str:
    context_lines = []
    for i, chunk in enumerate(context_chunks):
        source = chunk.get("doc_id", "unknown")[:8]
        context_lines.append(f"[Source: {source}] {chunk['text']}")

    context_str = "\n\n".join(context_lines)

    return f"""Context:
{context_str}

Question: {query}

Answer based on the context above:"""
