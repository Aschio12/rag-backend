RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Follow these rules:
1. Answer using ONLY the information from the provided context.
2. If the context does not contain the answer, say "I cannot find this information in the provided documents."
3. Cite the source name and page number when referencing information, like [Source: filename, Page 3].
4. Include confidence indicators when speculating.
5. Be concise and direct.
6. Do not make up information."""


def build_rag_prompt(query: str, context_chunks: list[dict]) -> str:
    context_lines = []
    for i, chunk in enumerate(context_chunks):
        doc_id = chunk.get("doc_id", "unknown")[:8]
        page = chunk.get("page_number", 0)
        score = chunk.get("score", 0.0)
        page_str = f", Page {page}" if page else ""
        context_lines.append(
            f"[Source: {doc_id}{page_str} | Score: {score:.2f}] {chunk['text']}"
        )

    context_str = "\n\n".join(context_lines)

    return f"""Context:
{context_str}

Question: {query}

Answer based on the context above. Always cite sources with [Source: name, Page X] when using specific information:"""
