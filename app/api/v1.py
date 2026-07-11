import json
import logging
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from app.chunker import chunk_text
from app.config import settings
from app.database import (
    add_document_meta,
    create_collection,
    create_kb,
    delete_collection,
    delete_document_meta,
    delete_kb,
    get_collection,
    get_document_meta,
    get_kb,
    list_collections,
    list_documents_meta,
    list_kbs,
    update_document_status,
    update_kb,
)
from app.embeddings import embed_text
from app.ingestion import ingest_document
from app.rag import answer, answer_stream
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.document import DocumentResponse, DocumentUploadResponse
from app.schemas.search import SearchRequest, SearchResult, SearchResponse
from app.store import _documents as old_documents
from app.vector_store import (
    _collection as chroma_collection,
    add_chunks as add_chunks_to_vector,
    search as vector_search,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ---- Knowledge Base Routes ----
@router.get("/knowledge-bases")
async def api_list_kbs():
    return list_kbs()


@router.post("/knowledge-bases")
async def api_create_kb(name: str = Form(...), description: str = Form("")):
    kb = create_kb(name, description)
    return kb


@router.get("/knowledge-bases/{kb_id}")
async def api_get_kb(kb_id: str):
    kb = get_kb(kb_id)
    if not kb:
        raise HTTPException(404, "Knowledge base not found")
    return kb


@router.patch("/knowledge-bases/{kb_id}")
async def api_update_kb(kb_id: str, name: str = Form(None), description: str = Form(None)):
    kb = update_kb(kb_id, name=name, description=description)
    if not kb:
        raise HTTPException(404, "Knowledge base not found")
    return kb


@router.delete("/knowledge-bases/{kb_id}")
async def api_delete_kb(kb_id: str):
    if not delete_kb(kb_id):
        raise HTTPException(404, "Knowledge base not found")
    return {"ok": True}


# ---- Collection Routes ----
@router.get("/collections")
async def api_list_collections(kb_id: str = Query(None)):
    return list_collections(kb_id)


@router.post("/collections")
async def api_create_collection(kb_id: str = Form(...), name: str = Form(...), description: str = Form("")):
    col = create_collection(kb_id, name, description)
    return col


@router.get("/collections/{col_id}")
async def api_get_collection(col_id: str):
    col = get_collection(col_id)
    if not col:
        raise HTTPException(404, "Collection not found")
    return col


@router.delete("/collections/{col_id}")
async def api_delete_collection(col_id: str):
    if not delete_collection(col_id):
        raise HTTPException(404, "Collection not found")
    return {"ok": True}


# ---- Document Routes ----
@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    collection_id: str = Form("default"),
    kb_id: str = Form("default"),
):
    doc_id = str(uuid.uuid4())
    filename = file.filename or "unknown"
    file_ext = Path(filename).suffix.lower()
    file_path = os.path.join(settings.upload_dir, f"{doc_id}{file_ext}")

    os.makedirs(settings.upload_dir, exist_ok=True)
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    doc_meta = add_document_meta(
        doc_id=doc_id,
        collection_id=collection_id,
        filename=filename,
        file_path=file_path,
        file_type=file_ext,
        status="processing",
    )

    try:
        result = ingest_document(file_path)
        chunk_count = len(result.get("chunks", []))
        update_document_status(doc_id, "indexed", chunk_count=chunk_count)
        return {
            "id": doc_id,
            "filename": filename,
            "chunk_count": chunk_count,
            "status": "indexed",
        }
    except Exception as e:
        update_document_status(doc_id, "error", error=str(e))
        raise HTTPException(500, f"Failed to ingest document: {e}")


@router.get("/documents")
async def list_documents(collection_id: str = Query(None)):
    return list_documents_meta(collection_id)


@router.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    doc = get_document_meta(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    return doc


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    doc = get_document_meta(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")

    # Remove from ChromaDB
    try:
        chroma_collection.delete(where={"doc_id": doc_id})
    except Exception as e:
        logger.warning(f"Chroma delete error: {e}")

    # Remove file
    try:
        os.remove(doc["file_path"])
    except Exception:
        pass

    delete_document_meta(doc_id)

    # Also remove from old in-memory store
    old_documents.pop(doc_id, None)

    return {"ok": True}


# ---- Legacy routes (backward compatible) ----
@router.post("/upload")
async def legacy_upload(file: UploadFile = File(...)):
    """Backward-compatible upload without KB/Collection."""
    return await upload_document(file)


@router.get("/old-documents")
async def legacy_list_documents():
    """Return old in-memory format for backward compat."""
    docs = list_documents_meta()
    return [
        {
            "id": d["id"],
            "filename": d["filename"],
            "chunk_count": d["chunk_count"],
            "status": d["status"],
        }
        for d in docs
    ]


# ---- Search & Chat (existing, updated) ----
@router.post("/search")
async def search_documents(req: SearchRequest):
    from app.database import list_documents_meta

    results = vector_search(req.query, top_k=req.top_k or 5, hybrid=req.hybrid)
    # If a collection_id is specified, filter results to only those documents in that collection
    if req.collection_id:
        collection_docs = list_documents_meta(collection_id=req.collection_id)
        valid_doc_ids = {d["id"] for d in collection_docs}
        results = [r for r in results if r.get("doc_id") in valid_doc_ids]

    return SearchResponse(
        query=req.query,
        results=[
            SearchResult(
                id=r["id"],
                text=r["text"],
                score=r["score"],
                doc_id=r["doc_id"],
                page_number=r.get("page_number", 0),
                filename=r.get("filename", ""),
            )
            for r in results
        ],
        total=len(results),
    )


@router.post("/chat")
async def chat(req: ChatRequest):
    result = answer(req.message, top_k=req.top_k or 5)
    return ChatResponse(
        answer=result["answer"],
        sources=[
            {
                "doc_id": s.get("doc_id", ""),
                "text": s.get("text", ""),
                "score": s.get("score", 0.0),
                "page_number": s.get("page_number", 0),
            }
            for s in result.get("sources", [])
        ],
        conversation_id=req.conversation_id or "",
    )


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    from fastapi.responses import StreamingResponse

    async def generate():
        async for chunk in answer_stream(req.message, top_k=req.top_k or 5):
            yield f"data: {json.dumps({'content': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ---- Preview ----
@router.get("/documents/{doc_id}/preview")
async def document_preview(doc_id: str, page: int = Query(0)):
    """Generate a page image preview for PDF documents."""
    doc = get_document_meta(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")

    file_path = doc["file_path"]
    if not file_path.lower().endswith(".pdf"):
        raise HTTPException(400, "Preview only available for PDF documents")

    try:
        from pdf2image import convert_from_path
        images = convert_from_path(file_path, first_page=page + 1, last_page=page + 1)
        if not images:
            raise HTTPException(404, "Page not found")

        from io import BytesIO
        buf = BytesIO()
        images[0].save(buf, format="PNG")
        buf.seek(0)
        from fastapi.responses import Response
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        raise HTTPException(500, f"Preview error: {e}")


# ---- OCR ----
@router.post("/ocr")
async def ocr_document(file: UploadFile = File(...)):
    """Extract text from an image using OCR."""
    try:
        import pytesseract
        from PIL import Image
        content = await file.read()
        from io import BytesIO
        img = Image.open(BytesIO(content))
        text = pytesseract.image_to_string(img)
        return {"text": text, "length": len(text)}
    except ImportError:
        raise HTTPException(501, "OCR not available (pytesseract not installed)")
    except Exception as e:
        raise HTTPException(500, f"OCR error: {e}")


# ---- Image Understanding ----
@router.post("/understand-image")
async def understand_image(file: UploadFile = File(...)):
    """Generate a description of an image using the LLM."""
    try:
        from PIL import Image
        from io import BytesIO
        content = await file.read()
        img = Image.open(BytesIO(content))

        # Simple approach: describe image via LLM with base64 encoding
        import base64
        from app.llm import generate_answer

        b64 = base64.b64encode(content).decode("utf-8")
        img_type = file.content_type or "image/png"
        prompt = "Describe this image in detail. What objects, text, and scenes do you see?"

        system = "You are an image analyst. Describe images accurately and concisely."
        response = generate_answer(system, prompt, image_data=b64, image_type=img_type)
        return {"description": response}
    except Exception as e:
        raise HTTPException(500, f"Image understanding error: {e}")


# ---- Query Expansion ----
@router.post("/query-expand")
async def query_expand(query: str = Form(...)):
    """Expand the query with related terms using LLM."""
    from app.llm import generate_answer
    prompt = (
        f"Given the user query: '{query}'\n\n"
        "Generate 5 alternative phrasings or expanded versions of this query "
        "to improve search retrieval. Return them as a comma-separated list."
    )
    result = generate_answer("You are a search query expansion expert.", prompt)
    expanded = [q.strip() for q in result.split(",") if q.strip()]
    return {"original": query, "expanded": expanded}


# ---- Query Rewriting ----
@router.post("/query-rewrite")
async def query_rewrite(query: str = Form(...)):
    """Rewrite the query for better search."""
    from app.llm import generate_answer
    prompt = (
        f"Rewrite the following user query to be more effective for a RAG search system. "
        f"Make it clear, specific, and well-structured for finding relevant information.\n\n"
        f"Original query: '{query}'\n\nRewritten query:"
    )
    result = generate_answer("You are a query rewriting specialist.", prompt)
    return {"original": query, "rewritten": result.strip()}


# ---- Query Decomposition ----
@router.post("/query-decompose")
async def query_decompose(query: str = Form(...)):
    """Decompose a complex query into simpler sub-queries."""
    from app.llm import generate_answer
    prompt = (
        f"Decompose the following complex query into 3-5 simpler sub-queries "
        f"that together cover the information needed to answer it.\n\n"
        f"Query: '{query}'\n\n"
        f"Return each sub-query on a new line, numbered."
    )
    result = generate_answer("You are a query decomposition expert.", prompt)
    sub_queries = [q.strip() for q in result.split("\n") if q.strip() and q[0].isdigit()]
    sub_queries = [q.split(". ", 1)[1] if ". " in q else q for q in sub_queries]
    return {"original": query, "sub_queries": sub_queries}


# ---- Reranking ----
@router.post("/rerank")
async def rerank_results(query: str = Form(...), results: str = Form("[]")):
    """Rerank search results using cross-encoder scoring via LLM."""
    import json as json_mod
    results_list = json_mod.loads(results)
    if not results_list:
        return {"results": []}

    from app.llm import generate_answer
    scored = []
    for r in results_list:
        text = r.get("text", "")
        prompt = (
            f"On a scale of 0.0 to 1.0, how relevant is the following text to the query?\n\n"
            f"Query: {query}\n\nText: {text}\n\n"
            f"Relevance score (just the number):"
        )
        try:
            score_text = generate_answer("You are a relevance judge.", prompt, max_tokens=10)
            score = float(score_text.strip())
            score = max(0.0, min(1.0, score))
        except Exception:
            score = r.get("score", 0.0)
        scored.append({**r, "rerank_score": score})

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return {"results": scored}


# ---- Knowledge Graph ----
@router.post("/knowledge-graph")
async def knowledge_graph(doc_id: str = Form(...)):
    """Extract entities and relationships from a document as a knowledge graph."""
    doc = get_document_meta(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")

    from app.llm import generate_answer

    # Get document text from chunks
    try:
        results = chroma_collection.get(where={"doc_id": doc_id})
        texts = results.get("documents", [])
        full_text = "\n".join(texts) if texts else ""
    except Exception:
        full_text = ""

    if not full_text:
        raise HTTPException(400, "No text content found in document")

    # Truncate for LLM context
    full_text = full_text[:8000]

    prompt = (
        f"Extract a knowledge graph from the following text. "
        f"Identify key entities and their relationships.\n\n"
        f"Text:\n{full_text}\n\n"
        f"Return the result as a JSON object with 'nodes' (array of {{'id': str, 'label': str, 'type': str}}) "
        f"and 'edges' (array of {{'source': str, 'target': str, 'label': str}}). "
        f"Include only the JSON, no other text."
    )

    try:
        result = generate_answer("You are a knowledge graph extraction expert.", prompt, max_tokens=2048)
        # Try to parse JSON from response
        import re
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if json_match:
            data = json_mod.loads(json_match.group())
            return data
        return {"nodes": [], "edges": []}
    except Exception as e:
        raise HTTPException(500, f"Knowledge graph extraction error: {e}")


# ---- Mind Map ----
@router.post("/mind-map")
async def mind_map(doc_id: str = Form(...)):
    """Generate a mind map from a document."""
    doc = get_document_meta(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")

    from app.llm import generate_answer

    try:
        results = chroma_collection.get(where={"doc_id": doc_id})
        texts = results.get("documents", [])
        full_text = "\n".join(texts) if texts else ""
    except Exception:
        full_text = ""

    if not full_text:
        raise HTTPException(400, "No text content found")

    full_text = full_text[:8000]

    prompt = (
        f"Create a mind map from the following text. "
        f"Identify the central topic, main branches, and sub-branches.\n\n"
        f"Text:\n{full_text}\n\n"
        f"Return as JSON: {{'central': 'topic', 'branches': [{{'name': 'branch', 'children': ['sub1', 'sub2']}}]}}"
    )

    try:
        result = generate_answer("You are a mind map generation expert.", prompt, max_tokens=2048)
        import re
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if json_match:
            data = json_mod.loads(json_match.group())
            return data
        return {"central": "Document", "branches": []}
    except Exception as e:
        raise HTTPException(500, f"Mind map error: {e}")


# ---- Summaries ----
@router.post("/summarize")
async def summarize_document(doc_id: str = Form(...)):
    """Generate an AI summary of a document."""
    doc = get_document_meta(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")

    from app.llm import generate_answer

    try:
        results = chroma_collection.get(where={"doc_id": doc_id})
        texts = results.get("documents", [])
        full_text = "\n".join(texts) if texts else ""
    except Exception:
        full_text = ""

    if not full_text:
        raise HTTPException(400, "No text content found")

    full_text = full_text[:8000]

    prompt = (
        f"Provide a comprehensive summary of the following document. "
        f"Include key points, main arguments, and conclusions.\n\n"
        f"Text:\n{full_text}"
    )
    summary = generate_answer("You are a document summarization expert.", prompt, max_tokens=1024)
    return {"doc_id": doc_id, "summary": summary}


# ---- Flashcards & Quiz ----
@router.post("/flashcards")
async def generate_flashcards(doc_id: str = Form(...), count: int = Form(5)):
    """Generate flashcards from document content."""
    doc = get_document_meta(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")

    from app.llm import generate_answer

    try:
        results = chroma_collection.get(where={"doc_id": doc_id})
        texts = results.get("documents", [])
        full_text = "\n".join(texts) if texts else ""
    except Exception:
        full_text = ""

    if not full_text:
        raise HTTPException(400, "No text content found")

    full_text = full_text[:8000]

    prompt = (
        f"Generate {count} flashcards from the following document. "
        f"A flashcard has a 'front' (question/term) and 'back' (answer/definition).\n\n"
        f"Text:\n{full_text}\n\n"
        f"Return as JSON array: [{{'front': '...', 'back': '...'}}]"
    )

    try:
        result = generate_answer("You are a flashcard generation expert.", prompt, max_tokens=2048)
        import re
        json_match = re.search(r'\[.*\]', result, re.DOTALL)
        if json_match:
            data = json_mod.loads(json_match.group())
            return {"flashcards": data[:count]}
        return {"flashcards": []}
    except Exception as e:
        raise HTTPException(500, f"Flashcard generation error: {e}")


@router.post("/quiz")
async def generate_quiz(doc_id: str = Form(...), count: int = Form(5)):
    """Generate quiz questions from document content."""
    doc = get_document_meta(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")

    from app.llm import generate_answer

    try:
        results = chroma_collection.get(where={"doc_id": doc_id})
        texts = results.get("documents", [])
        full_text = "\n".join(texts) if texts else ""
    except Exception:
        full_text = ""

    if not full_text:
        raise HTTPException(400, "No text content found")

    full_text = full_text[:8000]

    prompt = (
        f"Generate {count} multiple-choice quiz questions from the following document. "
        f"Each question should have 4 options with one correct answer.\n\n"
        f"Text:\n{full_text}\n\n"
        f"Return as JSON array: [{{'question': '...', 'options': ['A', 'B', 'C', 'D'], 'answer': 0}}]"
    )

    try:
        result = generate_answer("You are a quiz generation expert.", prompt, max_tokens=2048)
        import re
        json_match = re.search(r'\[.*\]', result, re.DOTALL)
        if json_match:
            data = json_mod.loads(json_match.group())
            return {"questions": data[:count]}
        return {"questions": []}
    except Exception as e:
        raise HTTPException(500, f"Quiz generation error: {e}")


# ---- Timeline ----
@router.post("/timeline")
async def extract_timeline(doc_id: str = Form(...)):
    """Extract a timeline of events from a document."""
    doc = get_document_meta(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")

    from app.llm import generate_answer

    try:
        results = chroma_collection.get(where={"doc_id": doc_id})
        texts = results.get("documents", [])
        full_text = "\n".join(texts) if texts else ""
    except Exception:
        full_text = ""

    if not full_text:
        raise HTTPException(400, "No text content found")

    full_text = full_text[:8000]

    prompt = (
        f"Extract a timeline of events from the following document. "
        f"Identify dates/periods and the events associated with them.\n\n"
        f"Text:\n{full_text}\n\n"
        f"Return as JSON array sorted by date: [{{'date': '...', 'event': '...', 'description': '...'}}]"
    )

    try:
        result = generate_answer("You are a timeline extraction expert.", prompt, max_tokens=2048)
        import re
        json_match = re.search(r'\[.*\]', result, re.DOTALL)
        if json_match:
            data = json_mod.loads(json_match.group())
            return {"events": data}
        return {"events": []}
    except Exception as e:
        raise HTTPException(500, f"Timeline extraction error: {e}")


# ---- Table Extraction ----
@router.post("/extract-tables")
async def extract_tables(doc_id: str = Form(...)):
    """Extract structured tables from document text."""
    doc = get_document_meta(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")

    try:
        results = chroma_collection.get(where={"doc_id": doc_id})
        texts = results.get("documents", [])
        full_text = "\n".join(texts) if texts else ""
    except Exception:
        full_text = ""

    if not full_text:
        raise HTTPException(400, "No text content found")

    # Use markdown tables already present, or use LLM to extract
    from app.llm import generate_answer

    full_text = full_text[:8000]
    prompt = (
        f"Extract any tabular data from the following document. "
        f"Identify rows and columns.\n\n"
        f"Text:\n{full_text}\n\n"
        f"Return as JSON array of tables: "
        f"[{{'headers': ['col1', 'col2'], 'rows': [['val1', 'val2']], 'caption': '...'}}]"
    )

    try:
        result = generate_answer("You are a table extraction expert.", prompt, max_tokens=2048)
        import re
        json_match = re.search(r'\[.*\]', result, re.DOTALL)
        if json_match:
            data = json_mod.loads(json_match.group())
            return {"tables": data}
        return {"tables": []}
    except Exception as e:
        raise HTTPException(500, f"Table extraction error: {e}")


# ---- Related Questions ----
@router.post("/related-questions")
async def related_questions(query: str = Form(...), answer_text: str = Form("")):
    """Generate related questions based on the query and answer."""
    from app.llm import generate_answer

    prompt = (
        f"Based on the following Q&A, suggest 5 related questions that the user might want to ask next.\n\n"
        f"User query: {query}\n"
        f"Answer: {answer_text}\n\n"
        f"Return as a comma-separated list of questions."
    )

    result = generate_answer("You are a question suggestion expert.", prompt)
    questions = [q.strip() for q in result.split(",") if q.strip()]
    return {"questions": questions[:5]}
