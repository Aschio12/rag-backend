from fastapi import APIRouter

router = APIRouter(prefix="/api/v1")


@router.get("/")
async def api_root():
    return {"message": "RAG API v1"}
