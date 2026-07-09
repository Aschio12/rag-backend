from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "RAG Knowledge Chatbot"
    app_version: str = "0.1.0"
    debug: bool = False

    openai_api_key: str = ""
    openai_model: str = "gpt-3.5-turbo"

    chroma_persist_dir: str = "chroma_data"
    upload_dir: str = "uploads"
    chunk_size: int = 512
    chunk_overlap: int = 64

    class Config:
        env_file = ".env"


settings = Settings()
