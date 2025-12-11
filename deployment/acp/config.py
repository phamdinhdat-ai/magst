import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from functools import lru_cache # For caching settings instance
from typing import Optional
# Load .env file from the project root if it exists

project_root_env = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')
if os.path.exists(project_root_env):
    load_dotenv(dotenv_path=project_root_env)
else:
    # Fallback if you might be running from within app/core or similar for tests
    load_dotenv()


class Settings(BaseSettings):
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "Intelligent Multi-Agent Chat System")
    API_V1_STR: str = os.getenv("API_V1_STR", "/api/v1")
    STORAGE_DIR: str = os.getenv("STORAGE_DIR", "./storage/company")
    MCP_SERVER_URL: str = os.getenv("MCP_SERVER_URL", "http://192.168.1.60:50051/sse")
    # Document upload settings
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "app/uploaded_files/documents")
    TEMP_UPLOAD_DIR: str = os.getenv("TEMP_UPLOAD_DIR", "app/uploaded_files/temp")
    MAX_UPLOAD_SIZE: int = int(os.getenv("MAX_UPLOAD_SIZE", 10 * 1024 * 1024))  # 10 MB default
    TEMP_FILE_EXPIRY: int = int(os.getenv("TEMP_FILE_EXPIRY", 24 * 60 * 60))  # 24 hours in seconds
    
    # CORS settings for frontend integration
    BACKEND_CORS_ORIGINS: list = [
        "http://localhost:3000",
        "http://localhost:8080", 
        "http://localhost:5000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:5000"
    ]
    
    # Database settings - use environment variables with fallbacks
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "datpd")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "datpd")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "gst_agents")
    POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "5432")
    
    # Use DATABASE_URL from environment if available, otherwise construct it
    POSTGRES_DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL", "postgresql+asyncpg://datpd:datpd@localhost:5432/gst_agents")
    # DATABASE_URL=postgresql+asyncpg://datpd:datpd@192.168.1.60:5432/gst_agents
    # LangGraph PostgresSaver (if used, or other persistent saver config)
    LANGGRAPH_POSTGRES_URL: Optional[str] = os.getenv("LANGGRAPH_POSTGRES_URL", None)

    # JWT Settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "good thing is comming!")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60 * 24))  # 24 hours
    REFRESH_TOKEN_EXPIRE_DAYS: int = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", 30))  # 30 days
    
    # Security settings
    PASSWORD_RESET_TOKEN_EXPIRE_HOURS: int = int(os.getenv("PASSWORD_RESET_TOKEN_EXPIRE_HOURS", 1))
    EMAIL_VERIFY_TOKEN_EXPIRE_HOURS: int = int(os.getenv("EMAIL_VERIFY_TOKEN_EXPIRE_HOURS", 24))
    MAX_LOGIN_ATTEMPTS: int = int(os.getenv("MAX_LOGIN_ATTEMPTS", 5))
    ACCOUNT_LOCKOUT_DURATION_MINUTES: int = int(os.getenv("ACCOUNT_LOCKOUT_DURATION_MINUTES", 30))
    
    # Session settings
    SESSION_EXPIRE_HOURS: int = int(os.getenv("SESSION_EXPIRE_HOURS", 24))
    MAX_SESSIONS_PER_CUSTOMER: int = int(os.getenv("MAX_SESSIONS_PER_CUSTOMER", 5))

    # LLM Settings
    # LLM_MODEL: str = 'qwen2.5:7b'  # Default model, can be overridden by env variable
    LLM_REASONING_MODEL: str = os.getenv("LLM_REASONING_MODEL_NAME", "Qwen/Qwen3-1.7B")
    LLM_MODEL: str = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen3-1.7B")
    API_KEY: str = os.getenv("API_KEY", "EMPTY")  # Default API key, can be overridden by env variable
    # LLM_REASONING_MODEL: str = os.getenv("LLM_REASONING_MODEL_NAME", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B") # user in  vast ai server
    # LLM_MODEL: str = os.getenv("LLM_MODEL_NAME", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B") # user in vast ai server
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL_NAME", "dengcao/Qwen3-Embedding-0.6B:Q8_0")

    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:7b")  # Default Ollama model, can be overridden by env variable
    OLLAMA_EMBEDDING_MODEL: str = os.getenv("OLLAMA_EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B")  # Default
    # LLM Provider settings
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")  # Options: "ollama", "vllm"
    REASONING_LLM_PROVIDER: str = os.getenv("REASONING_LLM_PROVIDER", "ollama")  # Options: "ollama", "vllm"
    
    # Ollama Settings
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # VLLM Settings
    # VLLM_API_URL: str = os.getenv("VLLM_API_URL", "http://localhost:6622")
    VLLM_API_URL: str = os.getenv("VLLM_API_URL", "http://142.182.153.12:40039") # public IP vast ai
    
    
    # HF Settings
    HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN", None)
    HF_EMBEDDING_MODEL: str = os.getenv("HF_EMBEDDING_MODEL", 
                                       os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                                   "hf_model", "Vietnamese_Embedding_v2"))
    HF_LLM_MODEL: str = os.getenv("HF_LLM_MODEL", "app/hf_model/Llama-2-7b-chat-hf")
    HF_RERANKER_MODEL: str = os.getenv("HF_RERANKER_MODEL", 
                                       os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                                   "hf_model", "Vietnamese_Reranker"))
    #Vector DB settings 
    
    COMPANY_DB: str = os.getenv("COMPANY_DB", "gst_vdb")
    GENETIC_DB: str = os.getenv("GENETIC_DB", "genetic_vdb")
    MEDICAL_DB: str = os.getenv("MEDICAL_DB", "medical_vdb")
    DRUGS_DB: str = os.getenv("DRUGS_DB", "drugs_vdb")
    PRODUCTS_DB: str = os.getenv("PRODUCTS_DB", "products_vdb")

    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 1024))
    OVERLAP_SIZE: int = int(os.getenv("OVERLAP_SIZE", 256))
    TOP_K_RETRIEVE: int = int(os.getenv("TOP_K_RETRIEVE", 15))
    TOP_K_RERANK: int = int(os.getenv("TOP_K_RERANK", 5))

    # ChromaDB Settings
    VECTOR_STORE_BASE_DIR: str = os.getenv("VECTOR_STORE_BASE_DIR", "./vector_stores_data")

    # File storage for uploaded documents
    UPLOADED_FILES_DIR: str = os.getenv("UPLOADED_FILES_DIR", "./uploaded_files")


    DATA_STORAGE_DIR: str = os.getenv("DATA_STORAGE_DIR", "app/data_storage")
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", 4))

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"

    def __init__(self, **values):
        super().__init__(**values)
        # If POSTGRES_DATABASE_URL is not set, construct it from components
        if not self.POSTGRES_DATABASE_URL:
            self.POSTGRES_DATABASE_URL = (
                f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
                f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
            )
        
        # Ensure upload directory exists
        os.makedirs(self.UPLOADED_FILES_DIR, exist_ok=True)
        os.makedirs(self.VECTOR_STORE_BASE_DIR, exist_ok=True)
        os.makedirs(self.STORAGE_DIR, exist_ok=True)
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)


@lru_cache() # Cache the settings object so it's only created once
def get_settings() -> Settings:
    return Settings()

settings = get_settings()

# Example of how to use it:
if __name__ == "__main__":
    print(f"Database URL: {settings.POSTGRES_DATABASE_URL}")
    print(f"LLM Model: {settings.LLM_MODEL}")
    print(f"Secret Key (first 5 chars): {settings.SECRET_KEY[:5]}...") # Don't print full secret key