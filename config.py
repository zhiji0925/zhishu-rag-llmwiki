"""全局配置：从 .env / 环境变量读取，相对路径基于项目目录解析"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _get(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


class Settings:
    # embedding
    embedding_provider = _get("EMBEDDING_PROVIDER", "openai_compatible")  # openai_compatible | mock
    embedding_base_url = _get("EMBEDDING_BASE_URL")
    embedding_api_key = _get("EMBEDDING_API_KEY")
    embedding_model = _get("EMBEDDING_MODEL")
    embedding_dim = int(_get("EMBEDDING_DIM", "1024") or 1024)
    embedding_query_instruct = _get("EMBEDDING_QUERY_INSTRUCT").replace("\\n", "\n")
    # llm
    llm_base_url = _get("LLM_BASE_URL")
    llm_api_key = _get("LLM_API_KEY")
    llm_model = _get("LLM_MODEL")
    # 路径与默认值
    raw_dir = (BASE_DIR / _get("RAW_DIR", "../Raw")).resolve()
    lancedb_dir = (BASE_DIR / _get("LANCEDB_DIR", "./data/lancedb")).resolve()
    top_k = int(_get("TOP_K", "8") or 8)


SETTINGS = Settings()


def llm_ready() -> bool:
    return bool(SETTINGS.llm_base_url and SETTINGS.llm_api_key and SETTINGS.llm_model)
