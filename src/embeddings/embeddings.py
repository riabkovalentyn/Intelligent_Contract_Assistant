from __future__ import annotations
from typing import Any, List, Dict
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger('embeddings')

def get_embeddings(texts: List[str], model: str = config.embedding_model) -> List[List[float]]:
    provider = config.embedding_provider
    if provider == "openai" and config.openai_api_key:
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model="text-embedding-3-small", api_key=config.openai_api_key)
        except Exception as e:
            logger.error("OpenAIEmbeddings failed (%s). Falling back to FakeEmbeddings.", e)
            return []
    logger.warning(f"Unsupported embedding provider: {provider}")


    from langchain_community.embeddings import FakeEmbeddings
    logger.info("Using FakeEmbeddings (local). Set OPENAI_API_KEY to use OpenAIEmbeddings.")
    return FakeEmbeddings(size=1536)