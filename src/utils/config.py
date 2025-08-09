from __future__ import annotations
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv(override=True)

@dataclass
class AppConfig:
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    vector_store: str = field(default_factory=lambda: os.getenv("VECTOR_STORE", "chroma"))
    data_dir: str = field(default_factory=lambda: os.getenv("DATA_DIR", "./data"))
    vector_dir: str = field(default_factory=lambda: os.getenv("VECTOR_DIR", "./data/vector_store"))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))

    def ensure_directories(self):
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.vector_dir, exist_ok=True)

config = AppConfig()
config.ensure_directories()