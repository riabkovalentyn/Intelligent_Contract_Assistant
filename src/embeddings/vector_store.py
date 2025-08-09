from __future__ import annotations
import os 
from typing import List, Dict, Optional
from langchain.schema import Document
from src.utils.config import config
from src.utils.logger import get_logger
from src.embeddings.embeddings import get_embeddings

logger = get_logger('vector_store')

class VectorStoreManager:
    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or config.vector_dir
        self.store_type = config.vector_store  # "chroma" | "faiss"
        self.embeddings = get_embeddings()
        self.vs = None

    def build_from_documents(self, docs: List[Document], collection_name: str = "contracts"):
        if self.store_type == "chroma":
            from langchain_community.vectorstores import Chroma
            logger.info("Building Chroma index in %s", self.persist_dir)
            self.vs = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                collection_name=collection_name,
                persist_directory=self.persist_dir,
            )
            self.vs.persist()
        elif self.store_type == "faiss":
            from langchain_community.vectorstores import FAISS
            logger.info("Building FAISS index in %s", self.persist_dir)
            self.vs = FAISS.from_documents(documents=docs, embedding=self.embeddings)
            os.makedirs(self.persist_dir, exist_ok=True)
            self.vs.save_local(self.persist_dir)
        else:
            raise ValueError(f"Unsupported VECTOR_STORE: {self.store_type}")
        return self.vs

    def load(self, collection_name: str = "contracts"):
        if self.store_type == "chroma":
            from langchain_community.vectorstores import Chroma
            if not os.path.isdir(self.persist_dir):
                raise FileNotFoundError(f"No Chroma directory: {self.persist_dir}")
            self.vs = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_dir,
            )
        elif self.store_type == "faiss":
            from langchain_community.vectorstores import FAISS
            if not os.path.isdir(self.persist_dir):
                raise FileNotFoundError(f"No FAISS directory: {self.persist_dir}")
            self.vs = FAISS.load_local(
                self.persist_dir,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            raise ValueError(f"Unsupported VECTOR_STORE: {self.store_type}")
        return self.vs

    def retriever(self, k: int = 4):
        if self.vs is None:
            self.load()
        return self.vs.as_retriever(search_kwargs={"k": k})