from __future__ import annotations
from typing import List, Dict
from dataclasses_json import config
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_documents(docs: List[Document], 
                    chunk_size: int = 1000, 
                    chunk_overlap: int = 200) -> List[Document]:
    chunk_size = chunk_size or config.chunk_size
    chunk_overlap = chunk_overlap or config.chunk_overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", " ", ""],
    )

    return splitter.split_documents(docs)