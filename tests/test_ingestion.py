from langchain.schema import Document
from src.ingestion.splitter import split_documents

def test_splitter_basic():
    docs = [Document(page_content="A"*500 + "\n\n" + "B"*500)]
    parts = split_documents(docs, chunk_size=300, chunk_overlap=50)
    assert len(parts) >= 3
    assert all(p.page_content for p in parts)