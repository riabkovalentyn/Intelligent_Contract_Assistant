from langchain.schema import Document
from src.embeddings.vector_store import VectorStoreManager
from src.qa.retrieval import build_qa_chain

def test_retrieval_qa_offline(tmp_path, monkeypatch):
    # force FAISS to avoid external deps
    monkeypatch.setenv("VECTOR_STORE", "faiss")
    from src.utils.config import AppConfig
    # rebuild config with env override
    cfg = AppConfig(vector_store="faiss", vector_dir=str(tmp_path))
    cfg.ensure_dirs()

    vsm = VectorStoreManager(persist_dir=str(tmp_path))
    docs = [
        Document(page_content="The governing law is the State of New York.", metadata={"source": "x", "page": 1}),
        Document(page_content="Termination clause: Either party may terminate with 30 days notice.", metadata={"source": "x", "page": 2}),
    ]
    vsm.build_from_documents(docs)
    retriever = vsm.retriever(k=2)
    chain = build_qa_chain(retriever)
    res = chain.invoke({"query": "What is the termination clause?"})
    assert "terminate" in res["result"].lower()