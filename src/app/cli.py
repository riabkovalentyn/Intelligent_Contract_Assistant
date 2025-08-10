from __future__ import annotations
import argparse
import json
import os
from typing import List
from src.utils.config import config
from src.utils.logger import get_logger
from src.ingestion.pdfLoader import load_pdf
from src.ingestion.splitter import split_documents
from src.embeddings.vector_store import VectorStoreManager
from src.qa.retrieval import build_qa_chain
from src.ner.ner import extract_entities_from_docs
from src.feedback.feedback import record_feedback

logger = get_logger("cli")


def cmd_ingest(pdf_path: str):
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(pdf_path)
    docs = load_pdf(pdf_path)
    if not docs:
        raise RuntimeError("No text found in PDF")
    chunks = split_documents(docs)
    vsm = VectorStoreManager()
    vsm.build_from_documents(chunks)
    logger.info("Ingestion complete. Vector store at %s", config.vector_dir)


def cmd_ask(question: str, k: int = 4, show_sources: bool = True, session: str | None = None):
    vsm = VectorStoreManager()
    retriever = vsm.retriever(k=k)
    chain = build_qa_chain(retriever)
    res = chain.invoke({"query": question})
    answer = res.get("result", "")
    sources = res.get("source_documents", []) or []

    print("\nAnswer:\n", answer)
    if show_sources and sources:
        print("\nSources:")
        for i, doc in enumerate(sources, start=1):
            meta = doc.metadata or {}
            loc = f'{os.path.basename(meta.get("source",""))}#p{meta.get("page","?")}'
            snippet = doc.page_content[:400].replace("\n", " ")
            print(f"\n[{i}] {loc}\n{snippet}...")


    try:
        fb = input("\nWas this helpful? (y/n, Enter to skip): ").strip().lower()
        if fb in ("y", "n"):
            record_feedback(
                session_id=session or "default",
                question=question,
                answer=answer,
                rating=1 if fb == "y" else 0,
                notes=None,
                sources=[
                    {"source": (d.metadata or {}).get("source"), "page": (d.metadata or {}).get("page")}
                    for d in sources
                ],
            )
            print("Feedback recorded.")
    except Exception:
        pass


def cmd_ner(pdf_path: str):
    docs = load_pdf(pdf_path)
    ents = extract_entities_from_docs(docs)
    print(json.dumps(ents, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(prog="contract-assistant", description="Intelligent Contract Assistant (LangChain)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest a PDF into vector store")
    p_ingest.add_argument("--pdf", required=True, help="Path to contract PDF")

    p_ask = sub.add_parser("ask", help="Ask a question about ingested contract")
    p_ask.add_argument("-q", "--question", required=True)
    p_ask.add_argument("-k", "--topk", type=int, default=4)
    p_ask.add_argument("--no-sources", action="store_true")
    p_ask.add_argument("--session", default=None)

    p_ner = sub.add_parser("ner", help="Extract entities (parties, dates, money) from a PDF")
    p_ner.add_argument("--pdf", required=True)

    args = parser.parse_args()

    if args.cmd == "ingest":
        cmd_ingest(args.pdf)
    elif args.cmd == "ask":
        cmd_ask(args.question, k=args.topk, show_sources=not args.no_sources, session=args.session)
    elif args.cmd == "ner":
        cmd_ner(args.pdf)


if __name__ == "__main__":
    main()