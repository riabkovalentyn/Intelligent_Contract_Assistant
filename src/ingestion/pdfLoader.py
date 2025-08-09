from __future__ import annotations
from typing import List, Dict
from langchain.schema import Document
from src.utils.logger import  get_logger

logger = get_logger('pdfLoader')

def _clean_text(text: str) -> str:
    return '\n'.join(line.strip() for line in text.splitlines() if line.strip())

def load_pdf_pdfplumber(file_path: str) -> List[Document]:
    import pdfplumber
    docs: List[Document] = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = _clean_text(text)
            if text:
                docs.append(Document(page_content=text, metadata={"source": file_path, "page": i}))
    return docs

def load_pdf_pypdf2(path: str) -> List[Document]:
    from PyPDF2 import PdfReader  # lazy import
    docs: List[Document] = []
    reader = PdfReader(path)
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = _clean_text(text)
        if text:
            docs.append(Document(page_content=text, metadata={"source": path, "page": i}))
    return docs

def load_pdf(path: str) -> List[Document]:
    try:
        logger.info(f"Loading PDF using pdfplumber: %s", path)
        docs = load_pdf_pdfplumber(path)
        if docs:
            return docs
    except Exception as e:
        logger.warning(f"pdfplumber failed for %s", e)

    logger.info(f"Loading PDF using PyPDF2: %s", path)
    return load_pdf_pypdf2(path)