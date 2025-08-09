from __future__ import annotations
from typing import Any
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger('retrieval')

def build_qa_chain(retriever: Any):
    if not config.openai_api_key and config.embedding_provider == 'openai':
        logger.warning("OPENAI_API_KEY is not set. Using FakeEmbeddings for retrieval.")

    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model='gpt-4o-mini',
        temperature=0.1,
        api_key=config.openai_api_key,
    )

    template =(
        "You are a contract analysis assistant. Use only the provided context to answer.\n"
        "If the answer is not in the context, say you don't know.\n\n"
        "Question: {question}\n\n"
        "Context:\n{context}\n\n"
        "Answer:"
    )

    prompt = PromptTemplate.from_template(template)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return chain