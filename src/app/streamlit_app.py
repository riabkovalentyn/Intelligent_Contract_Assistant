import os
import tempfile
import streamlit as st
from src.utils.config import config
from src.ingestion.pdfLoader import load_pdf
from src.ingestion.splitter import split_documents
from src.embeddings.vector_store import VectorStoreManager
from src.qa.retrieval import build_qa_chain
from src.ner.ner import extract_entities_from_docs
from src.feedback.feedback import record_feedback


st.set_page_config(page_title="Intelligent Contract Assistant", layout="wide")

st.sidebar.header("Settings")
k = st.sidebar.slider("Top-K passages", 1, 10, 4)
show_sources = st.sidebar.checkbox("Show sources", value=True)
session = st.sidebar.text_input("Session ID", value="web")

st.title("Intelligent Contract Assistant")

tab1, tab2, tab3 = st.tabs(["Ask", "Ingest PDF", "NER"])

with tab2:
    st.subheader("Ingest a contract PDF")
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        docs = load_pdf(tmp_path)
        if not docs:
            st.error("No text extracted from PDF.")
        else:
            chunks = split_documents(docs)
            vsm = VectorStoreManager()
            vsm.build_from_documents(chunks)
            st.success(f"Ingested {len(chunks)} chunks. Vector store saved to {config.vector_dir}")
        os.unlink(tmp_path)

with tab1:
    st.subheader("Ask a question")
    question = st.text_input("Question", placeholder="What is the termination clause?")
    ask = st.button("Ask", type="primary", disabled=not question)
    if ask:
        try:
            vsm = VectorStoreManager()
            retriever = vsm.retriever(k=k)
            chain = build_qa_chain(retriever)
            res = chain.invoke({"query": question})
            answer = res.get("result", "")
            sources = res.get("source_documents", []) or []
            st.markdown("### Answer")
            st.write(answer)

            if show_sources and sources:
                st.markdown("### Sources")
                for i, doc in enumerate(sources, start=1):
                    meta = doc.metadata or {}
                    loc = f'{os.path.basename(meta.get("source",""))}#p{meta.get("page","?")}'
                    with st.expander(f"[{i}] {loc}"):
                        st.write(doc.page_content)
            fb = st.radio("Was this helpful?", options=["Skip", "Yes", "No"], horizontal=True, index=0)
            if fb in ("Yes", "No"):
                record_feedback(
                    session_id=session,
                    question=question,
                    answer=answer,
                    rating=1 if fb == "Yes" else 0,
                    notes=None,
                    sources=[
                        {"source": (d.metadata or {}).get("source"), "page": (d.metadata or {}).get("page")}
                        for d in sources
                    ],
                )
                st.success("Feedback recorded.")
        except Exception as e:
            st.error(str(e))

with tab3:
    st.subheader("Extract entities (NER)")
    uploaded2 = st.file_uploader("Upload PDF for NER", type=["pdf"], key="nerpdf")
    if uploaded2:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded2.read())
            tmp_path2 = tmp.name
        try:
            docs = load_pdf(tmp_path2)
            ents = extract_entities_from_docs(docs)
            st.json(ents)
        finally:
            os.unlink(tmp_path2)