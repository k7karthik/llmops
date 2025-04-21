# app.py (Upgraded: Multi-PDF + Chat History)

import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains.question_answering import load_qa_chain
import tempfile
import os
import uuid

st.set_page_config(page_title="PDF Q&A Bot", layout="centered")
st.title("📄 Ask Questions About Multiple PDFs")

if "pdf_db" not in st.session_state:
    st.session_state.pdf_db = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

uploaded_files = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_id = uploaded_file.name
        if file_id not in st.session_state.pdf_db:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_pdf_path = tmp_file.name

            loader = PyPDFLoader(tmp_pdf_path)
            pages = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            docs = splitter.split_documents(pages)
            embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            db = FAISS.from_documents(docs, embedding)

            st.session_state.pdf_db[file_id] = db
            st.session_state.chat_history[file_id] = []
            os.unlink(tmp_pdf_path)

    selected_pdf = st.selectbox("Choose a PDF to ask from:", options=list(st.session_state.pdf_db.keys()))

    question = st.text_input("Ask a question about the selected PDF")

    if question and selected_pdf:
        with st.spinner("Thinking..."):
            db = st.session_state.pdf_db[selected_pdf]
            relevant_docs = db.similarity_search(question)

            llm = Ollama(model="llama2")
            qa_chain = load_qa_chain(llm, chain_type="stuff")
            answer = qa_chain.run(input_documents=relevant_docs, question=question)

            st.session_state.chat_history[selected_pdf].append((question, answer))

    if selected_pdf in st.session_state.chat_history:
        st.subheader("🕓 Chat History for:", divider="rainbow")
        for q, a in st.session_state.chat_history[selected_pdf]:
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {a}")
            st.markdown("---")
