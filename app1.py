import streamlit as st
import os
import tempfile

# LangChain & Groq
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Embeddings & Vector DB
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader

# ---- Streamlit Layout ----
st.set_page_config(page_title="Groq PDF RAG", layout="wide")
st.title("📄 Groq PDF Q&A")
st.write("Upload a PDF and ask questions from its content.")

# ---- Load API keys ----
GROQ_API_KEY = st.secrets["groq"]["api_key"]
HF_TOKEN = st.secrets["hf"]["token"]
os.environ["HF_TOKEN"] = HF_TOKEN

# ---- File Upload ----
uploaded_file = st.file_uploader("📤 Upload a PDF file", type=["pdf"])

retriever = None
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    # Load PDF into documents
    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()

    # Create embeddings and vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    st.success("✅ PDF processed and vectorstore created successfully")

# ---- User input & RAG ----
if retriever:
    if prompt := st.chat_input("Ask a question about the uploaded PDF..."):
        st.chat_message("user").write(prompt)

        # Initialize Groq LLM
        llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")

        # System prompt (no Nepali restriction)
        system_prompt = (
            "You are a helpful assistant. "
            "Use only the given context to answer the question. "
            "If the context is empty or irrelevant, reply with 'I don't know'.\n\n{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        # Create RAG chain
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # Run RAG
        rag_response = rag_chain.invoke({"input": prompt})
        answer = rag_response["answer"]

        # Show response
        st.chat_message("assistant").write(answer)
