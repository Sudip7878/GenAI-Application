import streamlit as st
import os
from dotenv import load_dotenv

# LangChain & Groq
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Embeddings & Vector DB
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# -----------------------------
# Load environment / Streamlit secrets
# -----------------------------
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN") or st.secrets.get("hf", {}).get("token")
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("groq", {}).get("api_key")

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Chat With Your PDF", layout="wide")
st.title("Chat With Your PDF")
st.write("Upload a PDF and ask anything related to it.")

# -----------------------------
# PDF Upload
# -----------------------------
uploaded_file = st.file_uploader("📤 Upload your PDF here", type=["pdf"])

retriever = None
if uploaded_file:
    with st.spinner("📑 Reading PDF..."):
        # Save to a temporary file
        temp_path = f"./temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load and split
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)

        # Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )

        # Create vectorstore (in-memory, no persist)
        vectorstore = Chroma.from_documents(splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        st.success("✅ Vectorstore created successfully. You can now start asking questions!")

# -----------------------------
# RAG Q&A Loop
# -----------------------------
if retriever:
    if prompt := st.chat_input("👉 Ask your question..."):
        st.chat_message("user").write(prompt)

        # Initialize Groq LLM
        llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")

        # System prompt (English)
        system_prompt = (
            "You are a helpful assistant. "
            "Always answer in clear, simple English. "
            "Use only the provided context to answer. "
            "If the context is empty or irrelevant, say 'I don’t know'. "
            "Keep answers short and concise (maximum 8 sentences).\n\n{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        # Build chain
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # Run RAG
        rag_response = rag_chain.invoke({"input": prompt})
        answer = rag_response["answer"]

        # Show answer
        st.chat_message("assistant").write(answer)
