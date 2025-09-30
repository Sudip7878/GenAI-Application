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
st.set_page_config(page_title="Sewa Chatbot (Nepali)", layout="wide")
st.title("📄 Sewa Chatbot (नेपाली)")
st.write("कृपया PDF अपलोड गर्नुहोस् र सोध्न सुरु गर्नुहोस्।")

# -----------------------------
# PDF Upload
# -----------------------------
uploaded_file = st.file_uploader("📤 यहाँ PDF अपलोड गर्नुहोस्", type=["pdf"])

retriever = None
if uploaded_file:
    with st.spinner("📑 PDF पढ्दैछु..."):
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

        st.success("✅ Vectorstore तयार भयो। अब तपाईं प्रश्न सोध्न सक्नुहुन्छ।")

# -----------------------------
# RAG Q&A Loop
# -----------------------------
if retriever:
    if prompt := st.chat_input("👉 प्रश्न लेख्नुहोस्..."):
        st.chat_message("user").write(prompt)

        # Initialize Groq LLM
        llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")

        # System prompt enforcing Nepali-only answers
        system_prompt = (
            "तपाईँ एउटा सहायक सहायक हुनुहुन्छ। "
            "सधैं केवल नेपाली भाषामा मात्र जवाफ दिनुहोस्। "
            "कुनै पनि हालतमा हिन्दी वा अन्य भाषा प्रयोग नगर्नुहोस्। "
            "दिइएको प्रसङ्ग (context) प्रयोग गरेर मात्र जवाफ दिनुहोस्। "
            "यदि प्रसङ्ग खाली छ वा सम्बन्धित छैन भने 'मलाई थाहा छैन' भन्नुहोस्। "
            "जवाफ छोटकरीमा दिनुहोस् (अधिकतम ८ वाक्यसम्म)।\n\n{context}"
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

        # Prepend explicit Nepali instruction
        user_input = "नेपालीमा जवाफ दिनुहोस्: " + prompt

        # Run RAG
        rag_response = rag_chain.invoke({"input": user_input})
        answer = rag_response["answer"]

        # Show answer
        st.chat_message("assistant").write(answer)
