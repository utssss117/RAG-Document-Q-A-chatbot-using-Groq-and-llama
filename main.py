import streamlit as st
import os
import time
from dotenv import load_dotenv

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings # Updated for better compatibility
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Load Environment Variables from .env file
groq_api_key = st.secrets["GROQ_API_KEY"]

# Streamlit UI
st.set_page_config(page_title="Groq RAG Chatbot", page_icon="📄")
st.title("📄 RAG Document Q&A With Groq & Llama 3")

# 1. Check if API Key exists before initializing LLM
if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found. Please add it to your .env file.")
    st.stop()

# LLM Setup
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based only on the provided context.
    If the answer is not in the context, say "I cannot find the answer in the provided documents."
    
    <context>
    {context}
    </context>

    Question: {input}
    """
)

# 2. Vector Embedding Logic
def create_vector_embedding():
    # Only run if vectors don't exist yet
    if "vectors" not in st.session_state:
        if not os.path.exists("research_papers"):
            st.error("📁 'research_papers' folder not found! Please create it and add PDFs.")
            return

        with st.spinner("✨ Creating embeddings... This may take a minute."):
            # Using HuggingFaceEmbeddings (runs locally on your CPU/GPU)
            st.session_state.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            # Load PDFs
            loader = PyPDFDirectoryLoader("research_papers")
            docs = loader.load()

            # Split text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            final_documents = text_splitter.split_documents(docs[:50])

            # Create FAISS vector store
            st.session_state.vectors = FAISS.from_documents(
                final_documents,
                st.session_state.embeddings
            )
            st.success("✅ Vector Database is Ready!")

# Sidebar/Button UI
if st.button("📚 Initialize Document Database"):
    create_vector_embedding()

# 3. Chat Interface
user_prompt = st.text_input("Ask a question from the research papers:")

if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("⚠️ Please click the 'Initialize Document Database' button first.")
    else:
        # Create the chains
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Performance tracking
        start_time = time.perf_counter()
        response = retrieval_chain.invoke({"input": user_prompt})
        end_time = time.perf_counter()

        # Display results
        st.write("### 📌 Answer:")
        st.write(response["answer"])
        st.info(f"⏱ Response generated in {end_time - start_time:.2f} seconds")

        # Show Retrieved Documents (for transparency)
        with st.expander("🔎 Source Context (Similarity Search)"):
            for i, doc in enumerate(response["context"]):
                st.markdown(f"**Source {i+1}:**")
                st.write(doc.page_content)
                st.divider()