import streamlit as st
import os
import time
from dotenv import load_dotenv

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader


# Load Environment Variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
# Streamlit UI
st.title("📄 RAG Document Q&A With Groq &llama3")


# LLM Setup (Groq Only)
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"

)
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based only on the provided context.
    
    <context>
    {context}
    </context>

    Question: {input}
    """
)
def create_vector_embedding():
    if "vectors" not in st.session_state:

        with st.spinner("Processing documents..."):
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
if st.button("📚 Create Document Embeddings"):
    create_vector_embedding()
    st.success("Vector Database is Ready ✅")

user_prompt = st.text_input("Ask a question from the research papers")
if user_prompt:

    if "vectors" not in st.session_state:
        st.warning("Please create embeddings first.")
    else:

        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()

        response = retrieval_chain.invoke({"input": user_prompt})

        st.write("### 📌 Answer:")
        st.write(response["answer"])

        st.write(f"⏱ Response time: {time.process_time() - start:.2f} seconds")

        # Show Retrieved Documents
        with st.expander("🔎 Document Similarity Search Results"):
            for doc in response["context"]:
                st.write(doc.page_content)
                st.write("--------------------------------------------------")