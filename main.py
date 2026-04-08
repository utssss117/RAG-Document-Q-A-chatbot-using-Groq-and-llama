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

# ==============================
# 🔐 Load Environment Variables
# ==============================
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# ==============================
# 🎨 Streamlit UI
# ==============================
st.set_page_config(page_title="Groq RAG Chatbot", page_icon="📄")
st.title("📄 RAG Document Q&A With Groq & Llama 3")

# ==============================
# ❌ API Key Check
# ==============================
if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found. Add it to .env file")
    st.stop()

# ==============================
# 🤖 LLM Setup
# ==============================
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)

# ==============================
# 🧠 Prompt Template
# ==============================
prompt = ChatPromptTemplate.from_template("""
Answer the question ONLY from the provided context.
If the answer is not in the context, say:
"I cannot find the answer in the provided documents."

<context>
{context}
</context>

Question: {input}
""")

# ==============================
# 📚 Create Vector DB
# ==============================
def create_vector_embedding():
    if "vectors" not in st.session_state:

        if not os.path.exists("research_papers"):
            st.error("📁 'research_papers' folder not found!")
            return

        with st.spinner("✨ Creating embeddings..."):

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            loader = PyPDFDirectoryLoader("research_papers")
            docs = loader.load()

            if not docs:
                st.error("❌ No PDFs found in folder")
                return

            # ✅ Improved Chunking
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=700,
                chunk_overlap=150
            )

            final_docs = splitter.split_documents(docs)

            # ✅ FAISS Vector Store
            st.session_state.vectors = FAISS.from_documents(
                final_docs,
                embeddings
            )

            st.success("✅ Vector Database Ready!")

# ==============================
# 📚 Button
# ==============================
if st.button("📚 Initialize Document Database"):
    create_vector_embedding()

# ==============================
# 💬 User Input
# ==============================
user_prompt = st.text_input("💬 Ask a question from your documents:")

if user_prompt:

    if "vectors" not in st.session_state:
        st.warning("⚠️ Initialize database first")
    else:
        try:
            # ✅ Improved Retriever
            retriever = st.session_state.vectors.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5}
            )

            document_chain = create_stuff_documents_chain(llm, prompt)
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            start = time.perf_counter()
            response = retrieval_chain.invoke({"input": user_prompt})
            end = time.perf_counter()

            # ✅ Output
            st.write("### 📌 Answer:")
            st.write(response["answer"])

            st.info(f"⏱ Response time: {end - start:.2f} sec")

            # ✅ Show Sources
            with st.expander("🔎 Source Context"):
                for i, doc in enumerate(response["context"]):
                    st.markdown(f"**Source {i+1}:**")
                    st.write(doc.page_content)
                    st.divider()

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")