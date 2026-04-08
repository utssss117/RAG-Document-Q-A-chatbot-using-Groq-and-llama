import streamlit as st
import os
import time
import logging
from dotenv import load_dotenv

# ==============================
# 🔐 Logging Setup
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
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
# ❌ API Key Validation
# ==============================
if not groq_api_key or not groq_api_key.strip():
    st.error("❌ GROQ_API_KEY not found or empty. Add it to .env file")
    logger.error("GROQ_API_KEY is missing or empty")
    st.stop()
else:
    logger.info("✅ GROQ_API_KEY loaded successfully")

# ==============================
# 🤖 LLM Setup
# ==============================
try:
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant"
    )
    logger.info("✅ Groq LLM initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Groq LLM: {str(e)}")
    st.error(f"❌ Failed to initialize LLM: {str(e)}")
    st.stop()

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
    """Create vector embeddings from PDF documents with proper error handling."""
    if "vectors" not in st.session_state:
        if not os.path.exists("research_papers"):
            logger.error("research_papers folder not found")
            st.error("📁 'research_papers' folder not found!")
            return False

        try:
            with st.spinner("✨ Creating embeddings..."):
                logger.info("Starting embedding creation...")

                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                loader = PyPDFDirectoryLoader("research_papers")
                docs = loader.load()

                if not docs:
                    logger.warning("No PDFs found in research_papers folder")
                    st.error("❌ No PDFs found in research_papers folder")
                    return False

                logger.info(f"Loaded {len(docs)} PDF documents")

                # ✅ Improved Chunking
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=700,
                    chunk_overlap=150
                )

                final_docs = splitter.split_documents(docs)
                logger.info(f"Split documents into {len(final_docs)} chunks")

                # ✅ FAISS Vector Store
                st.session_state.vectors = FAISS.from_documents(
                    final_docs,
                    embeddings
                )

                logger.info("✅ Vector database created successfully")
                st.success("✅ Vector Database Ready!")
                return True

        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}", exc_info=True)
            st.error(f"❌ Error creating embeddings: {str(e)}")
            return False
    else:
        st.info("ℹ️ Vector database already initialized")
        return True

# ==============================
# 📚 Button
# ==============================
if st.button("📚 Initialize Document Database"):
    create_vector_embedding()

# ==============================
# 💬 User Input with Validation
# ==============================
user_prompt = st.text_input("💬 Ask a question from your documents:")

if user_prompt:
    # Input validation
    prompt_length = len(user_prompt)
    
    if prompt_length > 2000:
        st.warning(f"⚠️ Question is too long ({prompt_length}/2000 characters). Please shorten it.")
        logger.warning(f"User provided oversized prompt: {prompt_length} characters")
    elif len(user_prompt.strip()) == 0:
        st.warning("⚠️ Please enter a valid question (not just whitespace)")
        logger.warning("User provided empty/whitespace-only prompt")
    else:
        if "vectors" not in st.session_state:
            st.warning("⚠️ Initialize database first by clicking the button above")
            logger.warning("User attempted query without initialized vector database")
        else:
            try:
                logger.info(f"Processing user query: {user_prompt[:100]}...")
                
                # ✅ Improved Retriever
                retriever = st.session_state.vectors.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 5}
                )

                # ✅ LCEL Chain Construction (LangChain 0.2.x compatible)
                def format_docs(docs):
                    """Format retrieved documents as context string."""
                    return "\n\n".join(doc.page_content for doc in docs)

                start = time.perf_counter()
                
                # Retrieve relevant documents
                docs = retriever.invoke(user_prompt)
                context_str = format_docs(docs)
                
                # Build and invoke prompt
                messages = prompt.invoke({"context": context_str, "input": user_prompt})
                
                # Get answer from LLM
                result = llm.invoke(messages)
                answer = result.content if hasattr(result, 'content') else str(result)
                
                end = time.perf_counter()
                
                response_time = end - start
                logger.info(f"Query processed successfully in {response_time:.2f} seconds")

                # ✅ Output
                st.write("### 📌 Answer:")
                st.write(answer)

                st.info(f"⏱ Response time: {response_time:.2f} sec")

                # ✅ Show Sources
                with st.expander("🔎 Source Context"):
                    if docs:
                        for i, doc in enumerate(docs):
                            st.markdown(f"**Source {i+1}:**")
                            st.write(doc.page_content)
                            st.divider()
                    else:
                        st.write("No relevant sources found.")

            except Exception as e:
                logger.error(f"Error processing query: {str(e)}", exc_info=True)
                st.error(f"❌ Error processing your question: {str(e)}")
                st.info("💡 Tip: Check that your PDFs are valid and your API key is correct")