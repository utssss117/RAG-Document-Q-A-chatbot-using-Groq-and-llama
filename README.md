# RAG Document Q&A Chatbot with Groq and Llama

A Retrieval-Augmented Generation (RAG) chatbot that allows you to ask questions about your research papers and documents using Groq's Llama model.

## Features

- 📄 **Document Processing**: Load and process PDF documents from the `research_papers/` folder
- 🤖 **AI-Powered Q&A**: Ask questions about your documents using Groq's Llama 3.1 model
- 🔍 **Semantic Search**: Find relevant information using vector embeddings and FAISS
- ⚡ **Fast Responses**: Get answers with response time tracking
- 📊 **Document Insights**: View retrieved document chunks for transparency

## 🚀 Live Demo

Try the chatbot online: [https://rag-document-q-a-chatbot-using-groq-and-llama-kuanicxo24agy6ut.streamlit.app/](https://rag-document-q-a-chatbot-using-groq-and-llama-kuanicxo24agy6ut.streamlit.app/)

## Prerequisites

- Python 3.10 or higher
- Groq API key (get one from [Groq Console](https://console.groq.com/))

## Installation

1. **Clone or download this repository**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Create a `.env` file in the root directory with your API keys. Copy from `.env.example` and fill in your credentials:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   HF_API_KEY=your_huggingface_api_key_here  # Optional, for HuggingFace models
   ```
   ⚠️ **Important**: Never commit `.env` to version control - it's in `.gitignore` by default

4. **Add your PDF documents:**
   Place your research papers and documents in the `research_papers/` folder.

## Usage

1. **Run the application:**
   ```bash
   streamlit run main.py
   ```

2. **Initialize the document database:**
   - Click the "📚 Initialize Document Database" button
   - Wait for the processing to complete (this may take a few minutes for large documents)
   - ✅ The vector database will be cached in memory for this session

3. **Ask questions:**
   - Enter your question in the text input field
   - Get AI-powered answers based on your documents
   - View response time and retrieved document chunks

## Project Structure

```
├── main.py                 # Main Streamlit application
├── requirements.txt        # Python dependencies
├── research_papers/        # Folder for PDF documents
│   ├── Attention.pdf
│   └── LLM.pdf
└── README.md              # This file
```

## Dependencies

- **streamlit**: Web app framework
- **langchain**: LLM framework for RAG implementation
- **langchain-groq**: Groq integration for LangChain
- **faiss-cpu**: Vector database for similarity search
- **sentence-transformers**: Text embeddings
- **pypdf**: PDF document loading
- **python-dotenv**: Environment variable management

## How It Works

1. **Document Loading**: PDFs are loaded from the `research_papers/` folder
2. **Text Splitting**: Documents are split into manageable chunks
3. **Embedding Creation**: Text chunks are converted to vector embeddings using HuggingFace models
4. **Vector Storage**: Embeddings are stored in a FAISS vector database
5. **Question Processing**: User questions are embedded and used to retrieve relevant document chunks
6. **Answer Generation**: Retrieved chunks are fed to Groq's Llama model to generate answers

## Configuration

The LLM Model**: llama-3.1-8b-instant (via Groq API)
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
- **Chunk Size**: 700 characters
- **Chunk Overlap**: 150 characters
- **Retrieval Method**: Maximum Marginal Relevance (MMR) with k=5 documents
- **Max Input Prompt**: 2000 characters
- **Max Documents**: 50 (for processing efficiency)

| Issue | Solution |
|-------|----------|
| **GROQ_API_KEY not found error** | Create a `.env` file in the project root with `GROQ_API_KEY=your_key_here`. Copy `.env.example` as a template. |
| **No documents found** | Ensure PDF files are placed in the `research_papers/` folder at the project root. |
| **Slow embedding creation** | First-time embedding creation takes time. Embeddings are cached in memory during the session. |
| **"Initialize database first" warning** | Click the "📚 Initialize Document Database" button before asking questions. |
| **Empty or whitespace API key** | Ensure your API key in `.env` is not empty or contains only whitespace. |
| **Very long responses** | Questions longer than 2000 characters are limited. Break your question into smaller parts. |

## Production Deployment

For production use:
- Use an `.env` file with your production API keys
- Consider implementing document caching to avoid re-processing PDFs
- Add monitoring/logging for API usage and errors
- Test with your specific PDF documents before deploying
- Keep `sentence-transformers` and `torch` dependencies up to date
- **No documents found**: Ensure PDF files are placed in the `research_papers/` folder
- **Slow processing**: For large document collections, consider increasing the chunk size or reducing the number of documents

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the [MIT License](LICENSE).