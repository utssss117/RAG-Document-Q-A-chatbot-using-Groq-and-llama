# RAG Document Q&A Chatbot with Groq and Llama

A Retrieval-Augmented Generation (RAG) chatbot that allows you to ask questions about your research papers and documents using Groq's Llama model.

## Features

- 📄 **Document Processing**: Load and process PDF documents from the `research_papers/` folder
- 🤖 **AI-Powered Q&A**: Ask questions about your documents using Groq's Llama 3.1 model
- 🔍 **Semantic Search**: Find relevant information using vector embeddings and FAISS
- ⚡ **Fast Responses**: Get answers with response time tracking
- 📊 **Document Insights**: View retrieved document chunks for transparency

## Prerequisites

- Python 3.8 or higher
- Groq API key (get one from [Groq Console](https://console.groq.com/))

## Installation

1. **Clone or download this repository**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Create a `.env` file in the root directory and add your Groq API key:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. **Add your PDF documents:**
   Place your research papers and documents in the `research_papers/` folder.

## Usage

1. **Run the application:**
   ```bash
   streamlit run main.py
   ```

2. **Create document embeddings:**
   - Click the "📚 Create Document Embeddings" button
   - Wait for the processing to complete (this may take a few minutes for large documents)

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

The application uses the following default settings:
- **Model**: llama-3.1-8b-instant
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters
- **Max Documents**: 50 (for processing efficiency)

## Troubleshooting

- **No API key error**: Make sure your `.env` file contains a valid `GROQ_API_KEY`
- **No documents found**: Ensure PDF files are placed in the `research_papers/` folder
- **Slow processing**: For large document collections, consider increasing the chunk size or reducing the number of documents

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the [MIT License](LICENSE).