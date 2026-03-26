# 🧠 DocMind v2 — RAG PDF Chat

DocMind is a high-performance Retrieval-Augmented Generation (RAG) application that allows you to chat with your PDF documents using real LLM chains and advanced retrieval techniques.

## 🚀 Features

- **Real RAG Chain**: Powered by `langchain-huggingface` and `FAISS`.
- **Advanced Retrieval**: Uses **Maximal Marginal Relevance (MMR)** for diverse and relevant context retrieval.
- **Interactive Chat UI**: Premium dark-themed Streamlit interface with full chat history.
- **Smart Indexing**: Recursive character text splitting and MiniLM embeddings.
- **Confidence Scores**: Real-time relevance confidence for every AI response.
- **Source Tracing**: View the exact passages used to generate each answer.
- **File Deduplication**: MD5 hashing to prevent redundant processing of the same document.

## 🛠️ Tech Stack

- **Frontend**: Streamlit (with custom CSS)
- **Framework**: LangChain
- **Vector Store**: FAISS
- **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`)
- **LLM**: HuggingFace Hub (`google/flan-t5-large` or others)

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Arawn-D/rag-pdf-chat.git
   cd rag-pdf-chat
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   Create a `.env` file and add your HuggingFace API Token:
   ```env
   HUGGINGFACEHUB_API_TOKEN=your_token_here
   ```

## 🚀 Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

## 📄 License

MIT
