# Notes RAG Assistant

A lightweight, efficient RAG (Retrieval-Augmented Generation) assistant that answers questions from your custom notes and PDF documents using local embeddings, persistent vector storage, and Groq's LLaMA 3.3 70B model.

---

## Features

- 📄 **PDF & TXT Ingestion**: Automatically parses text files and PDFs, extracting text with page-level tracking (`[Page X]`).
- ⚡ **Local Embeddings**: Uses `sentence-transformers` (`all-MiniLM-L6-v2`) for local embedding generation (zero embedding API cost).
- 💾 **Persistent Vector Store**: Uses ChromaDB persistent storage (`chroma_db/`) so documents are indexed once and loaded instantly on future runs.
- 🎯 **Source-Aware Retrieval**: Passes source metadata (`[File: doc.pdf]`) and top matching context chunks to the LLM for accurate, cited answers.
- 🔄 **Interactive Re-indexing**: Type `reindex` in the interactive console to rebuild the vector database whenever new files are added.
- 🚀 **LLaMA 3.3 70B via Groq**: Fast inference powered by Groq's API.

---

## Tech Stack

- **Language**: Python 3.10+
- **LLM Provider**: Groq API (`llama-3.3-70b-versatile`)
- **Vector Database**: ChromaDB (`chromadb.PersistentClient`)
- **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`)
- **PDF Parser**: `pypdf`
- **Environment Management**: `python-dotenv`

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Riyaz510/Notes-RAG.git
cd Notes-RAG
```

### 2. Set Up Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Key
Create a `.env` file in the root directory and add your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```
*(Get a free Groq API key at [console.groq.com](https://console.groq.com/))*

---

## Usage

1. Place your `.pdf` or `.txt` files in the `docs/` folder (it will be created automatically on first run if missing).
2. Run the assistant:
   ```bash
   python main.py
   ```
3. Ask questions in the interactive terminal:
   ```text
   You: What is the PDF about?
   Assistant: The PDF guide covers System Design Patterns for LLMs, RAG, and AI agents...

   You: reindex   # Refreshes the database if you add new documents
   You: exit      # Quits the program
   ```

---

## Project Structure

```text
Notes-RAG/
├── docs/                 # Place your PDF and TXT documents here (Git-ignored)
├── chroma_db/            # Persistent ChromaDB vector storage (Git-ignored)
├── .env                  # API keys and environment variables (Git-ignored)
├── .gitignore            # Git ignore configuration
├── main.py               # Main application entry point
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## License

MIT License
