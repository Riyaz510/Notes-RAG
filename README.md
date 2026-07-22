# notes-rag

A minimal RAG (Retrieval-Augmented Generation) chatbot that answers questions from your own notes and PDFs — local embeddings, local vector store, LLaMA 3.3 70B via Groq for generation.

## How it works

1. Drop `.txt` / `.pdf` files into `docs/`
2. Files are chunked and embedded locally with `sentence-transformers` (no API cost for embeddings)
3. Chunks are stored in a `ChromaDB` collection
4. On each question, the top-matching chunks are retrieved and passed to LLaMA 3.3 70B (Groq API) as context
5. The model answers strictly from that context

## Stack

- Python
- Groq API (LLaMA 3.3 70B)
- ChromaDB (vector store)
- sentence-transformers (local embeddings)
- pypdf

## Setup

```bash
pip install -r requirements.txt
export GROQ_API_KEY="your_key_here"
python main.py
```

Add files to the `docs/` folder it creates on first run, then rerun to index and start asking questions.

## Possible extensions

- Persist the Chroma collection to disk
- Wrap it in a small Flask/Express API
- Return source citations alongside answers
