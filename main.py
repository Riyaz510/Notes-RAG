"""
Mini RAG Assistant
------------------
Answers questions using your own notes/PDFs as context, via:
  - sentence-transformers for local embeddings (no API cost)
  - ChromaDB as the vector store (persistent)
  - Groq (LLaMA 3.3 70B) for the actual answer generation

Usage:
  1. Put .txt or .pdf files in a folder called `docs/`
  2. Set your Groq API key in `.env` (GROQ_API_KEY=your_key)
  3. Run: python main.py
"""

import os
import glob
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from groq import Groq
from dotenv import load_dotenv

DOCS_DIR = "docs"
DB_DIR = "chroma_db"
CHUNK_SIZE = 1000      # characters per chunk (~200 words)
CHUNK_OVERLAP = 200     # characters overlap
TOP_K = 5              # how many chunks to retrieve per question
MODEL = "llama-3.3-70b-versatile"


def load_documents(docs_dir):
    """Read all .txt and .pdf files from docs_dir and return raw text per file."""
    texts = {}
    for path in glob.glob(os.path.join(docs_dir, "*")):
        if path.lower().endswith(".pdf"):
            try:
                reader = PdfReader(path)
                full_text = []
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        full_text.append(f"[Page {i+1}]\n{page_text}")
                texts[path] = "\n\n".join(full_text)
            except Exception as e:
                print(f"Error reading PDF '{path}': {e}")
        elif path.lower().endswith(".txt"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    texts[path] = f.read()
            except Exception as e:
                print(f"Error reading text file '{path}': {e}")
    return texts


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks so context isn't lost at chunk boundaries."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def build_vector_store(force_reindex=False):
    """Load docs, chunk them, embed them, and store in a persistent Chroma collection."""
    client = chromadb.PersistentClient(path=DB_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    if force_reindex:
        try:
            client.delete_collection(name="notes")
        except Exception:
            pass

    collection = client.get_or_create_collection(name="notes", embedding_function=embed_fn)

    # Check existing count
    existing_count = collection.count()
    if existing_count > 0 and not force_reindex:
        print(f"Loaded existing vector database with {existing_count} chunks.")
        return collection

    print("Indexing documents from 'docs/' folder... This may take a moment.")
    documents = load_documents(DOCS_DIR)
    ids, all_chunks, metadatas = [], [], []

    for path, text in documents.items():
        filename = os.path.basename(path)
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            ids.append(f"{filename}-{i}")
            all_chunks.append(chunk)
            metadatas.append({"source": filename, "chunk_index": i})

    if all_chunks:
        # Add in batches to avoid batch size limits
        batch_size = 100
        for b in range(0, len(all_chunks), batch_size):
            collection.add(
                documents=all_chunks[b:b+batch_size],
                ids=ids[b:b+batch_size],
                metadatas=metadatas[b:b+batch_size]
            )
        print(f"Successfully indexed {len(all_chunks)} chunks from {len(documents)} file(s).")
    else:
        print("No documents found to index.")

    return collection


def get_indexed_files(collection):
    """Return a list of unique document filenames in the collection."""
    try:
        data = collection.get(include=["metadatas"])
        if data and data.get("metadatas"):
            sources = {m.get("source") for m in data["metadatas"] if m and "source" in m}
            return list(sources)
    except Exception:
        pass
    return []


def answer_question(collection, question, groq_client):
    """Retrieve relevant chunks, then ask the LLM to answer using context and metadata."""
    results = collection.query(query_texts=[question], n_results=TOP_K)
    
    formatted_chunks = []
    if results and results.get("documents") and results["documents"][0]:
        docs = results["documents"][0]
        metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
        for doc, meta in zip(docs, metas):
            source = meta.get("source", "Unknown Document") if meta else "Unknown Document"
            formatted_chunks.append(f"[File: {source}]\n{doc}")

    context = "\n\n---\n\n".join(formatted_chunks) if formatted_chunks else "No relevant context found."
    indexed_files = get_indexed_files(collection)
    files_list_str = ", ".join(indexed_files) if indexed_files else "None"

    prompt = f"""You are a helpful RAG Assistant. Answer the user's question using ONLY the provided context below.
Available documents in database: {files_list_str}

Context:
{context}

Question: {question}
Answer:"""

    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


def main():
    load_dotenv()

    if not os.path.isdir(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        print(f"Created '{DOCS_DIR}/' — add some .txt or .pdf files there and rerun.")
        return

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY is not set.")
        print("Please set GROQ_API_KEY in your .env file or environment variables.")
        return

    groq_client = Groq(api_key=api_key)
    collection = build_vector_store()

    print("\nAsk questions about your docs (type 'exit' to quit, or 'reindex' to refresh files).")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not question:
            continue

        if question.lower() in {"exit", "quit"}:
            break

        if question.lower() == "reindex":
            collection = build_vector_store(force_reindex=True)
            continue

        answer = answer_question(collection, question, groq_client)
        print(f"\nAssistant: {answer}")


if __name__ == "__main__":
    main()
