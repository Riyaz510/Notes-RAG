"""
Mini RAG Assistant
------------------
Answers questions using your own notes/PDFs as context, via:
  - sentence-transformers for local embeddings (no API cost)
  - ChromaDB as the vector store
  - Groq (LLaMA 3.3 70B) for the actual answer generation

Usage:
  1. Put .txt or .pdf files in a folder called `docs/`
  2. Set your Groq API key:  export GROQ_API_KEY="your_key_here"
  3. Run:  python main.py
"""

import os
import glob
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from groq import Groq

DOCS_DIR = "docs"
CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 50
TOP_K = 3              # how many chunks to retrieve per question
MODEL = "llama-3.3-70b-versatile"


def load_documents(docs_dir):
    """Read all .txt and .pdf files from docs_dir and return raw text per file."""
    texts = {}
    for path in glob.glob(os.path.join(docs_dir, "*")):
        if path.endswith(".pdf"):
            reader = PdfReader(path)
            texts[path] = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif path.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                texts[path] = f.read()
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


def build_vector_store():
    """Load docs, chunk them, embed them, and store in a local Chroma collection."""
    client = chromadb.Client()
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_or_create_collection(name="notes", embedding_function=embed_fn)

    documents = load_documents(DOCS_DIR)
    ids, all_chunks = [], []
    for path, text in documents.items():
        for i, chunk in enumerate(chunk_text(text)):
            ids.append(f"{os.path.basename(path)}-{i}")
            all_chunks.append(chunk)

    if all_chunks:
        collection.add(documents=all_chunks, ids=ids)
    print(f"Indexed {len(all_chunks)} chunks from {len(documents)} file(s).")
    return collection


def answer_question(collection, question, groq_client):
    """Retrieve relevant chunks, then ask the LLM to answer using only that context."""
    results = collection.query(query_texts=[question], n_results=TOP_K)
    context = "\n\n---\n\n".join(results["documents"][0]) if results["documents"] else ""

    prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say you don't know.

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
    if not os.path.isdir(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        print(f"Created '{DOCS_DIR}/' — add some .txt or .pdf files there and rerun.")
        return

    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    collection = build_vector_store()

    print("Ask questions about your docs (type 'exit' to quit).")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        answer = answer_question(collection, question, groq_client)
        print(f"\nAssistant: {answer}")


if __name__ == "__main__":
    main()
