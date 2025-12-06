# =====================================================
# backend/ingest_data.py
# =====================================================

import os
import shutil
import pathlib
from dotenv import load_dotenv
from tqdm import tqdm

# LangChain modules
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# -----------------------------------------------------
# Setup environment
# -----------------------------------------------------
project_root = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(project_root / ".env")

PERSIST_DIR = "embeddings/chroma"
COLLECTION_NAME = "compliance_docs_v2"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

print("✅ Environment initialized.")
print(f"✅ Using embedding model: {EMBED_MODEL}")

# -----------------------------------------------------
# 1️⃣ Load documents
# -----------------------------------------------------
def load_docs(data_dir="data"):
    """Loads text, Word, and PDF documents safely from the given directory."""
    from langchain_community.document_loaders import (
        TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
    )

    docs = []
    supported_ext = {".txt", ".pdf", ".docx"}

    for root, _, files in os.walk(data_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            path = os.path.join(root, f)

            if ext not in supported_ext:
                continue  # skip unsupported files

            try:
                if ext == ".txt":
                    loader = TextLoader(path, encoding="utf-8")
                elif ext == ".pdf":
                    loader = PyPDFLoader(path)
                elif ext == ".docx":
                    loader = UnstructuredWordDocumentLoader(path)

                loaded_docs = loader.load()
                if not loaded_docs:
                    print(f"⚠️ {f} returned 0 docs, skipping.")
                    continue

                docs.extend(loaded_docs)
            except Exception as e:
                print(f"⚠️ Failed to load {f}: {e}")

    if not docs:
        raise ValueError("❌ No documents loaded. Check your data/source_docs/ folder.")

    print(f"✅ Loaded {len(docs)} documents from {data_dir}")
    return docs


# -----------------------------------------------------
# 2️⃣ Chunk documents
# -----------------------------------------------------
def chunk_docs(docs, chunk_size=1500, chunk_overlap=300):
    """Splits documents into overlapping text chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    return chunks

# -----------------------------------------------------
# 3️⃣ Build vector index
# -----------------------------------------------------
def build_index(chunks, embeddings):
    """Creates a new Chroma vector store and embeds chunks."""
    # Clear old Chroma database for a clean build
    if os.path.exists(PERSIST_DIR):
        print("🧹 Removing old Chroma database...")
        shutil.rmtree(PERSIST_DIR)

    print("🏗️ Building new Chroma vector store...")
    db = Chroma(
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )

    print(f"📦 Adding {len(chunks)} chunks...")
    filtered_chunks = [c for c in chunks if c.page_content and c.page_content.strip()]

    if not filtered_chunks:
        raise ValueError("No valid text chunks to embed.Please check your source docs")
    
    db.add_documents(filtered_chunks)
    print(f"✅Added {len(filtered_chunks)} non-empty chunks.")



# -----------------------------------------------------
# 4️⃣ Main pipeline
# -----------------------------------------------------
def main():
    print("📥 Loading documents...")
    docs = load_docs(data_dir="data")
    print(f"✅ Loaded {len(docs)} documents")

    print("🔪 Chunking documents...")
    chunks = chunk_docs(docs)
    print(f"✅ Created {len(chunks)} chunks")

    print("🧩 Sample chunk preview:")
    for c in chunks[:3]:
        print(f"- {c.page_content[:200]!r}")


    print("🧬 Generating embeddings & building index...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    build_index(chunks, embeddings)

    # Verify index metadata
    db = Chroma(
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )
    print(f"📊 Collection Count: {db._collection.count()}")
    print(f"📋 Metadata: {db._collection.metadata}")

    print("\n✅ Embedding & ingestion complete!")

# -----------------------------------------------------
# 5️⃣ Run
# -----------------------------------------------------
if __name__ == "__main__":
    main()
