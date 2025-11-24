# =====================================================
# backend/ingest_data.py
# =====================================================

import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv


from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()  # to read OPENAI_API_KEY from .env


# -----------------------------------------------------
# 1️⃣  File Readers
# -----------------------------------------------------
def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def read_pdf(path: Path) -> str:
    from pypdf import PdfReader
    text = []
    with open(path, "rb") as f:
        pdf = PdfReader(f)
        for page in pdf.pages:
            text.append(page.extract_text() or "")
    return "\n".join(text)

def read_json_faq(path: Path) -> str:
    """Flatten JSON Q&A into a readable block of text"""
    data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    items = data.get("questions", data if isinstance(data, list) else [])
    blocks = []
    for it in items:
        q = it.get("question") or it.get("Question") or ""
        a = it.get("answer") or it.get("Answer") or ""
        blocks.append(f"Q: {q}\nA: {a}")
    return "\n\n---\n\n".join(blocks)


# -----------------------------------------------------
# 2️⃣  Cleaning & Categorization
# -----------------------------------------------------
def clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def infer_category(path: Path) -> str:
    p = path.as_posix().lower()
    if "/hr_policies/" in p: return "HR"
    if "/it_policies/" in p: return "IT"
    if "/data_priv_manual/" in p: return "Data Privacy"
    if "/finance_policies/" in p: return "Finance"
    if "/faqs/" in p: return "FAQ"
    return "General"


# -----------------------------------------------------
# 3️⃣  Load All Documents
# -----------------------------------------------------
def load_all_docs(data_root="data"):
    docs = []
    for root, _, files in os.walk(data_root):
        for fname in files:
            path = Path(root) / fname
            ext = path.suffix.lower()
            if ext in [".txt", ".md"]:
                raw = read_txt(path)
            elif ext in [".pdf"]:
                raw = read_pdf(path)
            elif ext in [".json"]:
                raw = read_json_faq(path)
            else:
                continue  # skip unsupported

            text = clean_text(raw)
            if not text:
                continue

            docs.append({
                "text": text,
                "metadata": {
                    "category": infer_category(path),
                    "source_file": path.name,
                    "source_path": str(path.relative_to(Path(data_root)))
                }
            })
    return docs


# -----------------------------------------------------
# 4️⃣  Chunking  ✅ (This is where your chunking lives)
# -----------------------------------------------------
def chunk_documents(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunked = []
    for d in docs:
        for chunk in splitter.split_text(d["text"]):
            # ✅ Wrap chunks as Document objects
            chunked.append(
                Document(
                    page_content=chunk,
                    metadata=d["metadata"]
                )
            )
    return chunked



# -----------------------------------------------------
# 5️⃣  Embedding + Indexing
# -----------------------------------------------------
def build_index(chunks, persist_dir="embeddings/chroma"):
    """
    Build a local Chroma vector index using HuggingFace sentence-transformers.
    Works fully offline and shows progress.
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from tqdm import tqdm

    print("🧠 Loading local embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print(f"📦 Building vector index at {persist_dir} ...")

    # Create batches to avoid memory spikes
    batch_size = 200
    total_batches = (len(chunks) // batch_size) + 1
    vs = None

    for i in tqdm(range(total_batches), desc="Embedding chunks"):
        start = i * batch_size
        end = start + batch_size
        batch = chunks[start:end]
        if not batch:
            continue

        if vs is None:
            vs = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=persist_dir
            )
        else:
            vs.add_documents(batch)

    if vs:
        if hasattr(vs, "persist_client"):
            vs.persist_client()
        elif hasattr(vs, "persist"):
            vs.persist()
        print("✅ Local embeddings successfully built and saved to:", persist_dir)
    else:
        print("⚠️ No documents were indexed")

        
    return vs


# -----------------------------------------------------
# 6️⃣  Pipeline Entry Point
# -----------------------------------------------------
def main():
    print("📥 Loading documents...")
    docs = load_all_docs("data")
    print(f"✅ Loaded {len(docs)} documents")

    print("🔪 Chunking documents...")
    chunks = chunk_documents(docs)
    print(f"✅ Created {len(chunks)} chunks")

    print("🧬 Generating embeddings & building index...")
    build_index(chunks)
    print("✅ Embeddings saved to: embeddings/chroma")


if __name__ == "__main__":
    main()
