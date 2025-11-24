# verify_index.py

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load the existing Chroma DB
vs = Chroma(
    persist_directory="embeddings/chroma",
    embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
)

# Test a query
query = "What is the policy on remote work?"
results = vs.similarity_search(query, k=2)

for i, doc in enumerate(results, 1):
    print(f"\n[{i}] Source:", doc.metadata)
    print(doc.page_content[:250], "...")
