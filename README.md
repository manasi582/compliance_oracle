Compliance Oracle
A Retrieval-Augmented Generation (RAG) system designed to answer compliance-related questions by retrieving andThis project leverages LangChain, ChromaDB, and RAGAS for document processing, retrieval, generation, and evalua------------------------------------------------------------
Features
------------------------------------------------------------
- Document Ingestion - Supports PDF, DOCX, and TXT files
- Vector Search - Uses ChromaDB with MMR (Maximal Marginal Relevance)
- LLM Integration - OpenAI (GPT-4o) & Groq (Llama 3.1)
- Evaluation Pipeline - RAGAS metrics: Faithfulness, Answer Correctness, Context Precision/Recall
------------------------------------------------------------
Setup & Installation
------------------------------------------------------------
1. Clone the Repository
git clone https://github.com/manasi582/compliance_oracle.git
cd compliance_oracle
2. Create a Virtual Environment
Mac/Linux:
python3 -m venv .venv
source .venv/bin/activate
Windows:
python -m venv .venv
.venv\Scripts\activate
3. Install Dependencies
pip install langchain langchain-community langchain-chroma langchain-openai langchain-groq langchain-huggingface
pip install chromadb ragas pandas python-dotenv tqdm datasets pypdf unstructured python-docx networkx
4. Configure Environment Variables
Create a .env file in the root directory:
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
------------------------------------------------------------
Usage
------------------------------------------------------------
1. Ingest Documents
python backend/ingest_data.py
2. Run the RAG Pipeline (Interactive)
python backend/rag_pipeline.py
