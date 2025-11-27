
# Compliance Bot

A **Retrieval-Augmented Generation (RAG)** system designed to answer **compliance-related questions** by retrieving and synthesizing information from internal regulatory documents.

This project leverages **LangChain**, **ChromaDB**, and **RAGAS** for document processing, retrieval, generation, and evaluation.



## Features

- **Document Ingestion** – Supports PDF, DOCX, and TXT files
- **Vector Search** – Uses ChromaDB with **MMR (Maximal Marginal Relevance)** for diverse and relevant retrieval 
- **LLM Integration** – Compatible with **OpenAI (GPT-4o)** and **Groq (Llama 3.1)**  
- **Evaluation Pipeline** – Integrated **RAGAS metrics**:
    i)Faithfulness  
    ii)Answer Correctness  
    iii)Context Precision / Recall 


  

## Setup & Installation

Follow these steps to set up the project locally.

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/manasi582/compliance_oracle.git
cd compliance_oracle 
```
### 2️⃣ Create a Virtual Environment
Mac/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
Windows: 
```bash
python -m venv .venv
.venv\\Scripts\\activate
```
### 3️⃣ Install Dependencies
```bash
pip install langchain langchain-community langchain-chroma langchain-openai langchain-groq langchain-huggingface
pip install chromadb ragas pandas python-dotenv tqdm datasets pypdf unstructured python-docx networkx
```
or 
```bash
pip install -r requirements.txt
```     
### 4️⃣ Configure Environment Variables
Create a .env file in the root directory:
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...

## USAGE
1️⃣ Ingest Documents
python backend/ingest_data.py

2️⃣ Run the RAG Pipeline (Interactive)
python backend/rag_pipeline.py

3️⃣ Evaluate the Model
python backend/evaluate_rag.py

Results will be saved to backend/ragas_metrics.json and backend/output_ragas_ready.csv

## Project Structure
compliance_oracle/
├── backend/
│   ├── ingest_data.py
│   ├── evaluate_rag.py
│   └── rag_pipeline.py
├── data/
│   ├── evaluation_dataset.csv
│   └── (source documents)
├── embeddings/
├── .env
└── .gitignore

## Evaluation Metrics (RAGAS)
Faithfulness – Consistency with retrieved context
Answer Correctness – Semantic accuracy vs ground truth
Context Precision – Relevant context retrieved
Context Recall – Completeness of retrieved context

## Tech Stack
LangChain | ChromaDB | RAGAS | OpenAI | Groq | Python

## Requirements
Python 3.9+
API key for OpenAI or Groq
~2GB free disk space

## Contributing
Feel free to submit a PR or open an issue for suggestions or bugs.

## Acknowledgements
LangChain, ChromaDB, RAGAS, OpenAI, Groq

#### ⚡ "AI won’t replace you — but someone who knows how to use AI will."


