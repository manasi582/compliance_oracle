# =====================================================
# backend/rag_pipeline.py — Compliance Oracle Core
# =====================================================

import os
import time
from dotenv import dotenv_values, load_dotenv

# -----------------------------------------------------
# 1️⃣ Setup & Configuration
# -----------------------------------------------------
# Define paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENV_PATH = os.path.join(BASE_DIR, ".env")

# Load environment variables
load_dotenv(ENV_PATH)

# Set Groq API Key explicitly if needed
config = dotenv_values(ENV_PATH)
if "GROQ_API_KEY" in config:
    os.environ["GROQ_API_KEY"] = config["GROQ_API_KEY"]

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# -----------------------------------------------------
# 2️⃣ The Retriever (Finds relevant info)
# -----------------------------------------------------
def load_vectorstore():
    """Loads the knowledge base."""
    # We use the same model used for ingestion
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    # Connect to the existing ChromaDB with the correct collection name
    return Chroma(
        persist_directory="embeddings/chroma",
        collection_name="compliance_docs_v2",  # IMPORTANT: Must match ingest_data.py
        embedding_function=embeddings,
    )


# -----------------------------------------------------
# 3️⃣ The Generator (Answers using the info)
# -----------------------------------------------------
def create_rag_chain():
    """Builds the complete RAG pipeline."""
    
    # A. Retriever: Finds top 5 relevant documents
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # B. Generator: Large Language Model with fallback
    # Priority: Groq (fast, free) → OpenAI (reliable fallback)
    if os.getenv("GROQ_API_KEY"):
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0,
                api_key=os.getenv("GROQ_API_KEY")
            )
            print("🧠 Using Groq Llama 3.1-8b-instant model.")
        except Exception as e:
            print(f"⚠️ Groq initialization failed: {e}")
            if os.getenv("OPENAI_API_KEY"):
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                print("🧠 Falling back to OpenAI GPT-4o-mini model.")
            else:
                raise ValueError("No working LLM API found. Please check your API keys.")
    elif os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        print("🧠 Using OpenAI GPT-4o-mini model.")
    else:
        raise ValueError("No LLM API key found. Please set GROQ_API_KEY or OPENAI_API_KEY in .env")

    # C. Prompt: Instructions for the AI (Hybrid approach)
    template = """
    You are ComplianceOracle, a helpful corporate compliance expert.
    
    INSTRUCTIONS:
    1. First, check if the context below contains relevant information to answer the question.
    2. If the context has relevant information, answer based on it and cite the source.
    3. If the context is not relevant or insufficient, you may provide a general answer based on your knowledge of compliance best practices, BUT add this disclaimer at the end:
       "⚠️ Note: This answer is based on general compliance knowledge, not your organization's specific policies."
    4. Keep your answer professional and concise.

    <context>
    {context}
    </context>

    Question: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)

    # D. Chain: Connects Retriever -> Prompt -> LLM -> Output
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# -----------------------------------------------------
# 4️⃣ Main Interactive Loop
# -----------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🤖 Compliance Oracle - Interview Demo Mode")
    print("="*50)
    print("Initializing system... (Loading AI models)")
    
    try:
        rag_chain = create_rag_chain()
        print("✅ System Ready! Ask a question about company policies.")
        print("(Type 'exit' or 'quit' to stop)\n")

        while True:
            query = input("❓ Your Question: ").strip()
            
            if query.lower() in ["exit", "quit"]:
                print("\n👋 Exiting demo. Good luck with the interview!")
                break
            
            if not query:
                continue

            print("\n   🔍 Retrieving relevant policy documents...")
            time.sleep(0.5) # Simulating "thinking" for effect
            
            print("   🧠 Generating answer...")
            start_time = time.time()
            
            # Run the RAG pipeline
            answer = rag_chain.invoke(query)
            
            elapsed = time.time() - start_time
            print(f"   ✨ Update processed in {elapsed:.2f}s\n")
            
            print(f"📝 Answer: \n{answer}\n")
            print("-" * 50 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Tip: Check your .env file or internet connection.")
