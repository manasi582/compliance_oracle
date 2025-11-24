# =====================================================
# backend/rag_pipeline.py — LangChain 1.0.8 + Groq API
# =====================================================


from dotenv import dotenv_values
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENV_PATH = os.path.join(BASE_DIR, ".env")

# Force parse
config = dotenv_values(ENV_PATH)
print("Parsed config:", config)

# Set manually to environment
os.environ["GROQ_API_KEY"] = config.get("GROQ_API_KEY", "")
print("Final key:", os.getenv("GROQ_API_KEY")[:8])



from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# -----------------------------------------------------
# 1️⃣ Load the existing Chroma vector DB
# -----------------------------------------------------
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return Chroma(
        persist_directory="embeddings/chroma",
        embedding_function=embeddings,
    )


# -----------------------------------------------------
# 2️⃣ Create the RAG pipeline manually (Runnable API)
# -----------------------------------------------------
def create_rag_chain():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = ChatPromptTemplate.from_template("""
    You are ComplianceOracle — an expert company compliance assistant.
    Use only the provided context to answer.
    If the answer isn’t found, say “I’m not sure about that based on current policies.”

    <context>
    {context}
    </context>

    Question: {question}
    """)

    # Function to format retrieved documents into a string
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Runnable-based RAG chain
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
# 3️⃣ Test query
# -----------------------------------------------------
if __name__ == "__main__":
    rag_chain = create_rag_chain()
    query = "What is the data privacy policy for remote work?"
    print(f"💬 Question: {query}\n")

    answer = rag_chain.invoke(query)
    print("🧠 Answer:\n", answer)
