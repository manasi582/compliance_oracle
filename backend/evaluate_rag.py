# =====================================================
# backend/evaluate_rag.py  —  Clean & Stable Version
# =====================================================

import os, time, json, logging, warnings, pathlib
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, faithfulness, context_recall, answer_correctness





# =====================================================
# 1️⃣ Environment Setup
# =====================================================

project_root = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(project_root / ".env")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message="InstructorLLM object has no attribute 'agenerate_prompt'")
logging.getLogger("ragas").setLevel(logging.ERROR)

print("✅ OPENAI_API_KEY loaded:", bool(os.getenv("OPENAI_API_KEY")))
print("✅ GROQ_API_KEY loaded:", bool(os.getenv("GROQ_API_KEY")))

# =====================================================
# 2️⃣ Create RAG Pipeline
# =====================================================

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap, RunnablePassthrough

def create_rag_pipeline():
    print("🔗 Connecting to Chroma vector store...")

    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from langchain_openai import ChatOpenAI

    # ✅ use same embedding model you used during ingestion
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory="embeddings/chroma", collection_name="compliance_docs_v2", embedding_function=embeddings)
    # ✅ Improved retrieval: k=8 for better recall, MMR for diversity to reduce redundancy
    retriever = db.as_retriever(
        search_type="mmr",  # Maximal Marginal Relevance for diverse results
        search_kwargs={
            "k": 8,  # Increased from 5 to 8 for better recall
            "fetch_k": 20,  # Fetch more candidates for MMR to choose from
            "lambda_mult": 0.7  # Balance between relevance (1.0) and diversity (0.0)
        }
    )

    # ✅ use Groq or OpenAI depending on what's available (prioritize Groq to avoid quota issues)
    if os.getenv("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=os.getenv("GROQ_API_KEY"))
        print("🧠 Using Groq Llama 3.1-8b-instant model.")
    elif os.getenv("OPENAI_API_KEY"):
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        print("🧠 Using OpenAI GPT-4o-mini model.")

    # ✅ Enhanced prompt with better instructions and structure
    prompt = ChatPromptTemplate.from_template("""
    You are an expert compliance assistant with deep knowledge of regulatory frameworks and corporate policies.
    
    INSTRUCTIONS:
    1. Carefully analyze the provided context documents
    2. Answer the question using ONLY information from the context
    3. If the context doesn't contain the answer, clearly state "Based on the provided context, this information is not available."
    4. Be precise, concise, and formal in your response
    5. Cite specific policies or regulations when applicable
    6. Structure your answer clearly with bullet points or numbered lists when appropriate
    
    CONTEXT DOCUMENTS:
    {context}
    
    QUESTION:
    {question}
    
    ANSWER:
    """)

    # ✅ Enhanced retrieval pipeline with document scoring and formatting
    def retrieve_and_format(question: str):
        docs = retriever.invoke(question)
        # Format context with document numbers for better structure
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"[Document {i}]\n{doc.page_content}")
        context = "\n\n".join(context_parts)
        return {"context": context, "question": question}

    rag_chain = (
        retrieve_and_format
        | prompt
        | llm
        | StrOutputParser()
    )

    print("✅ RAG pipeline built successfully.")
    return rag_chain, retriever


# =====================================================
# 3️⃣ Prepare Dataset
# =====================================================

def prepare_dataset(rag_chain, retriever, csv_path, output_path="backend/output_ragas_ready.csv"):
    """Fetches contexts + answers and prepares a RAGAS-compatible dataset."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path, quotechar='"', escapechar='\\', encoding="utf-8")

    if "Question" not in df.columns or "Expected_Answer" not in df.columns:
        raise ValueError("CSV must have 'Question' and 'Expected_Answer' columns.")

    df.rename(columns={
        "Question": "question",
        "Expected_Answer": "ground_truth"
    }, inplace=True)

    df["contexts"] = [[] for _ in range(len(df))]
    df["answer"] = ""

    print("\n📚 Generating answers for evaluation dataset...")

    for i, row in tqdm(df.iterrows(), total=len(df)):
        q = row["question"]
        contexts = []

        # 🔍 Retrieve context safely
        try:
            if hasattr(retriever, "get_relevant_documents"):
                docs = retriever.get_relevant_documents(q)
            elif hasattr(retriever, "invoke"):
                docs = retriever.invoke(q)
            elif callable(retriever):
                docs = retriever(q)
            else:
                raise AttributeError("Retriever has no recognized retrieval method.")
            contexts = [d.page_content for d in docs]
        except Exception as e:
            print(f"⚠️ Retrieval failed for Q{i+1}: {e}")

        # 🧠 Generate model answer
        try:
            answer = rag_chain.invoke(q)
        except Exception as e:
            print(f"⚠️ Generation failed for Q{i+1}: {e}")
            answer = ""

        df.at[i, "contexts"] = contexts
        df.at[i, "answer"] = answer
        time.sleep(1)

    df.to_csv(output_path, index=False)
    print(f"✅ Prepared dataset saved to {output_path}")

    return Dataset.from_pandas(df[["question", "contexts", "answer", "ground_truth"]])


# =====================================================
# 4️⃣ Run RAGAS Evaluation
# =====================================================

def run_ragas(dataset, save_path="backend/ragas_metrics.json", dev_mode=False):
    """Runs RAGAS metrics and saves results."""
    print("\n📊 Running RAGAS evaluation...")

    # 🧩 Limit for dev cycles
    if dev_mode:
        dataset = dataset.select(range(min(5, len(dataset))))
        print("⚙️ DEV_MODE active — evaluating only 5 samples.\n")

    # ✅ use Groq or OpenAI for evaluation (prioritize Groq to avoid quota issues)
    if os.getenv("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=os.getenv("GROQ_API_KEY"))
        print("🧠 Using Groq for RAGAS evaluation.")
    elif os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        print("🧠 Using OpenAI for RAGAS evaluation.")

    # ✅ Initialize embeddings for RAGAS to avoid default OpenAI usage
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    try:
        results = evaluate(
            dataset=dataset,
            metrics=[context_precision, faithfulness, context_recall, answer_correctness],
            llm=llm,
            embeddings=embeddings,
            batch_size=2,
            raise_exceptions=False
        )
    except Exception as e:
        print(f"❌ RAGAS evaluation failed: {e}")
        return

    print("\n✅ RAGAS evaluation complete.\n📈 Metrics Summary:")
    print("-" * 45)

    # Extract and normalize results
    # Extract and normalize results
    metrics_dict = {}
    try:
        # Try to get the results as a dictionary first
        if hasattr(results, "__dict__"):
            # Check if results has a scores attribute or similar
            if hasattr(results, "scores"):
                metrics_dict = results.scores
            elif hasattr(results, "to_dict"):
                metrics_dict = results.to_dict()
            else:
                # Try to convert to pandas and get mean
                try:
                    df_res = results.to_pandas()
                    # Only calculate mean for numeric columns
                    numeric_cols = df_res.select_dtypes(include=['float64', 'int64']).columns
                    metrics_dict = df_res[numeric_cols].mean().to_dict()
                except Exception as e:
                    print(f"⚠️ Could not convert to pandas: {e}")
                    # Try to access as dict
                    if isinstance(results, dict):
                        metrics_dict = results
        elif isinstance(results, dict):
            metrics_dict = results
        else:
            print(f"⚠️ Unknown results type: {type(results)}")
            print(f"Results object: {results}")

        # Display metrics
        if metrics_dict:
            for metric, score in metrics_dict.items():
                emoji = {
                    "context_precision": "🎯",
                    "faithfulness": "🧠",
                    "context_recall": "📚",
                    "answer_correctness": "✅",
                }.get(metric, "•")
                if isinstance(score, (float, int)):
                    print(f"{emoji}  {metric:<22} → {score:.3f}")
                else:
                    print(f"{emoji}  {metric:<22} → N/A")
        else:
            print("⚠️ No metrics found in results")
            # Try to print raw results for debugging
            print(f"\nRaw results type: {type(results)}")
            if hasattr(results, "to_pandas"):
                df_res = results.to_pandas()
                print(f"\nDataFrame columns: {df_res.columns.tolist()}")
                print(f"\nDataFrame head:\n{df_res.head()}")
                # Calculate mean for numeric columns only
                numeric_cols = df_res.select_dtypes(include=['float64', 'int64']).columns
                if len(numeric_cols) > 0:
                    print(f"\nNumeric metrics:")
                    for col in numeric_cols:
                        mean_val = df_res[col].mean()
                        emoji = {
                            "context_precision": "🎯",
                            "faithfulness": "🧠",
                            "context_recall": "📚",
                            "answer_correctness": "✅",
                        }.get(col, "•")
                        print(f"{emoji}  {col:<22} → {mean_val:.3f}")
                        metrics_dict[col] = mean_val
    except Exception as e:
        print(f"⚠️ Could not display metrics: {e}")
        import traceback
        traceback.print_exc()

    with open(save_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"\n💾 Metrics saved to {save_path}")


# =====================================================
# 5️⃣ Main
# =====================================================

def main():
    """End-to-end RAG evaluation pipeline."""
    EVAL_PATH = "data/evaluation_dataset.csv"
    OUTPUT_PATH = "backend/output_ragas_ready.csv"
    METRICS_PATH = "backend/ragas_metrics.json"
    DEV_MODE = True  # 🧩 Enable small test runs

    print("\n📁 Evaluation pipeline started...")

    # Step 1: Build RAG pipeline
    rag_chain, retriever = create_rag_pipeline()

    # Step 2: Test query
    print("\n🧠 Testing sample query...")
    try:
        test_query = "What is the company's data retention policy?"
        result = rag_chain.invoke(test_query)
        print("💬", result[:400], "...\n")
    except Exception as e:
        print(f"⚠️ Test query failed: {e}")
        return

    # Step 3: Prepare dataset
    try:
        dataset = prepare_dataset(rag_chain, retriever, EVAL_PATH, OUTPUT_PATH)
    except Exception as e:
        print(f"❌ Failed to prepare dataset: {e}")
        return

    # Step 4: Run RAGAS
    run_ragas(dataset, save_path=METRICS_PATH, dev_mode=DEV_MODE)

    print("\n🏁 Evaluation pipeline completed successfully.\n")


# =====================================================
# 6️⃣ Entry Point
# =====================================================

if __name__ == "__main__":
    main()
