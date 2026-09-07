import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

VECTOR_STORE_PATH = Path("vector_store")
RETRIEVAL_K = 4


def build_prompt(context: str, question: str) -> str:
    """Build a grounded prompt for retrieval-augmented answers."""
    return f"""
You are an assistant answering questions using ONLY the context below.
If the answer is not available in the context, say that the context does not contain enough information.

Context:
{context}

Question:
{question}

Answer:
"""


def ask_question(question: str) -> str:
    """Retrieve relevant document chunks and generate a context-grounded answer."""
    cleaned_question = question.strip()
    if not cleaned_question:
        return "Please enter a question."

    if not VECTOR_STORE_PATH.exists():
        raise FileNotFoundError(
            f"Vector store not found at '{VECTOR_STORE_PATH}'. "
            "Run 'python ingest.py' before querying."
        )

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not configured. Copy .env.example to .env "
            "and add your OpenAI API key before querying."
        )

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local(
        str(VECTOR_STORE_PATH),
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = vector_store.similarity_search(cleaned_question, k=RETRIEVAL_K)
    context = "\n\n".join(doc.page_content for doc in docs)

    llm = OpenAI(temperature=0)
    prompt = build_prompt(context, cleaned_question)

    return llm.invoke(prompt)


if __name__ == "__main__":
    while True:
        q = input("Ask a question (or 'exit'): ")
        if q.strip().lower() in {"exit", "quit"}:
            break
        print("\nAnswer:", ask_question(q), "\n")
