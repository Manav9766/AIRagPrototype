from dotenv import load_dotenv

from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def ask_question(question: str) -> str:
    # Load embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local(
        "vector_store",
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Retrieve relevant documents
    docs = vector_store.similarity_search(question, k=4)

    # Build context from retrieved docs
    context = "\n\n".join(doc.page_content for doc in docs)

    # Create LLM
    llm = OpenAI(temperature=0)

    # Final prompt (manual RAG)
    prompt = f"""
You are an assistant answering questions using ONLY the context below.

Context:
{context}

Question:
{question}

Answer:
"""

    return llm.invoke(prompt)

if __name__ == "__main__":
    while True:
        q = input("Ask a question (or 'exit'): ")
        if q.lower() == "exit":
            break
        print("\nAnswer:", ask_question(q), "\n")
