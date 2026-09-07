import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DEFAULT_SOURCE = Path("data/sample_docs.pdf")
VECTOR_STORE_PATH = "vector_store"


def ingest_documents(source_path: Path = DEFAULT_SOURCE) -> None:
    """Load a PDF, chunk it, embed it, and persist a FAISS vector store."""
    if not source_path.exists():
        raise FileNotFoundError(
            f"Source document not found: {source_path}. "
            "Add the PDF before running ingestion."
        )

    if source_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF source document, got: {source_path.name}")

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not configured. Copy .env.example to .env "
            "and add your OpenAI API key before running ingestion."
        )

    loader = PyPDFLoader(str(source_path))
    documents = loader.load()
    if not documents:
        raise ValueError(f"No readable pages were found in {source_path}.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError(f"No text chunks could be created from {source_path}.")

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)

    print(
        f"Ingested {len(documents)} page(s) into {len(chunks)} chunk(s) "
        f"and saved the vector store to '{VECTOR_STORE_PATH}'."
    )


if __name__ == "__main__":
    ingest_documents()
