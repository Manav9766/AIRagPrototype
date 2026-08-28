# AI RAG Prototype (Python)

## Overview
This project demonstrates a simple Retrieval-Augmented Generation (RAG) pipeline that combines large language models with vector-based document retrieval.

## Architecture
1. Documents are loaded and chunked.
2. Chunks are converted into vector embeddings.
3. Embeddings are stored in a local FAISS vector database.
4. User queries retrieve the most relevant chunks.
5. The LLM generates an answer using only the retrieved context.

## Tech Stack
- Python
- LangChain
- OpenAI API
- FAISS (vector database)
- PyPDF

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the OpenAI API key

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
```

The `.env` file is ignored by Git and should never be committed.

## Usage

A sample PDF is included at `data/sample_docs.pdf`.

### Build the vector store

```bash
python ingest.py
```

The ingestion script validates that the source file exists, is a PDF, contains readable pages, and produces text chunks before creating the local `vector_store/` directory.

### Ask questions

```bash
python query.py
```

Enter questions about the ingested document. Type `exit` or `quit` to end the session.

The query flow retrieves the top matching chunks from FAISS and instructs the model to answer only from that retrieved context. If the context does not contain enough information, the prompt directs the model to say so rather than invent an answer.

## Purpose
This prototype simulates how internal documents can be queried safely and efficiently using LLMs in an applied engineering setting.

## What this demonstrates
- Applied LLM integration
- Vector embeddings and semantic search
- Retrieval-Augmented Generation (RAG)
- Hallucination-aware prompt design
- Input validation for document ingestion
- Local vector-store persistence with FAISS

## Notes
The project focuses on AI integration and retrieval pipelines rather than model training.
