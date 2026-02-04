# AI RAG Prototype (Python)

## Overview
This project demonstrates a simple Retrieval-Augmented Generation (RAG) pipeline
that combines large language models with vector-based document retrieval.

## Architecture
1. Documents are loaded and chunked
2. Chunks are converted into vector embeddings
3. Embeddings are stored in a FAISS vector database
4. User queries retrieve relevant chunks
5. LLM generates grounded answers using retrieved context

## Tech Stack
- Python
- LangChain
- OpenAI API
- FAISS (vector database)

## Purpose
This prototype simulates how internal documents can be queried safely and efficiently
using LLMs in an applied engineering setting.

## What this demonstrates
- Applied LLM integration
- Vector embeddings and semantic search
- Retrieval-Augmented Generation (RAG)
- Hallucination-aware prompt design

## Notes
The project focuses on AI integration and retrieval pipelines rather than model training.
