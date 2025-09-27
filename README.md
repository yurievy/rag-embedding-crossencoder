# RAG Embedding + Cross-Encoder Pipeline

This repository contains a **RAG (Retrieval-Augmented Generation) pipeline** with the following components:

- **Embedding-based retriever**: uses `intfloat/multilingual-e5-base` for semantic search.  
- **Cross-encoder reranker**: uses `jinaai/jina-reranker-v2-base-multilingual` to rerank top candidates.  
- **Generative model**: uses `qwen2.5-1.5b-instruct-q5_k_m.gguf` for answer generation based on retrieved documents.
- **Web server**: uses Flask to provide a simple API and web interface for interacting with the pipeline.

## Usage

1. Clone repository:

```bash
git clone https://github.com/username/rag-embedding-crossencoder.git
```

2. Navigate to the project folder:

```bash
cd rag-embedding-crossencoder
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the pipeline:

```bash
python rag-pipeline.py
```
