# BM25 Retriever

This is a small script for passage retrieval using BM25. It also has a very simple RM3-style query expansion option and an optional dense reranking step.

## What it does

- Ranks passages with BM25 (`k1` and `b` are configurable)
- Can expand queries using RM3 over the top passages
- Can add a small score boost when query terms appear in the passage title
- Can rerank the top BM25 hits with a sentence-transformers model
- Writes output in a standard TREC run format

## Setup

Create and activate a virtual environment if you want, then:

```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords
```

## How to run

Basic example:

```bash
python bm25_retriever1.py \
  --passage_root wikipedia_passages \
  --queries dryruntopics.txt \
  --output run.txt \
  --top_k 1000
```

Some useful flags:

- `--rm3` turns on RM3-style query expansion
- `--title_boost` adds a fixed boost when titles match query terms
- `--use_dense_rerank` reranks top candidates with a dense model
- `--dense_weight` controls how much weight the dense score has (0–1)

For the full list of arguments, run:

```bash
python bm25_retriever1.py --help
```

