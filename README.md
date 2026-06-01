# NTCIR R2C2 Passage Retrieval (KRRISH)

Portfolio-ready implementation of a **passage retrieval** system for [NTCIR R2C2](https://research.nii.ac.jp/ntcir/) (Reading Comprehension over multi-domain corpora). This repository documents the **architecture, retrieval pipeline, and evaluation workflow** — not the passage corpus or pre-built indexes.

## What this project does

Given natural-language **questions** (topics), the system ranks **passages** from Wikipedia and Wookieepedia-style corpora and produces four official-style runs (PO-1 … PO-4) with increasing retrieval sophistication:

| Run | Approach |
|-----|----------|
| **PO-1** | Tuned BM25 over SQLite FTS5 (title + text weighting) |
| **PO-2** | BM25 + RM3-style pseudo-relevance feedback |
| **PO-3** | Hybrid lexical BM25 + lightweight semantic proxy (token cosine + char n-grams) |
| **PO-4** | PO-3 + focused reranking with query-term coverage on top candidates |

## Repository layout

```
NTCIR_R2C2_KRRISH/
  pr_submission_pipeline.py   # End-to-end index + retrieve + package runs
  validate_submission.py      # Format checks before NTCIR upload
  bm25_retriever1.py          # Earlier in-memory BM25 prototype (educational)
  build_index.py              # Earlier Whoosh-based indexer (superseded)
  requirements.txt
  ARCHITECTURE.md             # Deep dive for interviews
  topics/r2c2topics.xml       # Sample topic file (queries only)
```

## Quick start (code only)

You need the **official R2C2 passage files** locally (`.jsonl.gz` under `wikipedia_passages/` and `wookieepedia_passages/`). These are **not** in this repo.

```bash
cd NTCIR_R2C2_KRRISH
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python3 pr_submission_pipeline.py \
  --team_name KRRISH \
  --topics topics/r2c2topics.xml \
  --passages_dirs /path/to/wikipedia_passages /path/to/wookieepedia_passages \
  --db_path r2c2_passages_fts.db \
  --output_dir submission_output \
  --qid_mode preserve

python3 validate_submission.py \
  --submission_dir submission_output \
  --topics topics/r2c2topics.xml \
  --team_name KRRISH
```

The FTS index (`r2c2_passages_fts.db`) is created on first run and reused afterward (~5GB for the full corpus — keep it local, gitignored).

## How to explain this in an interview

1. **Problem**: Multi-hop, domain-mixed QA requires retrieving the right *passage*, not just a document — classic IR with strict submission format.
2. **Indexing choice**: SQLite FTS5 for scalable lexical search without loading the full corpus into RAM; Porter + `unicode61` tokenization; separate title/text BM25 weights.
3. **Retrieval cascade**: Cheap BM25 over millions of passages → normalize scores → combine with RM3 or semantic proxy → rerank only top-*k* for PO-4 (latency vs quality tradeoff).
4. **Engineering**: Deterministic tie-breaking, built-in validation, manifest with SHA256 hashes for reproducible submissions.
5. **Evolution**: `bm25_retriever1.py` (in-memory `rank_bm25`) → `build_index.py` (Whoosh) → `pr_submission_pipeline.py` (production FTS + four diverse runs).

See **[NTCIR_R2C2_KRRISH/ARCHITECTURE.md](NTCIR_R2C2_KRRISH/ARCHITECTURE.md)** for diagrams, data flow, and design tradeoffs.

## What is intentionally excluded

- Passage corpora (`.jsonl.gz`)
- SQLite FTS database (`*.db`)
- NTCIR submission run files and zips
- Virtualenv / `source/` vendored packages

## License / data

NTCIR corpora and topics are subject to NTCIR task rules. Use this code for learning and portfolio discussion; obtain official data from the task organizers for reproduction.
