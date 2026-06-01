# Architecture: NTCIR R2C2 Passage Retrieval Pipeline

## System overview

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│  Topic XML      │     │  Passage corpus      │     │  SQLite + FTS5      │
│  (qID + query)  │     │  *.jsonl.gz per doc  │────▶│  passages + fts     │
└────────┬────────┘     └──────────────────────┘     └──────────┬──────────┘
         │                                                       │
         │              ┌──────────────────────┐                 │
         └─────────────▶│ pr_submission_       │◀────────────────┘
                        │ pipeline.py          │
                        │  • BM25 candidates   │
                        │  • Score fusion      │
                        │  • Write PO-1…4      │
                        └──────────┬───────────┘
                                   │
                        ┌──────────▼───────────┐
                        │ validate_submission  │
                        │ + manifest (SHA256)  │
                        └──────────────────────┘
```

## Components

### 1. Indexing (`create_index`)

- **Input**: gzipped JSONL passages with `doc_id`, `chunk_id`, `title`, `text`.
- **Storage**: SQLite table `passages` + FTS5 virtual table `passages_fts` (content-synced with main table).
- **Tokenization**: `porter unicode61` — English-friendly stemming for recall.
- **Passage ID**: `{doc_id}_{chunk_id}`; submission uses `doc_id` as the returned identifier per task format.
- **Incremental**: Skips rebuild if rows exist unless `--rebuild_index`.

**Why FTS5 instead of in-memory BM25?**  
The R2C2 corpus is too large to hold tokenized postings in RAM on a laptop. FTS5 gives sub-second candidate retrieval with a single on-disk index.

### 2. Candidate generation (`bm25_candidates`)

- Query → OR-joined FTS query after punctuation normalization.
- SQL uses `bm25(passages_fts, title_weight, text_weight)` with **title_weight=1.6**, **text_weight=1.0** (titles are strong entity anchors in this task).
- Retrieves top `candidate_k` (default 500) per query.

### 3. Scoring runs (four variants)

All scores are **min-max normalized** per query before fusion so scales are comparable.

| Key | Formula (conceptually) | Purpose |
|-----|------------------------|---------|
| PO-1 | BM25 only | Strong lexical baseline |
| PO-2 | 0.65·BM25 + 0.35·RM3 | Pseudo-relevance: expand query from top feedback docs |
| PO-3 | 0.55·BM25 + 0.45·dense_proxy | Lexical + cheap semantic signal without GPU models |
| PO-4 | Rerank top `rerank_k` from PO-3 ordering | 0.25·BM25 + 0.55·dense_proxy + 0.20·coverage |

**RM3-style boost** (`rm3_boost`): From top feedback documents, collect frequent terms, measure token overlap with expanded set — classic PRF without a second retrieval pass.

**Dense proxy** (`dense_proxy_scores`): No transformer inference at submission time. Uses:
- Token-level cosine similarity (bag-of-words on stemmed-ish tokens)
- Character trigram Jaccard on query vs passage prefix

This trades model quality for **speed and reproducibility** while still diversifying PO-3/PO-4 from pure BM25.

**PO-4 reranking**: Re-score only the top `rerank_k` (default 120) PO-3 candidates with a denser proxy on full rerank window plus **query term coverage** in passage text.

### 4. Output (`write_run`)

NTCIR line format per ranked passage:

```
qid;rank;doc_id;safe_passage_text
```

- 20 results per query (`--per_query`)
- Text sanitized (no semicolons/newlines that break the format)

### 5. Validation

- **Inline** in pipeline: qid coverage, row counts, malformed lines.
- **`validate_submission.py`**: Pre-upload checks + zip member names (`{TEAM}-PO-1` … `PO-4`, `{TEAM}-PR.zip`).

### 6. Reproducibility manifest

`submission_manifest.json` records hyperparameters and SHA256 of each run file and zip — useful for ablation tracking and team handoff.

## Earlier prototypes (also in repo)

| File | Role |
|------|------|
| `bm25_retriever1.py` | Loads all passages into RAM, `rank_bm25` + optional SentenceTransformer rerank — good for small dry runs |
| `build_index.py` | Whoosh indexer over `wikipedia_passages/` — superseded by FTS5 approach |

## Design decisions (talking points)

1. **Single index, multiple runs** — One expensive index build; four scoring functions share the same candidate pool (efficient experimentation).
2. **Candidate → rerank** — Full corpus search only once; expensive signals only on top-*k*.
3. **No neural model in final pipeline** — Avoids GPU dependency and version lock-in for competition submission; semantic proxy is interpretable.
4. **Deterministic ranking** — Tie-break on `doc_id` so runs are byte-stable for hashing.
5. **Separation of concerns** — Indexing, retrieval, validation, and packaging are one script but logically distinct functions (easy to unit test or swap BM25 for dense retriever later).

## Possible extensions (if asked “what’s next?”)

- Replace `dense_proxy` with cross-encoder or bi-encoder reranking on PO-3/PO-4 candidates only.
- Learn fusion weights on dev qrels.
- Query rewriting / HyDE before FTS match.
- Shard FTS index or use Pyserini/Lucene for distributed scale.
