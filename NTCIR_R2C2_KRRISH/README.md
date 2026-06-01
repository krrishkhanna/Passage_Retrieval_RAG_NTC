# NTCIR R2C2 PR Pipeline

Production passage-retrieval pipeline for NTCIR R2C2 **Passage Retrieval (PR)**. Generates four diverse runs from a shared SQLite FTS5 index.

See the [repository root README](../README.md) for portfolio context and [ARCHITECTURE.md](ARCHITECTURE.md) for system design.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Build runs

Place official passage directories on disk, then:

```bash
python3 pr_submission_pipeline.py \
  --team_name KRRISH \
  --topics topics/r2c2topics.xml \
  --db_path r2c2_passages_fts.db \
  --passages_dirs /path/to/wikipedia_passages /path/to/wookieepedia_passages \
  --output_dir submission_output \
  --qid_mode preserve
```

Outputs (gitignored by default):

- `KRRISH-PO-1` … `KRRISH-PO-4`
- `KRRISH-PR.zip`
- `submission_manifest.json`

## Validate

```bash
python3 validate_submission.py \
  --submission_dir submission_output \
  --topics topics/r2c2topics.xml \
  --team_name KRRISH \
  --per_query 20
```

## Scripts

| Script | Description |
|--------|-------------|
| `pr_submission_pipeline.py` | Index passages, score four runs, zip + manifest |
| `validate_submission.py` | Standalone submission format checker |
| `bm25_retriever1.py` | In-memory BM25 + optional dense rerank (prototype) |
| `build_index.py` | Whoosh indexer (legacy) |

## Key parameters

| Flag | Default | Meaning |
|------|---------|---------|
| `--candidate_k` | 500 | BM25 candidates per query |
| `--rerank_k` | 120 | Candidates rescored for PO-4 |
| `--rebuild_index` | off | Force FTS rebuild |
| `--qid_mode` | preserve | Keep topic qIDs from XML |

## Data not in Git

- `wikipedia_passages/`, `wookieepedia_passages/` — official corpus
- `r2c2_passages_fts.db` — built locally (~5GB)
