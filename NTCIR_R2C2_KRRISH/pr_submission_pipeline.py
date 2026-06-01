#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import json
import os
import re
import sqlite3
import zipfile
from collections import Counter
from glob import glob
from math import sqrt
from typing import Dict, List, Set, Tuple

from tqdm import tqdm


def parse_topics(topics_path: str) -> List[Tuple[str, str]]:
    with open(topics_path, "r", encoding="utf-8") as f:
        raw = f.read()
    pairs = re.findall(r"<qID>(.*?)</qID>.*?<q>(.*?)</q>", raw, re.DOTALL)
    return [(qid.strip(), q.strip()) for qid, q in pairs]


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def apply_qid_mode(topics: List[Tuple[str, str]], qid_mode: str) -> List[Tuple[str, str]]:
    if qid_mode == "preserve":
        return topics
    if qid_mode == "sequential4":
        return [(f"{i:04d}", qtext) for i, (_qid, qtext) in enumerate(topics, start=1)]
    raise ValueError(f"Unsupported qid_mode: {qid_mode}")


def iter_passages(passages_dirs: List[str]):
    for base_dir in passages_dirs:
        files = glob(os.path.join(base_dir, "**", "*.jsonl.gz"), recursive=True)
        for path in tqdm(files, desc=f"Scanning {os.path.basename(base_dir)} files"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    doc_id = str(obj.get("doc_id", "")).strip()
                    chunk_id = str(obj.get("chunk_id", "0")).strip()
                    passage_id = f"{doc_id}_{chunk_id}"
                    title = str(obj.get("title", "")).strip()
                    text = str(obj.get("text", "")).strip()
                    if not doc_id or not text:
                        continue
                    yield passage_id, doc_id, os.path.basename(base_dir), title, text


def create_index(db_path: str, passages_dirs: List[str], rebuild: bool = False):
    if rebuild and os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA temp_store=MEMORY;")
    cur.execute("PRAGMA cache_size=-200000;")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS passages (
            passage_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            source TEXT NOT NULL,
            title TEXT,
            text TEXT NOT NULL
        );
        """
    )
    cur.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS passages_fts
        USING fts5(
            title,
            text,
            content='passages',
            content_rowid='rowid',
            tokenize='porter unicode61'
        );
        """
    )

    existing = cur.execute("SELECT COUNT(*) FROM passages;").fetchone()[0]
    if existing > 0 and not rebuild:
        print(f"Index already has {existing} passages. Reusing existing index.")
        conn.close()
        return

    insert_sql = """
        INSERT OR REPLACE INTO passages (passage_id, doc_id, source, title, text)
        VALUES (?, ?, ?, ?, ?);
    """
    rows = []
    batch_size = 2000
    total = 0

    for rec in iter_passages(passages_dirs):
        rows.append(rec)
        if len(rows) >= batch_size:
            cur.executemany(insert_sql, rows)
            conn.commit()
            total += len(rows)
            rows = []
            if total % 50000 == 0:
                print(f"Inserted {total} passages...")

    if rows:
        cur.executemany(insert_sql, rows)
        conn.commit()
        total += len(rows)

    print(f"Inserted total {total} passages. Building FTS index...")
    cur.execute(
        "INSERT INTO passages_fts(rowid, title, text) SELECT rowid, title, text FROM passages;"
    )
    conn.commit()
    print("FTS index ready.")
    conn.close()


def normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    mn = min(values)
    mx = max(values)
    if abs(mx - mn) < 1e-12:
        return [0.0 for _ in values]
    return [(v - mn) / (mx - mn) for v in values]


def clean_query_for_fts(q: str) -> str:
    q = re.sub(r"[^\w\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def build_or_query(q: str) -> str:
    cleaned = clean_query_for_fts(q)
    if not cleaned:
        return ""
    return " OR ".join(cleaned.split())


def safe_text_for_run(text: str) -> str:
    text = text.replace("\n", " ").replace("\r", " ")
    text = text.replace(";", ",")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_simple(text: str) -> List[str]:
    t = clean_query_for_fts(text.lower())
    stop = {"the", "and", "for", "with", "from", "that", "this", "was", "are", "were", "into"}
    return [x for x in t.split() if len(x) > 1 and x not in stop]


def bm25_candidates(
    conn: sqlite3.Connection,
    qtext: str,
    top_k: int,
    title_weight: float,
    text_weight: float,
) -> List[Dict]:
    cur = conn.cursor()
    query = build_or_query(qtext)
    if not query:
        return []
    sql = """
        SELECT p.doc_id, p.text, p.title, -bm25(passages_fts, ?, ?) AS score
        FROM passages_fts
        JOIN passages p ON p.rowid = passages_fts.rowid
        WHERE passages_fts MATCH ?
        ORDER BY score DESC
        LIMIT ?;
    """
    try:
        rows = cur.execute(sql, (title_weight, text_weight, query, top_k)).fetchall()
    except sqlite3.OperationalError:
        rows = cur.execute(
            sql, (title_weight, text_weight, clean_query_for_fts(qtext), top_k)
        ).fetchall()

    docs = []
    for doc_id, text, title, score in rows:
        docs.append(
            {
                "doc_id": doc_id,
                "text": text,
                "title": title or "",
                "bm25": float(score),
            }
        )
    return docs


def rm3_boost(query: str, docs: List[Dict], fb_docs: int = 20, fb_terms: int = 20) -> List[float]:
    if not docs:
        return []
    q_tokens = set(tokenize_simple(query))
    top_docs = sorted(docs, key=lambda d: d["bm25"], reverse=True)[: min(fb_docs, len(docs))]
    term_counts: Counter = Counter()
    for d in top_docs:
        term_counts.update(tokenize_simple(d["text"]))

    expand_terms = [t for t, _ in term_counts.most_common(fb_terms)]
    expanded = q_tokens.union(expand_terms)
    scores: List[float] = []
    for d in docs:
        d_tokens = tokenize_simple(d["text"])
        if not d_tokens:
            scores.append(0.0)
            continue
        overlap = sum(1 for t in d_tokens if t in expanded)
        scores.append(overlap / max(1, len(d_tokens)))
    return scores


def char_ngrams(text: str, n: int = 3) -> Set[str]:
    t = re.sub(r"\s+", " ", text.lower()).strip()
    if len(t) < n:
        return {t} if t else set()
    return {t[i : i + n] for i in range(len(t) - n + 1)}


def cosine_count(query_tokens: List[str], doc_tokens: List[str]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    q = Counter(query_tokens)
    d = Counter(doc_tokens)
    dot = sum(q[t] * d.get(t, 0) for t in q)
    qn = sqrt(sum(v * v for v in q.values())) + 1e-9
    dn = sqrt(sum(v * v for v in d.values())) + 1e-9
    return dot / (qn * dn)


def dense_proxy_scores(query: str, docs: List[Dict]) -> List[float]:
    if not docs:
        return []
    q_tokens = tokenize_simple(query)
    q_ngrams = char_ngrams(clean_query_for_fts(query), 3)
    scores: List[float] = []
    for d in docs:
        txt = d["text"][:1200]
        d_tokens = tokenize_simple(txt)
        d_ngrams = char_ngrams(txt, 3)
        cos = cosine_count(q_tokens, d_tokens)
        if not q_ngrams or not d_ngrams:
            jacc = 0.0
        else:
            inter = len(q_ngrams.intersection(d_ngrams))
            union = len(q_ngrams.union(d_ngrams))
            jacc = inter / max(1, union)
        scores.append((0.7 * cos) + (0.3 * jacc))
    return scores


def build_rankings(
    conn: sqlite3.Connection,
    topics: List[Tuple[str, str]],
    candidate_k: int,
    rerank_k: int,
) -> Dict[str, Dict[str, List[Dict]]]:
    out: Dict[str, Dict[str, List[Dict]]] = {}

    for qid, qtext in tqdm(topics, desc="Processing topics"):
        base_docs = bm25_candidates(conn, qtext, top_k=candidate_k, title_weight=1.6, text_weight=1.0)
        if not base_docs:
            out[qid] = {"po1": [], "po2": [], "po3": [], "po4": []}
            continue

        bm_n = normalize([d["bm25"] for d in base_docs])

        # PO-1: tuned BM25 baseline.
        po1_score = bm_n[:]

        # PO-2: BM25 + RM3-style expansion signal.
        rm3_n = normalize(rm3_boost(qtext, base_docs, fb_docs=25, fb_terms=20))
        po2_score = [(0.65 * b) + (0.35 * r) for b, r in zip(bm_n, rm3_n)]

        # PO-3: hybrid lexical + semantic proxy.
        dense_n = normalize(dense_proxy_scores(qtext, base_docs))
        po3_score = [(0.55 * b) + (0.45 * d) for b, d in zip(bm_n, dense_n)]

        # PO-4: hybrid + reranker (dense-heavy with query coverage on top candidates).
        top_for_rerank = sorted(
            range(len(po3_score)),
            key=lambda i: (po3_score[i], base_docs[i]["doc_id"]),
            reverse=True,
        )[: min(rerank_k, len(po3_score))]
        po4_score = po3_score[:]
        rerank_docs = [base_docs[i] for i in top_for_rerank]
        rerank_dense_n = normalize(dense_proxy_scores(qtext, rerank_docs))
        q_tokens = tokenize_simple(qtext)
        coverage_raw = [sum(1 for t in q_tokens if t in d["text"].lower()) for d in rerank_docs]
        coverage_n = normalize([float(x) for x in coverage_raw])
        for j, idx in enumerate(top_for_rerank):
            po4_score[idx] = (0.25 * bm_n[idx]) + (0.55 * rerank_dense_n[j]) + (0.20 * coverage_n[j])

        def rank_by(score_vec: List[float]) -> List[Dict]:
            # Secondary key keeps output deterministic when scores tie.
            order = sorted(
                range(len(score_vec)),
                key=lambda i: (score_vec[i], base_docs[i]["doc_id"]),
                reverse=True,
            )
            return [base_docs[i] for i in order]

        out[qid] = {
            "po1": rank_by(po1_score),
            "po2": rank_by(po2_score),
            "po3": rank_by(po3_score),
            "po4": rank_by(po4_score),
        }
    return out


def write_run(path: str, topics: List[Tuple[str, str]], rankings: Dict[str, List[Dict]]):
    with open(path, "w", encoding="utf-8") as out:
        for qid, _qtext in topics:
            docs = rankings[qid][:20]
            for rank, d in enumerate(docs, start=1):
                out.write(f"{qid};{rank};{d['doc_id']};{safe_text_for_run(d['text'])}\n")


def validate_outputs(
    topics: List[Tuple[str, str]],
    run_files: List[str],
    per_query: int,
) -> None:
    expected_qids = [qid for qid, _ in topics]
    expected_qid_set = set(expected_qids)

    for rf in run_files:
        seen_counts: Dict[str, int] = Counter()
        with open(rf, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                parts = line.rstrip("\n").split(";")
                if len(parts) < 4:
                    raise ValueError(f"{rf}: malformed line {line_no}")
                qid, rank, doc_id = parts[0], parts[1], parts[2]
                if qid not in expected_qid_set:
                    raise ValueError(f"{rf}: unknown qid '{qid}' on line {line_no}")
                if not rank.isdigit():
                    raise ValueError(f"{rf}: non-numeric rank '{rank}' on line {line_no}")
                if not doc_id:
                    raise ValueError(f"{rf}: empty doc_id on line {line_no}")
                seen_counts[qid] += 1

        missing = [qid for qid in expected_qids if qid not in seen_counts]
        if missing:
            raise ValueError(f"{rf}: missing {len(missing)} qids")

        wrong_k = [qid for qid, c in seen_counts.items() if c != per_query]
        if wrong_k:
            raise ValueError(
                f"{rf}: {len(wrong_k)} qids do not have exactly {per_query} rows"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Build 4 diverse PO runs: BM25, BM25+RM3, Hybrid, Hybrid+Reranker."
    )
    parser.add_argument("--team_name", default="KRRISH")
    parser.add_argument("--topics", default="dryruntopics.txt")
    parser.add_argument("--db_path", default="r2c2_passages_fts.db")
    parser.add_argument(
        "--passages_dirs",
        nargs="+",
        default=["wikipedia_passages", "wookieepedia_passages"],
    )
    parser.add_argument("--output_dir", default="submission_pr")
    parser.add_argument("--candidate_k", type=int, default=500)
    parser.add_argument("--rerank_k", type=int, default=120)
    parser.add_argument("--per_query", type=int, default=20)
    parser.add_argument(
        "--qid_mode",
        default="preserve",
        choices=["preserve", "sequential4"],
        help="preserve: keep qIDs from topic file; sequential4: force 0001,0002,...",
    )
    parser.add_argument("--rebuild_index", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    create_index(args.db_path, args.passages_dirs, rebuild=args.rebuild_index)

    topics = apply_qid_mode(parse_topics(args.topics), args.qid_mode)
    if len({qid for qid, _ in topics}) != len(topics):
        raise ValueError("Duplicate qIDs detected after qid_mode transformation.")

    conn = sqlite3.connect(args.db_path)
    rankings = build_rankings(conn, topics, candidate_k=args.candidate_k, rerank_k=args.rerank_k)
    conn.close()

    run_map = {
        "1": "po1",
        "2": "po2",
        "3": "po3",
        "4": "po4",
    }
    run_files = []
    for run_no, key in run_map.items():
        run_name = f"{args.team_name}-PO-{run_no}"
        out_path = os.path.join(args.output_dir, run_name)
        write_run(
            out_path,
            topics,
            {qid: rankings[qid][key] for qid, _ in topics},
        )
        run_files.append(out_path)
        print(f"Wrote {out_path}")

    zip_path = os.path.join(args.output_dir, f"{args.team_name}-PR.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rf in run_files:
            zf.write(rf, arcname=os.path.basename(rf))

    validate_outputs(topics, run_files, per_query=args.per_query)

    manifest = {
        "team_name": args.team_name,
        "topics_file": args.topics,
        "topics_count": len(topics),
        "qid_mode": args.qid_mode,
        "db_path": args.db_path,
        "candidate_k": args.candidate_k,
        "rerank_k": args.rerank_k,
        "per_query": args.per_query,
        "run_files": {os.path.basename(rf): file_sha256(rf) for rf in run_files},
        "submission_zip": {"path": zip_path, "sha256": file_sha256(zip_path)},
    }
    manifest_path = os.path.join(args.output_dir, "submission_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)

    print("Done.")
    print("Run files:")
    for rf in run_files:
        print(f" - {rf}")
    print(f"Submission zip: {zip_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
