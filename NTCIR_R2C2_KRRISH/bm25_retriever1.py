#!/usr/bin/env python3
import gzip
import json
import os
import re
import argparse
import string
import heapq
from tqdm import tqdm
from rank_bm25 import BM25Okapi

# Optional dense rerank (can be skipped if the package is missing)
_dense_available = True
try:
    from sentence_transformers import SentenceTransformer, util
except Exception:
    _dense_available = False

# NLP helpers for basic tokenization and stemming
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

_stopset = set(stopwords.words("english"))
_stemmer = PorterStemmer()


def tokenize(text):
    # Lowercase and remove punctuation, then drop stopwords and stem
    text = text.lower()
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    toks = [t.strip() for t in text.split() if t.strip()]
    toks = [t for t in toks if t not in _stopset]
    toks = [_stemmer.stem(t) for t in toks]
    return toks


def load_passages(root_dir):
    passages = []
    passage_ids = []
    titles = []

    for root, dirs, files in os.walk(root_dir):
        for f in tqdm(files, desc="Reading passage files"):
            if not f.endswith(".jsonl.gz"):
                continue
            path = os.path.join(root, f)
            with gzip.open(path, "rt") as fh:
                for line in fh:
                    data = json.loads(line)
                    pid = f'{data.get("doc_id","DOC")}_{data.get("chunk_id",0)}'
                    passages.append(data.get("text", ""))
                    passage_ids.append(pid)
                    titles.append(data.get("title", ""))

    return passages, passage_ids, titles


def build_bm25(tokenized_corpus, k1=1.5, b=0.75):
    return BM25Okapi(tokenized_corpus, k1=k1, b=b)


def rm3_expand(query_tokens, tokenized_corpus, bm25, top_k=10, expand_terms=10):
    """
    Simple RM3-style expansion.
    Looks at the top documents under the current query,
    counts terms, and appends the most frequent ones.
    """
    scores = bm25.get_scores(query_tokens)
    top_idx = heapq.nlargest(top_k, range(len(scores)), key=lambda i: scores[i])

    term_counts = {}
    for idx in top_idx:
        for t in tokenized_corpus[idx]:
            term_counts[t] = term_counts.get(t, 0) + 1

    top_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)[:expand_terms]
    expanded = query_tokens + [t for t, _ in top_terms]
    return expanded


def main(args):
    print("Loading passages from:", args.passage_root)
    passages, passage_ids, titles = load_passages(args.passage_root)
    print("Total passages loaded:", len(passages))

    print("Tokenizing corpus...")
    tokenized_corpus = []
    for p in tqdm(passages, desc="Tokenizing passages"):
        tokenized_corpus.append(tokenize(p))

    print("Building BM25 index...")
    bm25 = build_bm25(tokenized_corpus, k1=args.k1, b=args.b)

    dense_model = None
    if args.use_dense_rerank:
        if not _dense_available:
            print("sentence-transformers not installed. Dense rerank disabled.")
            args.use_dense_rerank = False
        else:
            print("Loading dense model:", args.dense_model)
            dense_model = SentenceTransformer(args.dense_model)

    # Load queries
    with open(args.queries, "r") as f:
        raw = f.read()

    pattern = re.findall(r"<qID>(.*?)</qID>.*?<q>(.*?)</q>", raw, re.DOTALL)
    queries = [(qid.strip(), qtext.strip()) for qid, qtext in pattern]
    print("Total queries loaded:", len(queries))

    with open(args.output, "w") as out:
        for qid, qtext in tqdm(queries, desc="Processing Queries"):

            qtokens = tokenize(qtext)

            if args.rm3:
                qtokens = rm3_expand(qtokens, tokenized_corpus, bm25)

            scores = bm25.get_scores(qtokens)

            # Title boost
            if args.title_boost:
                qtset = set(qtokens)
                for i, title in enumerate(titles):
                    if not title:
                        continue
                    tkn = tokenize(title)
                    if qtset.intersection(tkn):
                        scores[i] += args.title_boost_amount

            # Faster Top-K selection
            top_indices = heapq.nlargest(
                args.top_k,
                range(len(scores)),
                key=lambda i: scores[i]
            )

            # Optional dense reranking
            if args.use_dense_rerank and dense_model is not None:
                q_emb = dense_model.encode(qtext, convert_to_tensor=True)

                cand_texts = [passages[i] for i in top_indices]
                cand_embs = dense_model.encode(
                    cand_texts,
                    batch_size=32,
                    show_progress_bar=True,
                    convert_to_tensor=True
                )

                sims = util.cos_sim(q_emb, cand_embs)[0].cpu().tolist()

                bm_vals = [scores[i] for i in top_indices]
                minv, maxv = min(bm_vals), max(bm_vals)
                norm_bm = [(v - minv) / (maxv - minv + 1e-9) for v in bm_vals]

                combined = [
                    (args.dense_weight * sims[j]) +
                    ((1 - args.dense_weight) * norm_bm[j])
                    for j in range(len(sims))
                ]

                order = sorted(range(len(combined)),
                               key=lambda x: combined[x],
                               reverse=True)

                top_indices = [top_indices[o] for o in order]

            # Write TREC output
            for rank, idx in enumerate(top_indices):
                out.write(
                    f"{qid} Q0 {passage_ids[idx]} {rank+1} "
                    f"{float(scores[idx]):.6f} {args.run_name}\n"
                )

    print("Run written to", args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--passage_root", default="wikipedia_passages")
    parser.add_argument("--queries", default="dryruntopics.txt")
    parser.add_argument("--output", default="run.txt")
    parser.add_argument("--top_k", type=int, default=1000)
    parser.add_argument("--run_name", default="bm25_run1")

    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)

    parser.add_argument("--title_boost", action="store_true")
    parser.add_argument("--title_boost_amount", type=float, default=2.0)

    parser.add_argument("--rm3", action="store_true")

    parser.add_argument("--use_dense_rerank", action="store_true")
    parser.add_argument("--dense_model", default="all-MiniLM-L6-v2")
    parser.add_argument("--dense_weight", type=float, default=0.6)

    args = parser.parse_args()
    main(args)