import gzip
import json
import os
import re
from rank_bm25 import BM25Okapi
from tqdm import tqdm

# =========================
# 1. LOAD PASSAGES
# =========================

passages = []
passage_ids = []

for root, dirs, files in os.walk("wikipedia_passages"):
    for file in tqdm(files):
        if file.endswith(".jsonl.gz"):
            with gzip.open(os.path.join(root, file), "rt") as f:
                for line in f:
                    data = json.loads(line)
                    passages.append(data["text"])
                    passage_ids.append(f'{data["doc_id"]}_{data["chunk_id"]}')

print("Total passages loaded:", len(passages))

# =========================
# 2. BUILD BM25 INDEX
# =========================

print("Tokenizing corpus...")
tokenized_corpus = [p.split() for p in passages]

print("Building BM25 index...")
bm25 = BM25Okapi(tokenized_corpus)

# =========================
# 3. LOAD QUERIES
# =========================

def load_queries(filepath):
    queries = []
    with open(filepath, "r") as f:
        content = f.read()

    pattern = re.findall(
        r"<qID>(.*?)</qID>.*?<q>(.*?)</q>",
        content,
        re.DOTALL
    )

    for qid, qtext in pattern:
        queries.append((qid.strip(), qtext.strip()))

    return queries


queries = load_queries("dryruntopics.txt")
print("Total queries loaded:", len(queries))

# =========================
# 4. GENERATE RUN FILE
# =========================

run_name = "bm25_run1"
top_k = 100   # You can increase later if required

with open("run.txt", "w") as out:
    for qid, query in queries:
        tokenized_query = query.split()
        scores = bm25.get_scores(tokenized_query)

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        for rank, idx in enumerate(top_indices):
            out.write(
                f"{qid} Q0 {passage_ids[idx]} {rank+1} {scores[idx]:.4f} {run_name}\n"
            )

print("Run file generated: run.txt")
