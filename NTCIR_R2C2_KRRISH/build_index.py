import os
import gzip
import json
from whoosh import index
from whoosh.fields import Schema, TEXT, ID
from whoosh.analysis import StandardAnalyzer
from tqdm import tqdm

schema = Schema(
    passage_id=ID(stored=True),
    content=TEXT(stored=True, analyzer=StandardAnalyzer())
)

if not os.path.exists("indexdir"):
    os.mkdir("indexdir")

ix = index.create_in("indexdir", schema)
writer = ix.writer(limitmb=256)  # keeps RAM usage controlled

# Collect all jsonl.gz files first
all_files = []
for root, dirs, files in os.walk("wikipedia_passages"):
    for file in files:
        if file.endswith(".jsonl.gz"):
            all_files.append(os.path.join(root, file))

print(f"Total files to process: {len(all_files)}")

passage_count = 0

for filepath in tqdm(all_files, desc="Indexing files"):
    with gzip.open(filepath, "rt") as f:
        for line in f:
            data = json.loads(line)
            pid = f'{data["doc_id"]}_{data["chunk_id"]}'
            writer.add_document(
                passage_id=pid,
                content=data["text"]
            )
            passage_count += 1

writer.commit()

print(f"\nIndex built successfully.")
print(f"Total passages indexed: {passage_count}")