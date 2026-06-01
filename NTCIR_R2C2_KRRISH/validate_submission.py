#!/usr/bin/env python3
import argparse
import hashlib
import os
import re
import zipfile
from collections import Counter
from typing import List, Tuple


def parse_topics(topics_path: str) -> List[Tuple[str, str]]:
    with open(topics_path, "r", encoding="utf-8") as f:
        raw = f.read()
    pairs = re.findall(r"<qID>(.*?)</qID>.*?<q>(.*?)</q>", raw, re.DOTALL)
    return [(qid.strip(), q.strip()) for qid, q in pairs]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_run_file(path: str, expected_qids: List[str], per_query: int) -> None:
    expected_set = set(expected_qids)
    seen = Counter()
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.rstrip("\n").split(";")
            if len(parts) < 4:
                raise ValueError(f"{path}: malformed line {line_no}")
            qid, rank, doc_id = parts[0], parts[1], parts[2]
            if qid not in expected_set:
                raise ValueError(f"{path}: unknown qid '{qid}' on line {line_no}")
            if not rank.isdigit():
                raise ValueError(f"{path}: non-numeric rank '{rank}' on line {line_no}")
            if not doc_id:
                raise ValueError(f"{path}: empty doc_id on line {line_no}")
            seen[qid] += 1

    missing = [qid for qid in expected_qids if qid not in seen]
    if missing:
        raise ValueError(f"{path}: missing {len(missing)} qids")

    bad = [qid for qid, cnt in seen.items() if cnt != per_query]
    if bad:
        raise ValueError(f"{path}: {len(bad)} qids do not have {per_query} rows")


def validate_zip(path: str, team_name: str) -> None:
    expected_names = {f"{team_name}-PO-{i}" for i in (1, 2, 3, 4)}
    with zipfile.ZipFile(path, "r") as zf:
        names = set(zf.namelist())
    if names != expected_names:
        raise ValueError(
            f"{path}: zip entries mismatch. expected={sorted(expected_names)} got={sorted(names)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate NTCIR PR submission files.")
    parser.add_argument("--submission_dir", required=True)
    parser.add_argument("--topics", required=True)
    parser.add_argument("--team_name", default="KRRISH")
    parser.add_argument("--per_query", type=int, default=20)
    args = parser.parse_args()

    topics = parse_topics(args.topics)
    expected_qids = [qid for qid, _ in topics]
    if len(set(expected_qids)) != len(expected_qids):
        raise ValueError("Duplicate qIDs found in topics file.")

    run_files = [os.path.join(args.submission_dir, f"{args.team_name}-PO-{i}") for i in (1, 2, 3, 4)]
    zip_path = os.path.join(args.submission_dir, f"{args.team_name}-PR.zip")

    for rf in run_files:
        if not os.path.exists(rf):
            raise FileNotFoundError(f"Missing run file: {rf}")
        validate_run_file(rf, expected_qids, per_query=args.per_query)

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Missing zip file: {zip_path}")
    validate_zip(zip_path, args.team_name)

    print("Validation passed.")
    for rf in run_files:
        print(f"{os.path.basename(rf)} sha256={sha256(rf)}")
    print(f"{os.path.basename(zip_path)} sha256={sha256(zip_path)}")


if __name__ == "__main__":
    main()
