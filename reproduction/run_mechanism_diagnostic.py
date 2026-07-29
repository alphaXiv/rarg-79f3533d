#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


STOPWORDS = {
    "about", "after", "also", "author", "before", "between", "could", "during",
    "first", "from", "have", "into", "name", "paper", "person", "provide",
    "specific", "their", "there", "these", "this", "university", "what", "when",
    "where", "which", "whose", "with", "would", "years",
}


def load_rows(path: Path, limit: int, offset: int = 0) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
        if len(rows) == offset + limit:
            break
    return rows[offset : offset + limit]


def query_text(text: str) -> str:
    return (
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
        "Query: " + text
    )


def terms_for(question: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z'-]{4,}", question)
    unique: dict[str, str] = {}
    for token in tokens:
        if token.casefold() not in STOPWORDS:
            unique.setdefault(token.casefold(), token)
    return sorted(unique.values(), key=lambda token: (-len(token), token.casefold()))[:6]


def split_paragraphs(text: str) -> list[str]:
    paragraphs = []
    for part in re.split(r"\n\s*\n", text):
        part = part.strip()
        if len(part) < 120:
            continue
        paragraphs.extend(part[start : start + 1000] for start in range(0, len(part), 900))
    return [paragraph for paragraph in paragraphs if len(paragraph) >= 120]


def rg_matches(scope: Path, term: str, limit: int) -> list[dict[str, Any]]:
    script = (
        "set +o pipefail; "
        "xargs -d '\\n' rg -j1 -n -i -F --no-heading --color never -- \"$1\" < \"$2\" "
        "| head -n \"$3\""
    )
    proc = subprocess.run(
        ["bash", "-c", script, "_", term, str(scope), str(limit)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"rg failed {proc.returncode}: {proc.stderr[-1000:]}")
    matches = []
    for line in proc.stdout.splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[1].isdigit():
            matches.append({"path": parts[0], "line": int(parts[1]), "text": parts[2][:1000], "term": term})
    return matches


def rel_path(path: str, corpus: Path) -> str:
    return str(Path(path).resolve().relative_to(corpus.resolve()))


def first_rank(order: list[str], gold: set[str]) -> int | None:
    for index, path in enumerate(order, 1):
        if path in gold:
            return index
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--docid-map", type=Path, required=True)
    parser.add_argument("--embedding-model", required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    config = json.loads(args.config.read_text())
    rows = load_rows(
        args.dataset,
        int(config["query_limit"]),
        int(config.get("query_offset", 0)),
    )
    paths: list[str] = json.loads((args.index / "paths.json").read_text())
    docid_map: dict[str, str] = json.loads(args.docid_map.read_text())
    index = faiss.read_index(str(args.index / "index.faiss"))
    model = SentenceTransformer(args.embedding_model, device="cuda:0", trust_remote_code=True)
    model.max_seq_length = 512
    lex_order = sorted(paths)
    results = []

    with tempfile.TemporaryDirectory(prefix="rarg-mechanism-") as tmp:
        scope = Path(tmp) / "scope.txt"
        for row in rows:
            question = row["query"]
            gold = {docid_map[x] for x in map(str, row.get("gold_doc_ids", [])) if x in docid_map}
            qvec = model.encode(
                [query_text(question)],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype(np.float32)
            _, nearest = index.search(qvec, len(paths))
            ranked = [paths[i] for i in nearest[0] if i >= 0]
            lex_rank = first_rank(lex_order, gold)
            relevance_rank = first_rank(ranked, gold)

            paragraphs: list[tuple[str, str]] = []
            for rel in ranked[: int(config["entry_document_limit"])]:
                try:
                    text = (args.corpus / rel).read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                paragraphs.extend((rel, paragraph) for paragraph in split_paragraphs(text))
            entry_gold = False
            if paragraphs:
                pvecs = model.encode(
                    [paragraph for _, paragraph in paragraphs],
                    batch_size=32,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                selected = np.argsort(-(pvecs @ qvec[0]))[: int(config["entry_paragraphs"])]
                entry_gold = any(paragraphs[i][0] in gold for i in selected)

            scope.write_text("\n".join(str(args.corpus / rel) for rel in ranked) + "\n")
            candidates = []
            seen = set()
            for term in terms_for(question):
                for match in rg_matches(scope, term, 120):
                    key = (match["path"], match["line"], match["text"])
                    if key not in seen:
                        seen.add(key)
                        candidates.append(match)
                    if len(candidates) >= int(config["rerank_candidates"]):
                        break
                if len(candidates) >= int(config["rerank_candidates"]):
                    break
            ordered_top = candidates[: int(config["matches_per_step"])]
            ordered_gold = any(rel_path(match["path"], args.corpus) in gold for match in ordered_top)
            reranked_gold = False
            if candidates:
                mvecs = model.encode(
                    [match["text"] for match in candidates],
                    batch_size=32,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                selected = np.argsort(-(mvecs @ qvec[0]))[: int(config["matches_per_step"])]
                reranked_gold = any(
                    rel_path(candidates[i]["path"], args.corpus) in gold for i in selected
                )

            result = {
                "query_id": str(row["query_id"]),
                "gold_mapping_complete": len(gold) == len(row.get("gold_doc_ids", [])),
                "lexicographic_gold_rank": lex_rank,
                "relevance_gold_rank": relevance_rank,
                "rank_speedup": (lex_rank / relevance_rank) if lex_rank and relevance_rank else None,
                "relevance_recall_at_100": relevance_rank is not None and relevance_rank <= 100,
                "relevance_recall_at_1000": relevance_rank is not None and relevance_rank <= 1000,
                "relevance_recall_at_10000": relevance_rank is not None and relevance_rank <= 10000,
                "entry_gold_visible": entry_gold,
                "local_candidate_count": len(candidates),
                "ordered_top30_gold": ordered_gold,
                "reranked_top30_gold": reranked_gold,
            }
            results.append(result)
            print("MECHANISM_RESULT " + json.dumps(result, sort_keys=True), flush=True)

    speedups = [r["rank_speedup"] for r in results if r["rank_speedup"] is not None]
    summary = {
        "n": len(results),
        "condition": "paired_mechanism_diagnostic",
        "median_lexicographic_gold_rank": statistics.median(
            r["lexicographic_gold_rank"] for r in results if r["lexicographic_gold_rank"]
        ),
        "median_relevance_gold_rank": statistics.median(
            r["relevance_gold_rank"] for r in results if r["relevance_gold_rank"]
        ),
        "median_rank_speedup": statistics.median(speedups),
        "relevance_recall_at_100": statistics.mean(r["relevance_recall_at_100"] for r in results),
        "relevance_recall_at_1000": statistics.mean(r["relevance_recall_at_1000"] for r in results),
        "relevance_recall_at_10000": statistics.mean(r["relevance_recall_at_10000"] for r in results),
        "entry_gold_visibility": statistics.mean(r["entry_gold_visible"] for r in results),
        "ordered_top30_gold_visibility": statistics.mean(r["ordered_top30_gold"] for r in results),
        "reranked_top30_gold_visibility": statistics.mean(r["reranked_top30_gold"] for r in results),
        "mean_local_candidates": statistics.mean(r["local_candidate_count"] for r in results),
        "backend": "kubernetes",
        "gpu_model": "NVIDIA RTX PRO 6000 Blackwell",
        "allocated_gpus": 4,
        "wall_seconds": round(time.perf_counter() - started, 3),
    }
    print("=== REPRODUCTION_SUMMARY_BEGIN ===")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("=== REPRODUCTION_SUMMARY_END ===")


if __name__ == "__main__":
    main()
