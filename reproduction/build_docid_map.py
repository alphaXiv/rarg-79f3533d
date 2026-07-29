#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import pyarrow.parquet as pq

TITLE_RE = re.compile(r"(?mi)^title:\s*(.+?)\s*$")
INVALID_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
WHITESPACE_RE = re.compile(r"\s+")
MAX_STEM_LEN = 140


def sanitize_name(value: str, fallback: str) -> str:
    value = INVALID_CHARS_RE.sub(" ", value)
    value = WHITESPACE_RE.sub(" ", value).strip().strip(".")
    return value or fallback


def build_filename(text: str, url: str, docid: str) -> tuple[str, str]:
    title_match = TITLE_RE.search(text or "")
    title = title_match.group(1).strip() if title_match else None
    parsed = urlparse(url or "")
    domain = sanitize_name((parsed.hostname or "unknown-domain").lower(), "unknown-domain")
    path_name = Path(parsed.path).name
    stem = title or path_name or f"doc-{docid}"
    stem = sanitize_name(stem, f"doc-{docid}")[:MAX_STEM_LEN].rstrip(" .")
    return domain, f"{stem or f'doc-{docid}'}.txt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = json.loads(args.paths.read_text())
    path_set = set(paths)
    mapping: dict[str, str] = {}
    misses: list[str] = []
    pf = pq.ParquetFile(args.parquet)
    for row_group in range(pf.num_row_groups):
        rows = pf.read_row_group(row_group, columns=["docid", "text", "url"]).to_pylist()
        for row in rows:
            docid = str(row["docid"])
            domain, filename = build_filename(row.get("text") or "", row.get("url") or "", docid)
            base = Path(filename)
            candidates = [f"{domain}/{filename}", f"{domain}/{base.stem}__docid_{docid}{base.suffix}"]
            candidates.extend(
                f"{domain}/{base.stem}__docid_{docid}_{i}{base.suffix}" for i in range(2, 200)
            )
            found = next((candidate for candidate in candidates if candidate in path_set), None)
            if found:
                mapping[docid] = found
            else:
                misses.append(docid)

    args.output.write_text(json.dumps(mapping, sort_keys=True))
    print(
        f"DOCID_MAP mapped={len(mapping)} missing={len(misses)} "
        f"needed_output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
