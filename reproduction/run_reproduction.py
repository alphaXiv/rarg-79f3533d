#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import re
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np


def normalize_answer(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def exact_control(prediction: str, reference: str) -> bool:
    pred = normalize_answer(prediction)
    gold = normalize_answer(reference)
    return bool(gold and (pred == gold or gold in pred))


def token_f1(prediction: str, reference: str) -> float:
    pred = normalize_answer(prediction).split()
    gold = normalize_answer(reference).split()
    if not pred or not gold:
        return 0.0
    overlap = 0
    remaining = list(gold)
    for token in pred:
        if token in remaining:
            overlap += 1
            remaining.remove(token)
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def extract_json(text: str) -> Any:
    for candidate in re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.S):
        try:
            return json.loads(candidate)
        except Exception:
            pass
    starts = [i for i, ch in enumerate(text) if ch in "[{"]
    for start in starts:
        for end in range(len(text), start, -1):
            try:
                return json.loads(text[start:end])
            except Exception:
                continue
    return None


@dataclass
class Match:
    path: str
    line: int
    text: str
    term: str


class Models:
    def __init__(self, agent_path: str, embedding_path: str, device_index: int):
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = f"cuda:{device_index}"
        self.tokenizer = AutoTokenizer.from_pretrained(agent_path, trust_remote_code=True)
        self.llm = AutoModelForCausalLM.from_pretrained(
            agent_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(self.device).eval()
        self.embedder = SentenceTransformer(
            embedding_path,
            device=self.device,
            trust_remote_code=True,
        )
        self.embedder.max_seq_length = 512

    def generate(self, system: str, user: str, max_new_tokens: int = 512) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=28672).to(self.device)
        with self.torch.inference_mode():
            output = self.llm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True).strip()

    def embed(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        if query:
            texts = [
                "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
                + text
                for text in texts
            ]
        vectors = self.embedder.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)


def read_rows(path: str, limit: int) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def split_paragraphs(text: str) -> list[str]:
    raw = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    result: list[str] = []
    for part in raw:
        if len(part) < 120:
            continue
        for start in range(0, len(part), 900):
            chunk = part[start : start + 1100].strip()
            if len(chunk) >= 120:
                result.append(chunk)
    return result


def search_scope(scope_file: str, term: str, candidate_limit: int) -> list[Match]:
    script = (
        "set +o pipefail; "
        "xargs -d '\\n' rg -j1 -n -i -F --no-heading --color never -- \"$1\" < \"$2\" "
        "| head -n \"$3\""
    )
    proc = subprocess.run(
        ["bash", "-c", script, "_", term, scope_file, str(candidate_limit)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        timeout=180,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ripgrep pipeline failed with exit code {proc.returncode} for term={term!r}: "
            f"{proc.stderr[-1000:] if proc.stderr else 'no stderr'}"
        )
    matches: list[Match] = []
    for line in proc.stdout.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        try:
            line_no = int(parts[1])
        except ValueError:
            continue
        matches.append(Match(parts[0], line_no, parts[2][:1000], term))
    return matches


def gold_visible(matches: list[Match], gold_paths: set[str], corpus: Path) -> bool:
    for match in matches:
        try:
            rel = str(Path(match.path).resolve().relative_to(corpus.resolve()))
        except Exception:
            rel = match.path
        if rel in gold_paths:
            return True
    return False


def make_entry_points(
    models: Models,
    question: str,
    ranked_paths: list[str],
    corpus: Path,
    doc_limit: int,
    paragraph_limit: int,
) -> list[Match]:
    paragraphs: list[tuple[str, str]] = []
    for rel in ranked_paths[:doc_limit]:
        try:
            text = (corpus / rel).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        paragraphs.extend((rel, paragraph) for paragraph in split_paragraphs(text))
    if not paragraphs:
        return []
    paragraph_vectors = models.embed([p[1] for p in paragraphs])
    query_vector = models.embed([question], query=True)[0]
    scores = paragraph_vectors @ query_vector
    selected = np.argsort(-scores)[:paragraph_limit]
    return [Match(paragraphs[i][0], 0, paragraphs[i][1][:1000], "<entry>") for i in selected]


def propose_terms(models: Models, question: str, entry_points: list[Match]) -> list[str]:
    entry_text = "\n\n".join(f"[{m.path}] {m.text}" for m in entry_points)
    prompt = f"""Question:
{question}

Potential entry passages:
{entry_text if entry_text else "(none)"}

Return JSON only: {{"terms": ["literal phrase", ...]}}.
    Give 6 short literal ripgrep search strings, ordered from most distinctive to broader.
    Each string MUST be either one distinctive word or an exact phrase of at most 3 words.
Use names, dates, titles, places, and unusual clue phrases. Do not use regex syntax."""
    raw = models.generate(
        "You design high-precision corpus searches for multi-document questions.",
        prompt,
        max_new_tokens=320,
    )
    parsed = extract_json(raw)
    terms = parsed.get("terms", []) if isinstance(parsed, dict) else []
    stopwords = {
        "about", "after", "author", "before", "between", "company", "during",
        "first", "from", "historic", "middle", "name", "paper", "person",
        "specific", "their", "university", "what", "when", "where", "which",
        "whose", "with", "would",
    }
    cleaned: list[str] = []
    for term in terms:
        term = re.sub(r"\s+", " ", str(term)).strip(" \"'`")
        tokens = re.findall(r"[A-Za-z0-9£'-]+", term)
        if len(tokens) > 3:
            candidates = [token for token in tokens if token.casefold() not in stopwords and len(token) >= 4]
            tokens = [max(candidates, key=len)] if candidates else tokens[:1]
        term = " ".join(tokens)
        if 2 <= len(term) <= 80 and term.casefold() not in {x.casefold() for x in cleaned}:
            cleaned.append(term)
    if not cleaned:
        words = re.findall(r"[A-Za-z0-9£'-]{4,}", question)
        cleaned = [" ".join(words[i : i + 2]) for i in range(0, min(len(words), 12), 2)]
    return cleaned[:6]


def decide(models: Models, question: str, evidence: list[Match], remaining: int) -> dict[str, str]:
    evidence_text = "\n".join(
        f"[{i + 1}] {m.path}:{m.line} | {m.text}" for i, m in enumerate(evidence[-40:])
    )
    prompt = f"""Question:
{question}

Corpus evidence:
{evidence_text[-26000:]}

You have {remaining} searches remaining. Return JSON only:
{{"status":"answer" or "continue","answer":"short answer or empty","next_term":"literal search phrase or empty","reason":"brief"}}
Answer only when the evidence supports a specific answer. Otherwise propose one new literal search phrase."""
    raw = models.generate(
        "You are an evidence-grounded research agent. Never use outside knowledge as evidence.",
        prompt,
        max_new_tokens=420,
    )
    parsed = extract_json(raw)
    if not isinstance(parsed, dict):
        return {"status": "continue", "answer": "", "next_term": "", "reason": "unparsed"}
    return {key: str(parsed.get(key, "")) for key in ["status", "answer", "next_term", "reason"]}


def judge(models: Models, question: str, reference: str, prediction: str) -> bool:
    if "insufficient evidence" in prediction.casefold() or "unknown" == prediction.strip().casefold():
        return False
    if exact_control(prediction, reference):
        return True
    prompt = f"""Question: {question}
Reference answer: {reference}
Candidate answer: {prediction}

Judge whether the candidate explicitly identifies the same answer as the reference.
Missing, uncertain, abstaining, or merely related candidates are incorrect. Ignore only
case, punctuation, articles, and harmless explanation.
Return JSON only: {{"correct": true}} or {{"correct": false}}."""
    raw = models.generate(
        "You are the released-evaluator-compatible answer correctness judge.",
        prompt,
        max_new_tokens=48,
    )
    parsed = extract_json(raw)
    return bool(parsed.get("correct")) if isinstance(parsed, dict) else False


def run_query(
    row: dict[str, Any],
    condition: str,
    config: dict[str, Any],
    models: Models,
    corpus: Path,
    index: faiss.Index,
    paths: list[str],
    docid_map: dict[str, str],
    work_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    question = row["query"]
    qid = str(row["query_id"])
    gold_paths = {docid_map[x] for x in map(str, row.get("gold_doc_ids", [])) if x in docid_map}
    query_vector = models.embed([question], query=True)
    _, nearest = index.search(query_vector, len(paths))
    ranked_paths = [paths[i] for i in nearest[0] if i >= 0]
    default_paths = sorted(paths)
    traversal = default_paths if condition == "dci" else ranked_paths

    scope_path = work_dir / f"scope-{qid}.txt"
    scope_path.write_text("\n".join(str(corpus / rel) for rel in traversal) + "\n")

    entry_points: list[Match] = []
    if condition in {"rarg_plus", "rarg_pp"}:
        entry_points = make_entry_points(
            models,
            question,
            ranked_paths,
            corpus,
            int(config["entry_document_limit"]),
            int(config["entry_paragraphs"]),
        )

    terms = propose_terms(models, question, entry_points)
    evidence: list[Match] = list(entry_points)
    evidence_arrival_step = 0 if gold_visible(entry_points, gold_paths, corpus) else None
    tool_steps = 0
    observed_matches = len(entry_points)
    candidate_matches = len(entry_points)
    prediction = ""
    trace: list[dict[str, Any]] = []
    used_terms: set[str] = set()

    for step in range(1, int(config["max_steps"]) + 1):
        term = terms.pop(0) if terms else ""
        if not term:
            break
        if term.casefold() in used_terms:
            continue
        used_terms.add(term.casefold())
        candidate_limit = (
            int(config["rerank_candidates"]) if condition == "rarg_pp" else int(config["matches_per_step"])
        )
        candidates = search_scope(str(scope_path), term, candidate_limit)
        searched_term = term
        if not candidates and len(term.split()) > 1:
            fallback_tokens = [
                token for token in re.findall(r"[A-Za-z0-9£'-]+", term)
                if len(token) >= 4
            ]
            if fallback_tokens:
                searched_term = max(fallback_tokens, key=len)
                used_terms.add(searched_term.casefold())
                candidates = search_scope(str(scope_path), searched_term, candidate_limit)
        candidate_matches += len(candidates)
        shown = candidates
        if condition == "rarg_pp" and len(candidates) > int(config["matches_per_step"]):
            query = question + "\nLocal search intent: " + searched_term
            vectors = models.embed([m.text for m in candidates])
            qvec = models.embed([query], query=True)[0]
            order = np.argsort(-(vectors @ qvec))[: int(config["matches_per_step"])]
            shown = [candidates[i] for i in order]

        tool_steps += 1
        observed_matches += len(shown)
        evidence.extend(shown)
        if evidence_arrival_step is None and gold_visible(shown, gold_paths, corpus):
            evidence_arrival_step = step
        decision = decide(models, question, evidence, int(config["max_steps"]) - step)
        trace.append(
            {
                "step": step,
                "term": searched_term,
                "candidate_count": len(candidates),
                "shown_count": len(shown),
                "gold_visible": gold_visible(shown, gold_paths, corpus),
                "decision": decision,
            }
        )
        if decision.get("status", "").casefold() == "answer" and decision.get("answer", "").strip():
            prediction = decision["answer"].strip()
            break
        next_term = decision.get("next_term", "").strip(" \"'`")
        if (
            next_term
            and next_term.casefold() not in used_terms
            and next_term.casefold() not in {queued.casefold() for queued in terms}
        ):
            terms.insert(0, next_term[:100])

    if not prediction:
        final = decide(models, question, evidence, 0)
        prediction = final.get("answer", "").strip()
    if not prediction:
        prediction = "Insufficient evidence"

    judge_correct = judge(models, question, row["answer"], prediction)
    result = {
        "query_id": qid,
        "condition": condition,
        "question": question,
        "reference": row["answer"],
        "prediction": prediction,
        "exact_control": exact_control(prediction, row["answer"]),
        "token_f1": round(token_f1(prediction, row["answer"]), 4),
        "open_judge_correct": judge_correct,
        "gold_paths": sorted(gold_paths),
        "gold_mapping_complete": len(gold_paths) == len(row.get("gold_doc_ids", [])),
        "evidence_recovered": evidence_arrival_step is not None,
        "evidence_arrival_step": evidence_arrival_step,
        "tool_steps": tool_steps,
        "observed_matches": observed_matches,
        "candidate_matches": candidate_matches,
        "observed_characters": sum(len(m.text) for m in evidence),
        "wall_seconds": round(time.perf_counter() - started, 3),
        "trace": trace,
    }
    print("QUERY_RESULT " + json.dumps(result, sort_keys=True), flush=True)
    return result


def worker_main(
    worker_id: int,
    rows: list[dict[str, Any]],
    args_dict: dict[str, str],
    config: dict[str, Any],
    output_queue: mp.Queue,
) -> None:
    try:
        models = Models(args_dict["agent_model"], args_dict["embedding_model"], worker_id)
        index = faiss.read_index(str(Path(args_dict["index"]) / "index.faiss"))
        paths = json.loads((Path(args_dict["index"]) / "paths.json").read_text())
        docid_map = json.loads(Path(args_dict["docid_map"]).read_text())
        corpus = Path(args_dict["corpus"])
        with tempfile.TemporaryDirectory(prefix=f"rarg-w{worker_id}-") as tmp:
            for row in rows:
                output_queue.put(
                    (
                        "result",
                        run_query(
                            row,
                            config["condition"],
                            config,
                            models,
                            corpus,
                            index,
                            paths,
                            docid_map,
                            Path(tmp),
                        ),
                    )
                )
        output_queue.put(("done", worker_id))
    except Exception as exc:
        import traceback

        output_queue.put(("error", {"worker": worker_id, "error": repr(exc), "traceback": traceback.format_exc()}))


def summarize(results: list[dict[str, Any]], config: dict[str, Any], elapsed: float) -> dict[str, Any]:
    arrivals = [r["evidence_arrival_step"] for r in results if r["evidence_arrival_step"] is not None]
    summary = {
        "condition": config["condition"],
        "n": len(results),
        "query_ids": [r["query_id"] for r in results],
        "open_judge_accuracy": sum(r["open_judge_correct"] for r in results) / len(results),
        "exact_control_accuracy": sum(r["exact_control"] for r in results) / len(results),
        "mean_token_f1": statistics.mean(r["token_f1"] for r in results),
        "evidence_recall": sum(r["evidence_recovered"] for r in results) / len(results),
        "mean_evidence_arrival_step_recovered": statistics.mean(arrivals) if arrivals else None,
        "mean_tool_steps": statistics.mean(r["tool_steps"] for r in results),
        "mean_observed_matches": statistics.mean(r["observed_matches"] for r in results),
        "mean_candidate_matches": statistics.mean(r["candidate_matches"] for r in results),
        "mean_observed_characters": statistics.mean(r["observed_characters"] for r in results),
        "wall_seconds": elapsed,
        "backend": "kubernetes",
        "gpu_model": "NVIDIA RTX PRO 6000 Blackwell",
        "allocated_gpus": int(config["workers"]),
        "agent_model": config["agent_model"],
        "embedding_model": config["embedding_model"],
        "subset": "first 16 rows of released bcplus_qa_sample100.jsonl",
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--docid-map", required=True)
    parser.add_argument("--agent-model", required=True)
    parser.add_argument("--embedding-model", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    rows = read_rows(args.dataset, int(config["query_limit"]))
    workers = int(config["workers"])
    assignments = [rows[i::workers] for i in range(workers)]
    args_dict = vars(args)
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    processes = [
        ctx.Process(target=worker_main, args=(i, assignments[i], args_dict, config, queue))
        for i in range(workers)
    ]
    started = time.perf_counter()
    for process in processes:
        process.start()

    results: list[dict[str, Any]] = []
    done = 0
    errors: list[dict[str, Any]] = []
    while done < workers and not errors:
        kind, payload = queue.get()
        if kind == "result":
            results.append(payload)
        elif kind == "done":
            done += 1
        elif kind == "error":
            errors.append(payload)
            print("WORKER_ERROR " + json.dumps(payload, sort_keys=True), flush=True)

    for process in processes:
        process.join(timeout=60)
    if errors:
        raise RuntimeError(f"worker failures: {errors}")
    results.sort(key=lambda row: rows.index(next(x for x in rows if str(x["query_id"]) == row["query_id"])))
    if len(results) != len(rows):
        raise RuntimeError(f"incomplete results: {len(results)}/{len(rows)}")

    elapsed = round(time.perf_counter() - started, 3)
    digest = hashlib.sha256(
        "\n".join(r["query_id"] for r in results).encode()
    ).hexdigest()[:16]
    summary = summarize(results, config, elapsed)
    summary["query_subset_sha256_16"] = digest
    print("=== REPRODUCTION_SUMMARY_BEGIN ===", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print("=== REPRODUCTION_SUMMARY_END ===", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
