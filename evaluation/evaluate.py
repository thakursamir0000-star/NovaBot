"""Evaluate NovaBot on a document with known answers.

    python evaluation/evaluate.py                      # retrieval + system metrics (real models)
    python evaluation/evaluate.py --with-llm           # + answer quality via Groq (needs GROQ_API_KEY)
    python evaluation/evaluate.py --embedder hashing   # smoke test without torch — NOT real semantic numbers
    python evaluation/evaluate.py --doc book.pdf --questions my_questions.jsonl

Layer 1 — retrieval, per method: Recall@1/3/5, MRR@10, nDCG@5, split by
keyword vs paraphrase questions. A passage counts as relevant when it contains
the question's evidence phrase.
Layer 2 — answers (--with-llm): key-fact accuracy, abstention on unanswerable
questions, false abstentions, citation validity and support, and faithfulness
judged by the LLM against the retrieved passages.
Layer 3 — system: index build time, retrieval and answer latency (p50/p95),
tokens per answer.
"""
import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import llm  # noqa: E402
from embeddings import HashingEmbedder  # noqa: E402
from ingest import chunk_with_overlap, extract_text_by_page  # noqa: E402
from retrieval import DEFAULT_METHOD, build_index, get_reranker, retrieve  # noqa: E402
from vector_store import InMemoryStore  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_DOC = HERE / "handbook.txt"
DEFAULT_QUESTIONS = HERE / "questions.jsonl"
RESULTS_DIR = HERE / "results"
# The handbook is short, so it is cut into small chunks to give retrieval a real choice.
# Overlap stays above the longest evidence phrase so every phrase fits inside one chunk.
EVAL_CHUNK_SIZE = 60
EVAL_OVERLAP = 15
_PAGE_MARKER = re.compile(r"^--- page (\d+) ---$", re.MULTILINE)

JUDGE_PROMPT = """You are grading whether an answer is supported by source passages.

Passages:
{context}

Answer:
{answer}

Is every factual claim in the answer supported by the passages? Respond with JSON only:
{{"supported": true or false, "unsupported_claims": ["..."]}}"""


# ── loading ─────────────────────────────────────────────────────────────────

def load_pages(path):
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return extract_text_by_page(str(path))
    text = path.read_text(encoding="utf-8")
    parts = _PAGE_MARKER.split(text)
    if len(parts) == 1:
        return {0: text}
    return {int(num) - 1: body for num, body in zip(parts[1::2], parts[2::2])}


def load_questions(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize(text):
    """Lowercase, fold typographic dashes/quotes to ASCII, drop whitespace and markdown."""
    return re.sub(r"[\s*`]+", "", llm.fold_punctuation(text).lower())


def relevant_chunks(index, evidence):
    needle = normalize(evidence)
    return {i for i, c in enumerate(index.chunks) if needle in normalize(c["text"])}


# ── metrics ─────────────────────────────────────────────────────────────────

def reciprocal_rank(ranked, relevant, k=10):
    for rank, idx in enumerate(ranked[:k], 1):
        if idx in relevant:
            return 1.0 / rank
    return 0.0


def ndcg(ranked, relevant, k=5):
    dcg = sum(1.0 / math.log2(rank + 2) for rank, idx in enumerate(ranked[:k]) if idx in relevant)
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def includes_facts(answer, must_include):
    """Every item must appear; an item that is a list means "any of these"."""
    text = normalize(answer)
    return all(
        any(normalize(alt) in text for alt in (item if isinstance(item, list) else [item]))
        for item in must_include
    )


def percentile(values, q):
    return round(float(np.percentile(values, q)), 3) if values else None


def mean(values):
    return round(float(np.mean(values)), 4) if values else None


# ── layer 1 + 3: retrieval ──────────────────────────────────────────────────

def evaluate_retrieval(index, questions, configs, reranker=None):
    answerable = [q for q in questions if q.get("evidence")]
    report = {}
    for method, use_reranker in configs:
        name = method + (" + rerank" if use_reranker else "")
        rows, latencies = [], []
        for q in answerable:
            relevant = relevant_chunks(index, q["evidence"])
            start = time.perf_counter()
            results = retrieve(index, q["question"], top_k=10, method=method,
                               use_reranker=use_reranker, candidates=20, reranker=reranker)
            latencies.append(time.perf_counter() - start)
            ranked = [r["chunk_id"] for r in results]
            rows.append({
                "type": q["type"],
                **{f"recall@{k}": float(bool(set(ranked[:k]) & relevant)) for k in (1, 3, 5)},
                "mrr@10": reciprocal_rank(ranked, relevant),
                "ndcg@5": ndcg(ranked, relevant),
            })
        metrics = {m: mean([r[m] for r in rows]) for m in ("recall@1", "recall@3", "recall@5", "mrr@10", "ndcg@5")}
        for qtype in ("keyword", "paraphrase"):
            subset = [r["recall@5"] for r in rows if r["type"] == qtype]
            metrics[f"recall@5_{qtype}"] = mean(subset)
        metrics["latency_ms_p50"] = round(percentile(latencies, 50) * 1000, 1)
        metrics["latency_ms_p95"] = round(percentile(latencies, 95) * 1000, 1)
        report[name] = metrics
    return report


# ── layer 2 + 3: answers ────────────────────────────────────────────────────

def _judge(client, chunks, answer):
    context = "\n\n".join(f"[Source {i}] {c['text']}" for i, c in enumerate(chunks, 1))
    text, _ = llm.complete(client, [{"role": "user", "content": JUDGE_PROMPT.format(context=context, answer=answer)}],
                           temperature=0.0, max_tokens=600)
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return None
    try:
        return bool(json.loads(match.group(0)).get("supported"))
    except (json.JSONDecodeError, AttributeError):
        return None


def evaluate_answers(index, questions, client, method=DEFAULT_METHOD, top_k=5):
    cases = []
    for q in questions:
        t0 = time.perf_counter()
        chunks = retrieve(index, q["question"], top_k=top_k, method=method)
        t1 = time.perf_counter()
        answer, usage = llm.complete(client, llm.build_messages(q["question"], chunks, book_title="the Orion Valley handbook"))
        t2 = time.perf_counter()
        answer = answer or ""
        citations = llm.check_citations(answer, len(chunks))
        case = {
            "id": q["id"], "type": q["type"], "question": q["question"], "answer": answer,
            "abstained": citations["abstained"], "cited": citations["cited"], "invalid_citations": citations["invalid"],
            "retrieval_s": t1 - t0, "answer_s": t2 - t1,
            "tokens": getattr(usage, "total_tokens", None),
        }
        if q.get("evidence"):
            needle = normalize(q["evidence"])
            cited_chunks = [chunks[n - 1] for n in citations["cited"] if 1 <= n <= len(chunks)]
            case["correct"] = includes_facts(answer, q["must_include"]) and not citations["abstained"]
            case["citation_supported"] = any(needle in normalize(c["text"]) for c in cited_chunks)
            case["faithful"] = None if citations["abstained"] else _judge(client, chunks, answer)
        cases.append(case)

    answerable = [c for c in cases if c["type"] != "unanswerable"]
    unanswerable = [c for c in cases if c["type"] == "unanswerable"]
    answered = [c for c in answerable if not c["abstained"]]
    judged = [c["faithful"] for c in answered if c["faithful"] is not None]
    return {
        "method": method,
        "answer_accuracy": mean([float(c["correct"]) for c in answerable]),
        "abstention_on_unanswerable": mean([float(c["abstained"]) for c in unanswerable]),
        "false_abstention_rate": mean([float(c["abstained"]) for c in answerable]),
        "citation_rate": mean([float(bool(c["cited"])) for c in answered]),
        # Only answers that cite something can be valid or invalid.
        "citation_validity": mean([float(not c["invalid_citations"]) for c in cases if c["cited"]]),
        "citation_support": mean([float(c["citation_supported"]) for c in answered]),
        "faithfulness": mean([float(v) for v in judged]),
        "faithfulness_judged": len(judged),
        "answer_latency_s_p50": percentile([c["answer_s"] for c in cases], 50),
        "answer_latency_s_p95": percentile([c["answer_s"] for c in cases], 95),
        "tokens_per_answer": mean([c["tokens"] for c in cases if c["tokens"]]),
        "cases": cases,
    }


# ── reporting ───────────────────────────────────────────────────────────────

def markdown(report):
    lines = [f"**Document:** {report['document']} — {report['chunks']} chunks, embedder `{report['embedder']}`, "
             f"index built in {report['index_build_s']}s", "",
             "| Method | Recall@1 | Recall@3 | Recall@5 | MRR@10 | nDCG@5 | R@5 keyword | R@5 paraphrase | p50 ms |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, m in report["retrieval"].items():
        lines.append(f"| {name} | {m['recall@1']:.3f} | {m['recall@3']:.3f} | {m['recall@5']:.3f} | {m['mrr@10']:.3f} | "
                     f"{m['ndcg@5']:.3f} | {m['recall@5_keyword']:.3f} | {m['recall@5_paraphrase']:.3f} | {m['latency_ms_p50']} |")
    if "answers" in report:
        a = report["answers"]
        lines += ["", f"**Answers** (method `{a['method']}`, top-5 passages)", "",
                  "| Answer accuracy | Abstains when unanswerable | False abstentions | Answers with citations | "
                  "Citation validity | Citation support | Faithfulness | Latency p50 / p95 | Tokens / answer |",
                  "|---|---|---|---|---|---|---|---|---|",
                  f"| {a['answer_accuracy']} | {a['abstention_on_unanswerable']} | {a['false_abstention_rate']} | "
                  f"{a['citation_rate']} | {a['citation_validity']} | {a['citation_support']} | {a['faithfulness']} "
                  f"({a['faithfulness_judged']} judged) | {a['answer_latency_s_p50']}s / {a['answer_latency_s_p95']}s | "
                  f"{a['tokens_per_answer']} |"]
        problems = answer_issues(a["cases"])
        if problems:
            lines += ["", f"<details><summary>{len(problems)} misses and uncited answers</summary>", "", *problems,
                      "", "</details>"]
    return "\n".join(lines)


def answer_issues(cases):
    """One line per case worth a look: wrong, wrongly refused, not refused, or uncited."""
    lines = []
    for c in cases:
        if c["type"] == "unanswerable":
            label = None if c["abstained"] else "did not abstain"
        elif c["abstained"]:
            label = "false abstention"
        elif not c["correct"]:
            label = "wrong answer"
        elif not c["cited"]:
            label = "no citation detected"
        else:
            label = None
        if label:
            answer = " ".join(c["answer"].split())[:200].replace("|", "/")
            lines.append(f"- **{label}** — #{c['id']} {c['question']} → {answer}")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--doc", default=str(DEFAULT_DOC), help="PDF or handbook-style .txt")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--embedder", choices=["model", "hashing"], default="model",
                        help="hashing = no torch needed, but NOT a real semantic model")
    parser.add_argument("--chunk-size", type=int, default=EVAL_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=EVAL_OVERLAP)
    parser.add_argument("--with-llm", action="store_true", help="also evaluate answers with Groq")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    embedder = HashingEmbedder() if args.embedder == "hashing" else None

    start = time.perf_counter()
    chunks = chunk_with_overlap(load_pages(args.doc), chunk_size=args.chunk_size, overlap=args.overlap)
    index = build_index(chunks, "eval", InMemoryStore(), embedder=embedder)
    build_s = round(time.perf_counter() - start, 2)

    configs = [("bm25", False), ("semantic", False), ("weighted", False), ("rrf", False)]
    reranker = None
    if args.embedder == "model":
        reranker = get_reranker()
        configs += [("weighted", True), ("rrf", True)]

    report = {
        "document": Path(args.doc).name, "chunks": len(index.chunks),
        "embedder": "hashing (smoke test)" if embedder else "all-MiniLM-L6-v2",
        "index_build_s": build_s,
        "retrieval": evaluate_retrieval(index, questions, configs, reranker),
    }

    if args.with_llm:
        client = llm.get_client()
        if client is None:
            raise SystemExit("--with-llm needs GROQ_API_KEY in the environment.")
        report["answers"] = evaluate_answers(index, questions, client)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "latest.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    table = markdown(report)
    print(table)
    print(f"\nFull results: {out}")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write("## NovaBot evaluation\n\n" + table + "\n")


if __name__ == "__main__":
    main()
