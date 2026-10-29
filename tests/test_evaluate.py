import json

import pytest

from embeddings import HashingEmbedder
from evaluation.evaluate import (
    DEFAULT_DOC,
    DEFAULT_QUESTIONS,
    EVAL_CHUNK_SIZE,
    EVAL_OVERLAP,
    answer_issues,
    evaluate_answers,
    evaluate_retrieval,
    includes_facts,
    load_pages,
    load_questions,
    ndcg,
    reciprocal_rank,
    relevant_chunks,
)
from ingest import chunk_with_overlap
from retrieval import build_index
from tests.conftest import FakeClient
from vector_store import InMemoryStore


@pytest.fixture(scope="module")
def handbook_index():
    chunks = chunk_with_overlap(load_pages(DEFAULT_DOC), chunk_size=EVAL_CHUNK_SIZE, overlap=EVAL_OVERLAP)
    return build_index(chunks, "eval", InMemoryStore(), embedder=HashingEmbedder())


def test_ranking_metrics():
    assert reciprocal_rank([5, 2, 9], {9}) == pytest.approx(1 / 3)
    assert reciprocal_rank([5, 2], {9}) == 0.0
    assert ndcg([1, 2], {1, 2}) == pytest.approx(1.0)
    assert 0 < ndcg([3, 1], {1}) < 1


def test_fact_matching_ignores_case_spacing_and_markdown():
    assert includes_facts("It is **Priya  Raman**.", ["priya raman"])
    assert includes_facts("Route 9 closes above 40cm of snow", ["40 cm"])
    assert includes_facts("once a year", [["12 months", "once a year"]])
    assert not includes_facts("Tuesdays only", ["tuesday", "friday"])


# Real answers from CI that the first grader marked wrong: gpt-oss wrote U+2011 non-breaking hyphens.
@pytest.mark.parametrize("answer, facts", [
    ("- Wetland transect WT‑3 is closed for restoration until 2027【Source 1】【Source 2】", ["wt-3", "2027"]),
    ("- Use form IR‑2 to report an accident. [Source 1]", ["ir-2"]),
    ("- The station follows a zero single‑use plastics policy since 2021 [Source 1]", ["single-use"]),
    ("Transect WT–3 is closed", ["wt-3"]),  # en dash
])
def test_fact_matching_folds_typographic_dashes(answer, facts):
    assert includes_facts(answer, facts)


def test_every_evidence_phrase_lands_inside_a_chunk(handbook_index):
    for q in load_questions(DEFAULT_QUESTIONS):
        if q.get("evidence"):
            assert relevant_chunks(handbook_index, q["evidence"]), q["id"]


def test_retrieval_evaluation_runs(handbook_index):
    report = evaluate_retrieval(handbook_index, load_questions(DEFAULT_QUESTIONS), [("bm25", False), ("rrf", False)])
    assert set(report) == {"bm25", "rrf"}
    assert report["bm25"]["recall@5_keyword"] == 1.0
    assert 0 < report["rrf"]["mrr@10"] <= 1


def test_answer_evaluation_scores_cases(handbook_index):
    questions = [
        {"id": 1, "type": "keyword", "question": "When was the station founded?",
         "evidence": "was founded in 1987", "must_include": ["1987"]},
        {"id": 2, "type": "unanswerable", "question": "Is there Wi-Fi in the cabins?"},
    ]

    def responder(kwargs):
        prompt = kwargs["messages"][-1]["content"]
        if "Respond with JSON only" in prompt:
            return json.dumps({"supported": True, "unsupported_claims": []})
        if "Wi-Fi" in prompt:
            return "I couldn't find this in the book."
        source = next(i for i, part in enumerate(prompt.split("[Source ")[1:], 1) if "founded in 1987" in part)
        return f"It was founded in 1987 [Source {source}]."

    report = evaluate_answers(handbook_index, questions, FakeClient(responder))
    assert report["answer_accuracy"] == 1.0
    assert report["abstention_on_unanswerable"] == 1.0
    assert report["false_abstention_rate"] == 0.0
    assert report["citation_rate"] == 1.0
    assert report["citation_validity"] == 1.0
    assert report["citation_support"] == 1.0
    assert report["faithfulness"] == 1.0
    assert report["tokens_per_answer"] == 42
    assert answer_issues(report["cases"]) == []


def test_issues_list_flags_each_problem():
    base = {"question": "q", "answer": "a | b", "cited": [1], "abstained": False, "correct": True}
    cases = [
        {**base, "id": 1, "type": "keyword"},
        {**base, "id": 2, "type": "keyword", "correct": False},
        {**base, "id": 3, "type": "keyword", "abstained": True},
        {**base, "id": 4, "type": "keyword", "cited": []},
        {**base, "id": 5, "type": "unanswerable"},
        {**base, "id": 6, "type": "unanswerable", "abstained": True},
    ]
    labels = [line.split("**")[1] for line in answer_issues(cases)]
    assert labels == ["wrong answer", "false abstention", "no citation detected", "did not abstain"]
