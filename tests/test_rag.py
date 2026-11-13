from __future__ import annotations

import csv
import json
from pathlib import Path

from rf_agent import rag
from rf_agent.contracts import SourceType


def test_retrieve_returns_ranked_citations():
    results = rag.retrieve("Latin hypercube sampling expansion algorithm", k=3)

    assert results
    assert results[0].document_id == "a05"
    assert results[0].page == 1
    assert results[0].chunk_id
    assert "LHS" in results[0].quote
    assert results[0].source_type == SourceType.KNOWLEDGE_SUGGESTION
    assert results[0].needs_confirmation is True


def test_retrieve_returns_no_evidence_without_matching_terms():
    assert rag.retrieve("qzxvplm 839204", k=5) == []
    assert rag.retrieve("", k=5) == []


def test_retrieve_refuses_handoff_negative_questions():
    questions_path = Path(rag.KNOWLEDGE_DIR) / "evaluation" / "eval_questions.csv"
    with questions_path.open(encoding="utf-8-sig", newline="") as handle:
        negatives = [row["question"] for row in csv.DictReader(handle) if row["type"] == "negative"]

    assert len(negatives) == 3
    assert all(rag.retrieve(question, k=5) == [] for question in negatives)


def test_ingest_markdown_preserves_page_and_writes_index(tmp_path, monkeypatch):
    knowledge_dir = tmp_path / "knowledge"
    source = tmp_path / "sample.md"
    source.write_text(
        "<!-- page: 7 -->\nA resonator uses a spiral inductor and a capacitor.",
        encoding="utf-8",
    )
    monkeypatch.setattr(rag, "KNOWLEDGE_DIR", knowledge_dir)
    monkeypatch.setattr(rag, "INDEX_PATH", knowledge_dir / "index.json")
    monkeypatch.setattr(rag, "MIN_RELEVANCE_SCORE", 0.0)

    evidence = rag.ingest(source)

    assert len(evidence) == 1
    assert evidence[0].document_id == "sample"
    assert evidence[0].page == 7
    assert evidence[0].chunk_id == "sample-7-0"
    assert json.loads((knowledge_dir / "index.json").read_text(encoding="utf-8"))["chunks"]
    assert rag.retrieve("spiral inductor", k=1)[0].document_id == "sample"


def test_evaluate_recall_reports_recall_and_mrr():
    metrics = rag.evaluate_recall(
        [("Latin hypercube sampling", "a05"), ("Gaussian process surrogate", "a01")],
        k=5,
    )

    assert metrics["recall@5"] == 1.0
    assert 0 < metrics["mrr@5"] <= 1.0
