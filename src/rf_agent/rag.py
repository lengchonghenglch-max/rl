"""Offline BM25 retrieval over the versioned RF knowledge corpus."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rf_agent.contracts import Evidence, SourceType

_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_DIR = _ROOT / "data" / "knowledge"
INDEX_PATH = KNOWLEDGE_DIR / "index.json"
MIN_RELEVANCE_SCORE = 14.0811
_INDEX_CACHE: tuple[Path, int, list[dict[str, Any]], _BM25] | None = None
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_\-.]*|\d+(?:\.\d+)?")
_CJK = re.compile(r"[\u4e00-\u9fff]+")
_STOPWORDS = {
    "the", "a", "an", "of", "to", "and", "or", "in", "on", "for", "is", "are",
    "be", "with", "as", "at", "by", "it", "this", "that", "from", "we", "can", "will",
}
_PAGE_MARKER = re.compile(r"^\s*<!--\s*page:\s*(\d+)\s*-->\s*$", re.IGNORECASE)


def _tokenize(text: str) -> list[str]:
    lowered = text.lower()
    tokens = [word for word in _WORD.findall(lowered) if word not in _STOPWORDS]
    for run in _CJK.findall(lowered):
        tokens.extend([run] if len(run) == 1 else [run[i : i + 2] for i in range(len(run) - 1)])
    return tokens


class _BM25:
    def __init__(self, documents: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.term_counts: list[dict[str, int]] = []
        self.lengths: list[int] = []
        document_frequency: dict[str, int] = {}
        for tokens in documents:
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            self.term_counts.append(counts)
            self.lengths.append(len(tokens))
            for token in counts:
                document_frequency[token] = document_frequency.get(token, 0) + 1
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        total = len(self.lengths)
        self.idf = {
            token: math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

    def scores(self, query_tokens: Sequence[str]) -> list[float]:
        scores: list[float] = []
        for counts, length in zip(self.term_counts, self.lengths):
            norm = self.k1 * (1 - self.b + self.b * length / max(self.average_length, 1e-9))
            score = 0.0
            for token in set(query_tokens):
                frequency = counts.get(token, 0)
                if frequency:
                    score += self.idf[token] * frequency * (self.k1 + 1) / (frequency + norm)
            scores.append(score)
        return scores


def _read_index(index_path: Path | None = None) -> list[dict[str, Any]]:
    index_path = index_path or INDEX_PATH
    if not index_path.is_file():
        return []
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    chunks = payload.get("chunks", [])
    return [chunk for chunk in chunks if chunk.get("text") and chunk.get("doc_id")]


def _get_retriever() -> tuple[list[dict[str, Any]], _BM25]:
    global _INDEX_CACHE
    path = INDEX_PATH
    if not path.is_file():
        return [], _BM25([])
    modified = path.stat().st_mtime_ns
    if _INDEX_CACHE is None or _INDEX_CACHE[:2] != (path, modified):
        chunks = _read_index(path)
        bm25 = _BM25([_tokenize(str(chunk["text"])) for chunk in chunks])
        _INDEX_CACHE = (path, modified, chunks, bm25)
    return _INDEX_CACHE[2], _INDEX_CACHE[3]


def _split_text(text: str, doc_id: str, chunk_chars: int = 700, overlap: int = 120) -> list[dict[str, Any]]:
    page = 1
    page_text: list[str] = []
    pages: list[tuple[int, str]] = []
    for line in text.splitlines():
        match = _PAGE_MARKER.match(line)
        if match:
            if page_text:
                pages.append((page, "\n".join(page_text).strip()))
                page_text = []
            page = int(match.group(1))
        else:
            page_text.append(line)
    if page_text:
        pages.append((page, "\n".join(page_text).strip()))

    chunks: list[dict[str, Any]] = []
    for page_number, page_content in pages:
        start = 0
        while start < len(page_content):
            end = min(start + chunk_chars, len(page_content))
            if end < len(page_content):
                boundary = page_content.rfind("\n", start + chunk_chars // 2, end)
                if boundary > start:
                    end = boundary
            content = page_content[start:end].strip()
            if content:
                chunks.append({
                    "doc_id": doc_id,
                    "page": page_number,
                    "chunk_id": f"{doc_id}-{page_number}-{len(chunks)}",
                    "text": content,
                })
            if end >= len(page_content):
                break
            start = max(start + 1, end - overlap)
    return chunks


def _load_source(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() in {".md", ".markdown", ".txt"}:
        return _split_text(path.read_text(encoding="utf-8"), path.stem)
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Unsupported knowledge file: {path}")
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PDF ingestion requires PyMuPDF (pip install pymupdf)") from exc
    doc_id = path.stem.split("__", 1)[0]
    chunks: list[dict[str, Any]] = []
    with fitz.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            page_text = f"<!-- page: {page_number} -->\n{page.get_text()}"
            chunks.extend(_split_text(page_text, doc_id, chunk_chars=700, overlap=120))
    return chunks


def ingest(path: str | Path) -> list[Evidence]:
    """Ingest Markdown/text/PDF files and update the shared JSON chunk index."""
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    files = sorted(p for p in source.rglob("*") if p.is_file()) if source.is_dir() else [source]
    parsed: list[dict[str, Any]] = []
    selected: dict[str, Path] = {}
    for file_path in files:
        if file_path.suffix.lower() in {".md", ".markdown", ".txt", ".pdf"}:
            if "detail_zh" in file_path.parts or "evaluation" in file_path.parts:
                continue
            if file_path.name in {"README.md", "HANDOFF_README.md"}:
                continue
            doc_id = file_path.stem.split("__", 1)[0]
            previous = selected.get(doc_id)
            is_text = file_path.suffix.lower() in {".md", ".markdown", ".txt"}
            previous_is_text = previous is not None and previous.suffix.lower() in {".md", ".markdown", ".txt"}
            if previous is None or (is_text and not previous_is_text):
                selected[doc_id] = file_path
    for file_path in selected.values():
        parsed.extend(_load_source(file_path))
    if not parsed:
        return []

    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    current = _read_index()
    doc_ids = {chunk["doc_id"] for chunk in parsed}
    chunks = [chunk for chunk in current if chunk["doc_id"] not in doc_ids] + parsed
    payload = {"version": 1, "chunks": chunks}
    INDEX_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    global _INDEX_CACHE
    _INDEX_CACHE = None
    return [_as_evidence(chunk) for chunk in parsed]


def _as_evidence(chunk: dict[str, Any], score: float = 0.0) -> Evidence:
    return Evidence(
        document_id=str(chunk["doc_id"]),
        page=int(chunk["page"]) if chunk.get("page") is not None else None,
        chunk_id=str(chunk["chunk_id"]),
        quote=str(chunk["text"]),
        score=float(score),
        source_type=SourceType.KNOWLEDGE_SUGGESTION,
        needs_confirmation=True,
    )


def _retrieve(query: str, k: int, min_score: float) -> list[Evidence]:
    if k < 1:
        raise ValueError("k must be at least 1")
    query_tokens = _tokenize(query.strip())
    if not query_tokens:
        return []
    chunks, bm25 = _get_retriever()
    if not chunks:
        return []
    scores = bm25.scores(query_tokens)
    if max(scores, default=0.0) < min_score:
        return []
    ranked = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))
    return [_as_evidence(chunks[index], scores[index]) for index in ranked if scores[index] > 0][:k]


def retrieve(query: str, k: int = 5) -> list[Evidence]:
    """Return cited results, refusing queries below the handoff relevance threshold."""
    return _retrieve(query, k, MIN_RELEVANCE_SCORE)


def evaluate_recall(test_questions: list[tuple[str, str]], k: int = 5) -> dict[str, float]:
    """Measure document Recall@k and MRR for (query, expected_document_id) pairs."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if not test_questions:
        return {f"recall@{k}": 0.0, f"mrr@{k}": 0.0}
    reciprocal_ranks: list[float] = []
    hits = 0
    for query, expected_doc_id in test_questions:
        results = _retrieve(query, k, min_score=0.0)
        rank = next((i for i, result in enumerate(results, start=1)
                     if result.document_id == expected_doc_id), None)
        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
    return {
        f"recall@{k}": hits / len(test_questions),
        f"mrr@{k}": sum(reciprocal_ranks) / len(test_questions),
    }
