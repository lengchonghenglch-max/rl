# RF Knowledge Corpus

This directory contains the RAG handoff corpus from 2026-11-13.

- `detail/` contains page-marked extracted full text; `index.json` is the retrieval index.
- `paper/` contains the 25 source PDFs used to prepare the extracted text.
- `detail_zh/` contains optional Chinese translations for 12 documents.
- `evaluation/` contains the 30-question retrieval set and BM25 baseline reports.
- `manifest.csv` records document metadata, hashes, source URLs, and license notes.
- `KB_VERSION.json` records the frozen corpus and baseline metrics.

The app reads `index.json` directly and works offline. To rebuild it from source files,
call `rf_agent.rag.ingest(path)` with a Markdown or PDF file (or a directory of them).
Retrieval returns cited `Evidence` items. A result marked as a knowledge suggestion
requires engineer confirmation before it is treated as a design constraint.

The handoff reports baseline Recall@5 of 0.8148 and MRR@5 of 0.6111 across 30
questions. These figures are the handoff's recorded baseline, not a claim that every
retrieval run or generated answer has been independently validated.
