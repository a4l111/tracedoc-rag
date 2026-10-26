# TraceDoc-RAG

Python 3.10+ project with local parsing and indexing, MoArk Qwen3-Embedding-8B, and DeepSeek API question answering. DeepSeek is not used for embeddings.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
New-Item -ItemType Directory -Force data/raw
```

Copy source files into `data/raw/`, then run:

```powershell
python app.py parse
```

For optional full-page PDF renders:

```powershell
python app.py parse --render-pages
```

The command recursively scans PDF, DOCX, PPTX, TXT and common raster image extensions. Legacy `.ppt` files are not supported. Image content is verified with Pillow, rather than trusting its extension. Each source gets a stable path-based ID and an output folder under `data/parsed/` containing `document.md`, `images/`, `manifest.json`, and `errors.jsonl`. PDF Markdown uses `<!-- PAGE:n -->` and converts detected tables to Markdown. DOCX Markdown uses paragraph IDs, includes embedded raster images, converts simple tables to Markdown and merged cells to HTML, and has no invented page numbers. PPTX follows the presentation's slide order, uses `<!-- SLIDE:n -->`, and extracts text, speaker notes and embedded raster images without inventing PDF page numbers. PDF, DOCX and PPTX image assets have IDs and available source locations or nearby text; captions are left empty when not identified. TXT encoding and character count are recorded in its manifest. Standalone images, DOCX images and PPTX images are read by local RapidOCR, with extracted text labeled `<!-- OCR:local ... -->` in Markdown. PDF table detection and OCR can miss or misread complex layouts; check the source before relying on them. Failed documents get an error record, and parsing continues with the remaining files. The CLI returns exit code 1 if any file fails.

## Build indexes

`build` reads only `data/parsed/`; it never reparses `data/raw/`. Choose one chunk mode per snapshot:

```powershell
python app.py build                         # structured headings, then 400/50 windows
python app.py build --chunk-mode fixed      # 400 characters, 50 overlap
```

Set `MOARK_API_KEY` in `.env` before building. The token must have access to `Qwen3-Embedding-8B` and `Qwen3-Reranker-8B` in MoArk. Build requests 4096-dimensional vectors and sends each chunk's text to MoArk; query sends the question and rerank candidates. Each build creates `data/indexes/{snapshot_id}/` with one `chunks.jsonl` shared by BM25, MoArk Qwen3-Embedding-8B and local bge-m3. `chunk_id` is a stable machine ID; `display_name` is for people. Page fields are populated only when the parsed Markdown has real PDF page markers. Query selects the latest complete MoArk snapshot by default or accepts its exact ID. Older Qwen snapshots from other providers cannot be queried; rebuild to use MoArk:

```powershell
python app.py query "问题"                         # latest complete MoArk snapshot; answer and citations
python app.py query "问题" --snapshot-id SNAPSHOT_ID
python app.py query "问题" --retrieval-only        # Top 5, no DeepSeek call; MoArk is still used
```

Query takes Top 20 from each route, fuses ranks with RRF (k=60), keeps Top 50, then calls MoArk `Qwen3-Reranker-8B` in batches of at most 20 candidates to select Top 5. It prints each chunk's ID, source, page or heading, reranker score, and text, then calls DeepSeek for a cited answer unless `--retrieval-only` is set. Both MoArk calls use `https://api.moark.com/v1`, a 30-second timeout, and two retries. MoArk free-trial tokens can run out during Build or Query; cross-batch reranker score comparability still needs evaluation with labeled questions. The local bge-m3 embedding model loads from local files or the Hugging Face cache. If official bge-m3 files are placed in `data/models/bge-m3/`, build and query use that local checkpoint and record the weight checksum in the snapshot metadata. Use `python app.py build --download-models` only if local bge-m3 weights are unavailable. Failed embeddings are skipped and recorded per chunk in `errors.jsonl`; they are never replaced with zero vectors. `build_report.json` marks a snapshot incomplete when a route has missing vectors. An empty FAISS file is a valid empty index, not a usable semantic search index. Query refuses an empty route.

## Ask with DeepSeek

Create `.env` from `.env.example`, then set `DEEPSEEK_API_KEY` to your own key. Use a complete snapshot ID from `build`:

```powershell
python app.py ask SNAPSHOT_ID "这篇论文讨论什么问题？"
```

`ask` uses the same three-route Query pipeline and MoArk reranker, then sends its final Top 5 chunks as numbered Context to `deepseek-chat` through the OpenAI Python SDK (`https://api.deepseek.com`). The API key comes from `DEEPSEEK_API_KEY`; `.env` is loaded into the environment when present. The request has a 30-second timeout and two automatic retries. If the key is missing or the API fails, the CLI still prints the retrieved Top 5. The answer lists only source numbers used in its citations. Chunks that end mid-sentence are marked in Context, and the prompt tells DeepSeek to omit their unfinished clauses. Review the cited chunks before relying on an answer: model output and OCR can be wrong. Neither prompts nor raw API responses are written to disk. `ask`, default `query`, and `all` send selected document text to DeepSeek. `build` sends chunks to MoArk; `query`, `ask`, `all`, and `evaluate` send the question and rerank candidates to MoArk. `parse` and `compare-chunks` remain local. `query --retrieval-only` avoids DeepSeek but still calls MoArk for embedding and reranking.

## Comparison and evaluation

Build both modes from the same existing parsed artifacts, then compare their chunk distributions:

```powershell
python app.py build --chunk-mode structured
python app.py build --chunk-mode fixed
python app.py compare-chunks STRUCTURED_SNAPSHOT FIXED_SNAPSHOT
```

`compare-chunks` writes a structural comparison to `data/evaluation/` without making quality claims. For actual Recall@5 and MRR, prepare a human-labeled UTF-8 JSONL file. Each line needs a question and at least one relevant source location:

```json
{"question":"你的测试问题","relevant":[{"doc_id":"实际 manifest 中的 doc_id","page":7}]}
```

For PPTX, use `{"doc_id":"...","slide":2}` instead of `page`. Omit `page` or `slide` only when the whole document is relevant; for a snapshot-specific label, use `{"chunk_id":"实际 chunk_id"}` instead. A document/location label can be reused across fixed and structured snapshots. Run:

```powershell
python app.py evaluate GOLD.jsonl --snapshot-id STRUCTURED_SNAPSHOT --snapshot-id FIXED_SNAPSHOT --k 5
python app.py test GOLD.jsonl --snapshot-id STRUCTURED_SNAPSHOT --snapshot-id FIXED_SNAPSHOT --k 5
```

The report includes BM25, Dense, bge-m3, and reranked hybrid results for the same questions, along with missed labels and gold/chunk checksums for reproducibility. Recall@K is the mean fraction of human-labeled source locations reached by at least one chunk per question (K can be 1–5); MRR uses the first relevant hit. No quality score is reported without labels.

### Local MMLongBench-Doc labels

If the official benchmark checkout is at `MMLongBench-Doc/`, run:

```powershell
python app.py prepare-mmlongbench --max-documents 3
```

This reads PDFs and `data/samples.json` from the downloaded benchmark, parses the first three PDF names into `data/benchmarks/mmlongbench/parsed/`, and writes `gold.jsonl`, `unanswerable.jsonl`, and `import_report.json`. It does not use or modify the project's `data/raw/`, and makes no model API calls. Benchmark evidence page indexes start at zero; the converter adds one to match `<!-- PAGE:n -->` in parsed Markdown. Empty evidence lists are kept separately as unanswerable questions and are excluded from retrieval Recall/MRR. The default is only a deterministic pilot, not a representative score for the full benchmark. Use `--max-documents 135` for all locally available PDFs after checking disk space and model quota.

To build and evaluate this isolated subset once MoArk quota is available:

```powershell
python app.py build --parsed-dir data/benchmarks/mmlongbench/parsed --indexes-dir data/benchmarks/mmlongbench/indexes
python app.py test data/benchmarks/mmlongbench/gold.jsonl --indexes-dir data/benchmarks/mmlongbench/indexes --snapshot-id SNAPSHOT_ID
```

Before using any model quota, run the local BM25 and chunking baseline:

```powershell
python app.py evaluate-local
```

On the three-document pilot this produced 352 chunks in each mode. Fixed and structured modes both measured Recall@5 `0.0529`, Recall@20 `0.4471`, and MRR@5 `0.0735`. None of the parsed chunks had a Markdown heading, so structured chunking had no heading boundary to use and behaved like the fixed window strategy. These are measured pilot BM25 results only; they are not Dense, RRF, reranker, or answer-quality scores.

The pilot produced 352 structured chunks from 48 pages across 3 PDFs, so even this small Build would send hundreds of chunks to MoArk. One original benchmark row has an evidence page outside its 17-page PDF and is listed under `invalid_labels` instead of being silently corrected. The evaluator measures whether any retrieved chunk reaches each human-labeled source page; multiple chunks from one page count as one location. If parsing yields no chunk for a labeled page, that label counts as missed. Answer correctness and abstention on `unanswerable.jsonl` need separate evaluation.

## Run every stage

```powershell
python app.py all "问题"
```

`all` parses `data/raw/` into a new run-specific folder under `data/parsed/_runs/`, builds a new snapshot, then retrieves and answers. It sends the final Top 5 document excerpts to DeepSeek. The existing parsed artifacts and snapshots are not overwritten.

## Shared LAN web interface

Install the added FastAPI, Uvicorn and multipart dependencies with `pip install -r requirements.txt`. Start locally with `python app.py serve`, then open `http://127.0.0.1:8000`. For a trusted LAN, bind to this computer's LAN address, for example `python app.py serve --host 192.168.1.10 --port 8000`, and open that address from another device. Do not expose this unauthenticated service to the internet. Everyone who can reach it can upload and view the shared documents, indexes and answers.

The page can upload a file (maximum 50 MB) and parse it, parse files already in `data/raw/`, build a new fixed or structured snapshot, list documents and saved jobs, and query a complete snapshot with or without the DeepSeek answer. Jobs are serialized and saved under `data/service/jobs/`; queued or running jobs are marked interrupted after a service restart and can be started again manually. Build reports and old snapshots are preserved. The service shows a truthful build stage but cannot show per-chunk model progress until the existing Build pipeline exposes a progress callback. `data/service/events.jsonl` stores only event type, success and elapsed time for basic usage metrics; it does not store questions or raw model responses. Token usage, per-user feedback and document categories are not implemented yet. The CLI remains available independently.

The [AGENTS.md](AGENTS.md) project rules describe evidence and citation handling for future changes. They are development guidance; the runtime DeepSeek constraints remain in `src/qa/answer.py`.

`data/raw/`, `data/parsed/`, `data/indexes/`, `data/models/`, `data/evaluation/`, and `.env` are ignored by Git because they may contain private material or large local model files.
