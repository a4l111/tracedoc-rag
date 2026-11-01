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

For image-only PDF pages, run local RapidOCR on a full-page render:

```powershell
python app.py parse --ocr-scanned-pages
```

This option only OCRs PDF pages with no native text. It stores the rendered page as an asset and labels the extracted text `review_status=unreviewed`; verify names and numbers against the page image. It does not call MoArk. Parsing into an existing output directory replaces that document's parsed files, so use a separate `--output-dir` when comparing results. An existing Build snapshot is unchanged; run Build again to index newly parsed OCR text.

### Evidence quality limits

- Page-level Recall@5 measures whether a retrieved chunk points to a labeled page. It does not prove that the final Top 5 contains every number needed to answer. The original saved 18-question report has hybrid Recall@5 0.6944 versus Dense 0.7222. A separate, user-confirmed review of question 12 changes its evidence pages from `[8, 10]` to `[10]`, making both routes 0.7222 under that derived label; those original final chunks omit 46% and 39%. A later complete snapshot (`20261029T155325Z-5c7d8346`) adds source-grounded page context to long table chunks. Its Q12 final Top 5 includes one chunk with the chart title and both values; the reviewed-label 18-question rerun has hybrid Recall@5 0.7500 and MRR 0.6852. These are separate snapshots and reports. Answer “insufficient evidence” whenever the selected chunks lack the necessary values.
- Local OCR on image-only PDF pages can recover search terms, but may merge words or miss small print. Its output is marked `unreviewed`; verify names, dates, and numbers against the saved page render before citing them. A revised parse requires a new Build snapshot.
- MoArk vision descriptions are linked to image assets and pages, yet are also `unreviewed` until a person checks the chart legend, units, labels, and numeric values against the original image. Do not present an unreviewed description as verified source text. The audit steps and observed failures are in `data/validation/parse_cases/CASE_AUDIT.md`.

Visual descriptions are opt-in because they send extracted images to MoArk and consume model quota:

```powershell
python app.py parse --describe-images
python app.py parse --render-pages --describe-images
```

The visual pass applies to real image assets extracted from PDF, DOCX, PPTX and standalone images. Each successful description is inserted beside its original Markdown image reference and saved under that asset's manifest metadata with provider, model, prompt version, source page or slide when available, and `review_status=unreviewed`. Raw model responses are not persisted. One failed image records a `vision_description` error and does not discard the locally parsed text or the remaining images. PDF page renders are described only when `--render-pages` is also selected.

The command recursively scans PDF, DOCX, PPTX, XLSX, TXT and common raster image extensions. Legacy `.ppt` files are not supported. Image content is verified with Pillow, rather than trusting its extension. Each source gets a stable path-based ID and an output folder under `data/parsed/` containing `document.md`, `images/`, `manifest.json`, and `errors.jsonl`. PDF Markdown uses `<!-- PAGE:n -->` for the physical PDF page number, counting from 1, and converts detected tables to Markdown. DOCX Markdown uses paragraph IDs, includes embedded raster images, converts simple tables to Markdown and merged cells to HTML, and has no invented page numbers. PPTX follows the presentation's slide order, uses `<!-- SLIDE:n -->`, and extracts text, speaker notes and embedded raster images without inventing PDF page numbers. PDF, DOCX and PPTX image assets have IDs and available source locations or nearby text; captions are left empty when not identified. TXT encoding and character count are recorded in its manifest. Standalone images, DOCX images and PPTX images are read by local RapidOCR, with extracted text labeled `<!-- OCR:local ... -->` in Markdown. PDF table detection and OCR can miss or misread complex layouts; check the source before relying on them. Failed documents get an error record, and parsing continues with the remaining files. The CLI returns exit code 1 if any file fails.

XLSX files are also supported. The parser reads saved cell values only (no formula execution or external links), writes sheet headings and original row numbers to Markdown, and records sheet dimensions in `manifest.json`. It reads at most 5,000 rows and 100 columns per sheet, recording a limit error if truncated. Formula cells without cached results remain blank; charts, macros and legacy `.xls` files are not parsed.

## Build indexes

`build` reads only `data/parsed/`; it never reparses `data/raw/`. Choose one chunk mode per snapshot:

```powershell
python app.py build                         # structured headings, then 400/50 windows
python app.py build --chunk-mode fixed      # 400 characters, 50 overlap
```

Set `MOARK_API_KEY` in `.env` before building. The token must have access to `Qwen3-Embedding-8B` and `Qwen3-Reranker-8B` in MoArk. Build requests 4096-dimensional vectors and sends each chunk's text to MoArk; query sends the question and rerank candidates. Each build creates `data/indexes/{snapshot_id}/` with one `chunks.jsonl` shared by BM25, MoArk Qwen3-Embedding-8B and local bge-m3. `chunk_id` is a stable machine ID; `display_name` is for people. Page fields are populated only when the parsed Markdown has physical PDF page markers. Query selects the latest complete MoArk snapshot by default or accepts its exact ID. Older Qwen snapshots from other providers cannot be queried; rebuild to use MoArk:

```powershell
python app.py query "问题"                         # latest complete MoArk snapshot; answer and citations
python app.py query "问题" --snapshot-id SNAPSHOT_ID
python app.py query "问题" --retrieval-only        # Top 5, no DeepSeek call; MoArk is still used
```

Query takes Top 20 from each route, fuses ranks with RRF (k=60), keeps Top 50, then calls MoArk `Qwen3-Reranker-8B` in batches of at most 20 candidates to select Top 5. Candidate sets larger than 20 use a versioned two-stage tournament so the finalists are compared in one request. It prints each chunk's ID, source, page or heading, reranker score, and text, then calls DeepSeek for a cited answer unless `--retrieval-only` is set. Both MoArk calls use `https://api.moark.com/v1`, a 30-second timeout, and two retries. The completed labeled evaluation found that this strategy still underperforms the Dense route; the measured failure is documented below rather than hidden. The local bge-m3 embedding model loads from local files or the Hugging Face cache. If official bge-m3 files are placed in `data/models/bge-m3/`, build and query use that local checkpoint and record the weight checksum in the snapshot metadata. Use `python app.py build --download-models` only if local bge-m3 weights are unavailable. Failed embeddings are recorded per chunk in `errors.jsonl`; they are never replaced with zero vectors. Build retains only Chunk IDs with valid embeddings in both vector routes, then rebuilds BM25 and both FAISS indexes on that same effective set. Excluded IDs and the action taken are recorded in `repair_log.jsonl`. `build_report.json` distinguishes a usable aligned snapshot (`complete`) from one with exclusions (`partial`). If no ID survives, the snapshot is incomplete and Query refuses it. Web retry jobs retain `retried_from` to link a repair attempt to its earlier job.

Build also writes an optional, independent image-vector index when MoArk `Qwen3-VL-Embedding-8B` is available. Every unique image asset across `data/parsed/` is embedded to a normalized 2048-dimensional vector and stored as `visual_index.faiss` plus `visual_images.jsonl` and `visual_meta.json`. This route is additive: a snapshot stays `complete` for text retrieval even when the visual embedding model is unavailable or a single image fails, and failures are recorded in `visual_errors.jsonl` without discarding text chunks. Visual vectors only add a text-to-image search route; they never alter text chunks or their page/slide locations.

## Ask with DeepSeek

Create `.env` from `.env.example`, then set `DEEPSEEK_API_KEY` to your own key. Use a complete snapshot ID from `build`:

```powershell
python app.py ask SNAPSHOT_ID "这篇论文讨论什么问题？"
```

`ask` uses the same three-route Query pipeline and MoArk reranker, then sends its final Top 5 chunks as numbered Context to `deepseek-chat` through the OpenAI Python SDK (`https://api.deepseek.com`). The API key comes from `DEEPSEEK_API_KEY`; `.env` is loaded into the environment when present. The request has a 30-second timeout and two automatic retries. If the key is missing or the API fails, the CLI still prints the retrieved Top 5. The answer lists only source numbers used in its citations. Chunks that end mid-sentence are marked in Context, and the prompt tells DeepSeek to omit their unfinished clauses. Review the cited chunks before relying on an answer: model output and OCR can be wrong. Neither prompts nor raw API responses are written to disk. `ask`, default `query`, and `all` send selected document text to DeepSeek. `build` sends chunks to MoArk; `query`, `ask`, `all`, and `evaluate` send the question and rerank candidates to MoArk. `parse` and `compare-chunks` remain local. `query --retrieval-only` avoids DeepSeek but still calls MoArk for embedding and reranking.

## Spreadsheet calculation, diagrams and slide structure

These structured-answer features run over the saved parse artifacts and are exposed through the Web page. They keep model output separate from verified source text and never present a generated figure or computed total as if it were a retrieved source citation.

- **Spreadsheet calculation (`/api/table/plan`, `/api/table/execute`).** For a parsed XLSX, DeepSeek proposes a declarative JSON query plan (`count`, `distinct_count`, `group_count`, `sum`, `avg`, `min`, `max`, `numeric_summary`, `top`) with at most six filter conditions. It never sees row data and never runs code. The plan is signed with an HMAC and must be confirmed before execution, then it is revalidated and computed deterministically on this machine from the saved `spreadsheet.json` snapshot. The signature, a 15-minute expiry, and source/snapshot hashes are checked at execution, so a tampered or stale plan is rejected. Formulas are not executed; the file's cached values are used, empty and invalid cells are excluded and reported, and percentages and thousand separators are converted by exact decimal arithmetic. A spreadsheet can be browsed cell-by-cell at `/api/source/{doc_id}/spreadsheet`.
- **Diagram generation (`/api/diagram`).** Given a question, DeepSeek returns a small declarative JSON graph (`description`, `nodes`, `edges`) from the retrieved Context. `graph_description` validates the structure, deduplicates node IDs, rejects edges to missing nodes, escapes Mermaid entity characters, and emits a safe `flowchart TD` diagram. The output is always labeled model-generated and not manually verified; non-JSON responses pass through as ordinary text rather than a diagram. The page renders the Mermaid diagram locally (bundled under `src/service/vendor/`).
- **Slide structure preview (`/api/source/{doc_id}/ppt-structure/{slide}`).** For a parsed PPTX, this reconstructs a labeled structural thumbnail and a Mermaid graph directly from the slide XML (shape positions, text, group transforms, and explicit connector endpoints). It is a structural preview, not an Office slide render: connector direction is never inferred, and connectors are marked undirected.

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

This reads PDFs and `data/samples.json` from the downloaded benchmark, parses the first three PDF names into `data/benchmarks/mmlongbench/parsed/`, and writes `gold.jsonl`, `unanswerable.jsonl`, and `import_report.json`. It does not use or modify the project's `data/raw/`, and makes no model API calls. The importer retains each source `evidence_pages` label unchanged and records it in Gold. These labels are evaluation evidence locations, not a guaranteed global PDF numbering convention; rows outside the PDF physical-page range are flagged for review. The parser and Query citations independently use one-based physical PDF pages. Verify label-to-page correspondence before interpreting Recall/MRR. Empty evidence lists are kept separately as unanswerable questions and are excluded from retrieval Recall/MRR. The default is only a deterministic pilot, not a representative score for the full benchmark. Use `--max-documents 135` for all locally available PDFs after checking disk space and model quota.

To build and evaluate this isolated subset once MoArk quota is available:

```powershell
python app.py build --parsed-dir data/benchmarks/mmlongbench/parsed --indexes-dir data/benchmarks/mmlongbench/indexes
python app.py test data/benchmarks/mmlongbench/gold.jsonl --indexes-dir data/benchmarks/mmlongbench/indexes --snapshot-id SNAPSHOT_ID
```

For image-heavy benchmark pages, create a separate visual-enriched parsed tree. This renders only human-labeled evidence pages and asks MoArk `Qwen3-VL-30B-A3B-Instruct` for a description. The original parsed tree is not modified. Every inserted description is tied to a real page image and marked `review_status=unreviewed`; the parsed description is saved, but the raw model response is not:

```powershell
python app.py describe-mmlongbench
python app.py build --parsed-dir data/benchmarks/mmlongbench/parsed_vision --indexes-dir data/benchmarks/mmlongbench/indexes_vision
```

The evidence-page selection above uses Gold labels and is suitable for diagnosis only. For a retrieval evaluation without that label leakage, describe **every** page in the selected PDFs, then build and evaluate a separate snapshot:

```powershell
python app.py describe-mmlongbench --all-pages --output-dir data/benchmarks/mmlongbench/parsed_vision_all
python app.py build --parsed-dir data/benchmarks/mmlongbench/parsed_vision_all --indexes-dir data/benchmarks/mmlongbench/indexes_vision_all
python app.py evaluate data/benchmarks/mmlongbench_page_corrected/gold.jsonl --indexes-dir data/benchmarks/mmlongbench/indexes_vision_all --snapshot-id SNAPSHOT_ID
```

The all-page description step does not read Gold labels. Image descriptions remain unreviewed model output; inspect original page images before treating a visual claim as verified evidence.

To make photo-count questions searchable, run a second pass over the **saved page renders** and rebuild a new snapshot. This pass counts visible people per photograph, stores page and asset provenance, and marks every count as unreviewed model output. It does not read the original PDF or Gold labels:

```powershell
python app.py inventory-photos --parsed-dir data/benchmarks/mmlongbench/parsed_vision_all --output-dir data/benchmarks/mmlongbench/parsed_photo_inventory
python app.py build --parsed-dir data/benchmarks/mmlongbench/parsed_photo_inventory --indexes-dir data/benchmarks/mmlongbench/indexes_photo_inventory
python app.py query "Among all the pictures, how many of them contain only one person?" --snapshot-id SNAPSHOT_ID --indexes-dir data/benchmarks/mmlongbench/indexes_photo_inventory
```

Build adds page-scoped photo chunks and, only when every saved page has a complete count, document-wide single-person and maximum-count summary chunks. Each summary lists its evidence pages and source image asset IDs. Crowd counts are estimates and remain marked unreviewed; inspect original page images before asserting an exact number of people.

The three-document pilot produced 22/22 evidence-page descriptions and a complete 415-chunk snapshot, `20261026T093858Z-8329980e`. Local evaluation of the saved BM25 and bge-m3 routes requires no MoArk quota:

```powershell
python app.py evaluate-snapshot-local data/benchmarks/mmlongbench/gold.jsonl --indexes-dir data/benchmarks/mmlongbench/indexes_vision --snapshot-id 20261026T093858Z-8329980e --output-dir data/benchmarks/mmlongbench/evaluation_vision
```

Measured on the 17 answerable pilot questions, BM25 reached Recall@5 `0.1294` and MRR `0.1294`; local bge-m3 reached Recall@5 `0.2294` and MRR `0.1225`. These are route-level retrieval measurements, not Dense, RRF, reranker, or answer-quality results.

Formal Dense and hybrid evaluation calls MoArk for every question. Use a stable resume directory so a quota or network interruption does not repeat completed questions:

```powershell
python app.py test data/benchmarks/mmlongbench/gold.jsonl --indexes-dir data/benchmarks/mmlongbench/indexes_vision --snapshot-id 20261026T093858Z-8329980e --output-dir data/benchmarks/mmlongbench/evaluation_vision --resume-dir data/benchmarks/mmlongbench/evaluation_vision/full-20261026
```

The checkpoint is accepted only when its snapshot, gold checksum, routes, K and reranker strategy match. The completed formal results and remaining quality failure are recorded in the acceptance section below.

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

## Acceptance report

Generate a reproducible, offline acceptance report from saved artifacts. This command does not reparse documents or call a model API:

```powershell
python app.py acceptance `
  --snapshot-id SNAPSHOT_ID `
  --parsed-dir PARSED_DIR `
  --indexes-dir INDEXES_DIR `
  --evaluation-report EVALUATION_REPORT_JSON
```

It checks the required project skeleton, per-document outputs, traceable unreviewed visual descriptions, complete snapshot artifacts, chunk checksum and three-route alignment, labeled route metrics, and whether final hybrid Recall@5 is at least as high as the best individual route. It writes `report.json` and `report.md` under ignored `data/acceptance/` and exits with code 1 when any mandatory check fails. A failed quality check is evidence to investigate, not a reason to hide or rewrite the measured score.

The completed 17-question evaluation on 2026-10-28 still has one failed quality check. The two-stage batch tournament improved hybrid Recall@5 from `0.0588` to `0.0882`, but this remains below Dense Recall@5 `0.3235`. Generate a candidate-coverage and per-question ranking-error report without rerunning the reranker:

Saved local reports are under `data/benchmarks/mmlongbench/evaluation_vision/full-tournament-20261027/`; reviewed-label outputs are under `data/benchmarks/mmlongbench/reviewed_evaluation/`. These ignored data artifacts are not part of the Git repository and must be delivered separately only if the course requires them.

```powershell
python app.py analyze-retrieval GOLD.jsonl `
  --snapshot-id SNAPSHOT_ID `
  --indexes-dir INDEXES_DIR `
  --evaluation-report EVALUATION_REPORT_JSON `
  --output-dir ANALYSIS_DIR
```

This command recreates each route's Top 20 (including a MoArk query embedding), reconstructs RRF Top 50, and compares those candidates with the saved final Top 5. It distinguishes candidate misses from evidence that reached RRF but was removed by reranking.

For selected failures, record the numeric two-stage reranker decisions. This consumes both query-embedding and reranker quota. The output deliberately excludes document text, prompts, and raw API responses:

```powershell
python app.py diagnose-reranker GOLD.jsonl `
  --snapshot-id SNAPSHOT_ID `
  --indexes-dir INDEXES_DIR `
  --evaluation-report EVALUATION_REPORT_JSON `
  --question-number 9 --question-number 12 --question-number 13 `
  --output-dir DIAGNOSTIC_DIR
```

The report shows each relevant candidate's RRF rank, initial batch/rank/score, finalist status, final rank/score, and Top-5 status. Page-only benchmark labels can still disagree with answer-bearing adjacent or spanning chunks; such cases must be reviewed rather than silently relabeled.

Audit that disagreement offline with a conservative normalized exact-substring check:

```powershell
python app.py audit-evaluation-answers GOLD.jsonl `
  --snapshot-id SNAPSHOT_ID `
  --indexes-dir INDEXES_DIR `
  --evaluation-report EVALUATION_REPORT_JSON `
  --output-dir ANSWER_AUDIT_DIR
```

Pure numeric answers are excluded because their accidental occurrence produces false positives. This auxiliary audit may identify cases for manual label review, but it never changes Recall@5, MRR, gold labels, or acceptance status.

Create a separate human-review template (the command refuses to overwrite an existing review file), then apply only confirmed records to a derived gold file and rescore the already saved Top 5 without model calls:

```powershell
python app.py prepare-gold-review GOLD.jsonl `
  --question-number 1 --question-number 5 --question-number 13 `
  --output-path GOLD_REVIEW.jsonl

python app.py apply-gold-review GOLD.jsonl GOLD_REVIEW.jsonl `
  --snapshot-id SNAPSHOT_ID `
  --indexes-dir INDEXES_DIR `
  --evaluation-report EVALUATION_REPORT_JSON `
  --output-dir REVIEWED_EVALUATION_DIR
```

Reviews are matched strictly by `benchmark_row`; question, source, answer and original pages must still match. `gold_reviewed.jsonl` and both JSON/Markdown audit reports are written separately. The original gold file is never overwritten, and original and reviewed metrics are shown together.

## Shared LAN web interface

Install the added FastAPI, Uvicorn and multipart dependencies with `pip install -r requirements.txt`. Start locally with `python app.py serve`, then open `http://127.0.0.1:8000`. For a trusted LAN, bind to this computer's LAN address, for example `python app.py serve --host 192.168.1.10 --port 8000`, and open that address from another device. Do not expose this unauthenticated service to the internet. Everyone who can reach it can upload and view the shared documents, indexes and answers.

For the optional Windows Web onedir package, create an isolated build environment, install both dependency files, and run the reproducible build script:

```powershell
python -m venv .build-venv
.build-venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-build.txt
powershell -ExecutionPolicy Bypass -File scripts\build_windows.ps1
```

The ignored output is `dist/TraceDocRAG/`. It contains a real `TraceDocRAG.exe`, required runtime libraries, public configuration examples and empty data folders, but no model weights, indexes, source documents, conversations or real API keys. See `DISTRIBUTION.md` for the end-user instructions. `start_web.bat` remains a source-distribution fallback that requires an existing Python environment; it is not the standalone executable.

The page can upload a file (maximum 50 MB) and parse it, parse files already in `data/raw/`, build a new fixed or structured snapshot, list and categorize documents, view saved jobs, and query a complete snapshot with or without the DeepSeek answer. Categories are shared labels stored by document ID outside the parsed artifact. Jobs are serialized and saved under `data/service/jobs/`; queued or running jobs are marked interrupted after a service restart. Failed, partial, and interrupted jobs can be retried from their saved safe parameters; legacy jobs without those parameters are rejected instead of guessed. Build reports and old snapshots are preserved. Build shows real chunk counts for chunking, BM25, Qwen3 embedding, and bge-m3 embedding; model loading has no per-chunk progress. Each final Top 5 result links to its source Markdown and same-page or same-slide image assets when the saved snapshot's source hash matches the current parsed manifest. The page can show up to 100 structured parse/build errors without returning chunk text or raw model responses. `data/service/events.jsonl` stores event type, success, elapsed time, and DeepSeek token counts only when the API returns usage; it does not store questions, prompts, document text, keys, or raw model responses. Shared thumbs-up/down feedback stores only turn ID and rating in the local conversation database. The CLI remains available independently.

The page also streams DeepSeek answers as server-sent events (`/api/query/stream`) with a cancel endpoint; a canceled generation is logged as success with no partial answer saved. When a snapshot has a visual index, querying also returns the top image matches (`visual_hits`) beside the text chunks, each with its page/slide and an image URL. A parsed XLSX can be browsed and computed in place, a parsed PPTX can show its per-slide structure preview, and a question can produce a validated Mermaid flowchart. Mermaid and KaTeX are bundled under `src/service/vendor/`, so those renders work offline without a CDN; `GET /api/activity` returns request counts, failure counts, elapsed times, and the DeepSeek token usage actually returned by the API (unknown when the API returns no `usage`).

The Web page can optionally describe images during upload or `data/raw` parsing. It also provides shared conversations. A conversation carries at most the latest 20 completed turns and approximately 48,000 characters into DeepSeek to interpret follow-up questions; retrieval uses the latest six questions as bounded query context. Every turn still searches the selected complete snapshot and obtains a fresh final Top 5. Previous answers and their citation numbers are explicitly untrusted context, never evidence. Only the current turn's Top 5 may be cited.

Shared conversations are stored in ignored `data/service/conversations.sqlite3`. They contain questions, final displayed answers, snapshot IDs and compact evidence locations, but not raw prompts, raw API responses or chunk text. Because the LAN service has no login, every LAN user can currently list, read and delete these conversations. Do not use shared history for private questions until authentication and per-user ownership are implemented.

When the page is opened on the service computer through `127.0.0.1` or `localhost`, it also shows a model and key administration section. Administrative API requests from other LAN addresses are rejected. The roles remain constrained by the project contract: DeepSeek `deepseek-chat` is the final answer model; MoArk `Qwen3-Embedding-8B`, `Qwen3-Reranker-8B`, and `Qwen3-VL-30B-A3B-Instruct` provide embedding, reranking, and visual description. The page can adjust the approved endpoint timeout and save or clear keys, but cannot silently switch providers or model roles.

On Windows, keys entered through the page are encrypted with the current Windows user's DPAPI and stored outside the repository under `%LOCALAPPDATA%\TraceDoc-RAG\secrets.json`. The browser and administration API receive only `environment`, `encrypted_store`, or `missing` status and never receive the saved plaintext. Existing `DEEPSEEK_API_KEY` and `MOARK_API_KEY` environment variables take precedence, so clearing the encrypted copy does not clear an environment variable. On non-Windows systems, keep using environment variables; the first version intentionally refuses to create an unencrypted fallback key file. Non-secret model settings are stored under ignored `data/service/model_settings.json`.

The [AGENTS.md](AGENTS.md) project rules describe evidence and citation handling for future changes. They are development guidance; the runtime DeepSeek constraints remain in `src/qa/answer.py`.

`data/raw/`, `data/parsed/`, `data/indexes/`, `data/models/`, `data/evaluation/`, and `.env` are ignored by Git because they may contain private material or large local model files.
