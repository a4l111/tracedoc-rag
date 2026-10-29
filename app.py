"""Command-line entry point for the multimodal RAG coursework."""

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import typer
from openai import APIError
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from src.parser.pipeline import parse_file, supported_files

if not sys.stdout.isatty() and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

app = typer.Typer(help="Multimodal RAG coursework CLI.", no_args_is_help=True)
console = Console()


@app.callback()
def main() -> None:
    """Multimodal RAG commands."""


@app.command()
def parse(
    raw_dir: Path = typer.Option(Path("data/raw"), help="Input document directory."),
    output_dir: Path = typer.Option(Path("data/parsed"), help="Parsed output directory."),
    render_pages: bool = typer.Option(False, help="Also render each PDF page as a PNG."),
    describe_images: bool = typer.Option(False, help="Send extracted images to MoArk Qwen3-VL."),
    ocr_scanned_pages: bool = typer.Option(False, help="Run local OCR on PDF pages with no native text; results are unreviewed."),
) -> None:
    """Parse PDF, DOCX, PPTX, TXT and image files under data/raw/."""
    if not raw_dir.is_dir():
        console.print(f"[red]Input directory does not exist:[/red] {raw_dir}")
        raise typer.Exit(code=1)

    files = supported_files(raw_dir)
    if not files:
        console.print(f"No supported files found in {raw_dir}")
        return

    succeeded = 0
    failed = 0
    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Parsing", total=len(files))
        for source in files:
            progress.update(task, description=f"Parsing {source.name}")
            result = parse_file(source, raw_dir, output_dir, render_pages, describe_images, ocr_scanned_pages)
            if result:
                succeeded += 1
            else:
                failed += 1
            progress.advance(task)
    console.print(f"Parsed: {succeeded}; failed: {failed}; output: {output_dir}")
    if failed:
        raise typer.Exit(code=1)


@app.command()
def build(
    parsed_dir: Path = typer.Option(Path("data/parsed"), help="Read existing parse artifacts only."),
    indexes_dir: Path = typer.Option(Path("data/indexes"), help="Snapshot output directory."),
    chunk_mode: str = typer.Option("structured", help="structured or fixed."),
    download_models: bool = typer.Option(False, help="Allow downloading the local bge-m3 model."),
) -> None:
    """Build one chunk snapshot and BM25, Qwen3-Embedding-8B and bge-m3 indexes."""
    from src.index.build import build as build_snapshot

    try:
        folder = build_snapshot(parsed_dir, indexes_dir, chunk_mode, download_models)
    except ValueError as exc:
        console.print(f"[red]Build failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"Snapshot: {folder}")
    report = json.loads((folder / "build_report.json").read_text(encoding="utf-8"))
    for route, result in report["routes"].items():
        console.print(f"{route}: {result['indexed_count']}/{report['chunk_count']} chunks indexed")
    if not report["complete"]:
        console.print("[yellow]Build is partial; inspect build_report.json and errors.jsonl.[/yellow]")


@app.command()
def query(
    question: str = typer.Argument(..., help="Search question."),
    snapshot_id: str | None = typer.Option(None, help="Exact snapshot ID; defaults to latest complete build."),
    indexes_dir: Path = typer.Option(Path("data/indexes"), help="Snapshot directory."),
    answer: bool = typer.Option(True, "--answer/--retrieval-only", help="Generate a cited DeepSeek answer or show local retrieval only."),
) -> None:
    """Retrieve Top 5 from one snapshot, then answer with citations by default."""
    from src.retrieval.query import latest_complete_snapshot, query as retrieve

    try:
        selected_snapshot = snapshot_id or latest_complete_snapshot(indexes_dir)
        results = retrieve(indexes_dir, selected_snapshot, question,
                           on_step=lambda step: console.print(f"[cyan]{step}...[/cyan]"))
    except (ValueError, FileNotFoundError, ImportError, OSError, APIError) as exc:
        console.print(f"[red]Query failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    snapshot = json.loads((indexes_dir / selected_snapshot / "snapshot.json").read_text(encoding="utf-8"))
    source_files = {source["doc_id"]: source["source_file"] for source in snapshot["sources"]}
    console.print(f"Snapshot: {selected_snapshot}")
    for rank, chunk in enumerate(results, start=1):
        page = (f"{chunk['page_start']}" if chunk["page_start"] == chunk["page_end"]
                else f"{chunk['page_start']}-{chunk['page_end']}") if chunk["page_start"] is not None else "N/A"
        heading = " > ".join(chunk["heading_path"]) or "N/A"
        slide = (f"{chunk['slide_start']}" if chunk.get("slide_start") == chunk.get("slide_end")
                 else f"{chunk['slide_start']}-{chunk['slide_end']}") if chunk.get("slide_start") is not None else "N/A"
        console.print(f"\n#{rank} chunk_id: {chunk['chunk_id']}")
        console.print(f"source: {source_files[chunk['doc_id']]}")
        console.print(f"page: {page} / slide: {slide} / heading: {heading}")
        console.print(f"score: {chunk['score']:.6f}")
        console.print(f"text: {chunk['text']}")
    if answer:
        from src.qa.answer import answer_question

        console.print("\n" + answer_question(indexes_dir, selected_snapshot, question,
                                              Path("config.yaml"), chunks=results,
                                              on_step=lambda step: console.print(f"[cyan]{step}...[/cyan]")))


@app.command()
def ask(
    snapshot_id: str = typer.Argument(..., help="Exact complete snapshot ID."),
    question: str = typer.Argument(..., help="Question to answer."),
    indexes_dir: Path = typer.Option(Path("data/indexes"), help="Snapshot directory."),
    config_path: Path = typer.Option(Path("config.yaml"), help="QA configuration file."),
) -> None:
    """Retrieve from one snapshot, then answer with the DeepSeek API."""
    from src.qa.answer import answer_question

    try:
        result = answer_question(indexes_dir, snapshot_id, question, config_path,
                                 on_step=lambda step: console.print(f"[cyan]{step}...[/cyan]"))
    except (ValueError, FileNotFoundError, RuntimeError, APIError) as exc:
        console.print(f"[red]Ask failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(result)


@app.command()
def evaluate(
    gold_path: Path = typer.Argument(..., help="JSONL file with human-labeled relevant chunks or document pages."),
    snapshot_ids: list[str] = typer.Option([], "--snapshot-id", help="Repeat for each snapshot to compare."),
    k: int = typer.Option(5, min=1, max=5, help="Recall@K cutoff, from 1 to 5."),
    indexes_dir: Path = typer.Option(Path("data/indexes"), help="Snapshot directory."),
    output_dir: Path = typer.Option(Path("data/evaluation"), help="Evaluation report directory."),
    resume_dir: Path | None = typer.Option(None, help="Existing incomplete evaluation folder to resume."),
) -> None:
    """Compute Recall@5 and MRR; compare chunk modes using the same gold labels."""
    from src.evaluation.metrics import evaluate_snapshot, render_report
    from src.retrieval.query import latest_complete_snapshot

    if resume_dir:
        folder = resume_dir
        folder.mkdir(parents=True, exist_ok=True)
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        folder = output_dir / run_id
        folder.mkdir(parents=True, exist_ok=False)
    try:
        selected = snapshot_ids or [latest_complete_snapshot(indexes_dir)]
        results = [evaluate_snapshot(indexes_dir, snapshot_id, gold_path, k=k,
                                     checkpoint_path=folder / f"checkpoint-{snapshot_id}.json")
                   for snapshot_id in selected]
    except (ValueError, FileNotFoundError, APIError) as exc:
        console.print(f"[red]Evaluation failed:[/red] {exc}")
        console.print(f"Checkpoint folder: {folder}")
        raise typer.Exit(code=1) from exc
    (folder / "report.json").write_text(json.dumps({"gold_file": str(gold_path), "snapshots": results},
                                           ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (folder / "report.md").write_text(render_report(results, gold_path), encoding="utf-8")
    console.print(f"Evaluation report: {folder / 'report.md'}")


@app.command(name="evaluate-snapshot-local")
def evaluate_snapshot_local(
    gold_path: Path = typer.Argument(..., help="Human-labeled JSONL."),
    snapshot_id: str = typer.Option(..., help="Exact complete snapshot ID."),
    indexes_dir: Path = typer.Option(..., help="Snapshot directory."),
    output_dir: Path = typer.Option(Path("data/evaluation"), help="Report directory."),
) -> None:
    """Evaluate saved BM25 and local bge-m3 indexes without remote API calls."""
    from src.evaluation.metrics import evaluate_snapshot, render_report

    result = evaluate_snapshot(indexes_dir, snapshot_id, gold_path, k=5,
                               enabled_routes=("bm25", "bge_m3"), hybrid=False)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-local-" + uuid.uuid4().hex[:8]
    folder = output_dir / run_id
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "report.json").write_text(json.dumps({"gold_file": str(gold_path), "snapshots": [result]},
                                                    ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (folder / "report.md").write_text(render_report([result], gold_path), encoding="utf-8")
    console.print(f"Local snapshot report: {folder / 'report.md'}")


app.command(name="test")(evaluate)


@app.command(name="acceptance")
def acceptance(
    snapshot_id: str = typer.Option(..., help="Exact complete snapshot ID."),
    parsed_dir: Path = typer.Option(Path("data/parsed"), help="Parsed artifacts to verify."),
    indexes_dir: Path = typer.Option(Path("data/indexes"), help="Snapshot directory."),
    evaluation_report: Path = typer.Option(..., help="Completed labeled report.json."),
    output_dir: Path = typer.Option(Path("data/acceptance"), help="Acceptance report directory."),
) -> None:
    """Verify saved deliverables without reparsing or calling a model API."""
    from src.evaluation.acceptance import run

    folder = run(Path.cwd(), parsed_dir, indexes_dir, snapshot_id, evaluation_report, output_dir)
    result = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    console.print(f"Acceptance report: {folder / 'report.md'}")
    console.print(f"PASS={result['summary']['pass']} WARN={result['summary']['warn']} "
                  f"FAIL={result['summary']['fail']}")
    if result["summary"]["fail"]:
        raise typer.Exit(code=1)


@app.command(name="analyze-retrieval")
def analyze_retrieval(
    gold_path: Path = typer.Argument(..., help="Human-labeled JSONL."),
    snapshot_id: str = typer.Option(..., help="Exact complete snapshot ID."),
    evaluation_report: Path = typer.Option(..., help="Completed evaluation report.json."),
    indexes_dir: Path = typer.Option(Path("data/indexes"), help="Snapshot directory."),
    output_dir: Path = typer.Option(Path("data/evaluation/analysis"), help="Analysis output directory."),
) -> None:
    """Analyze Top-20/RRF candidate coverage and per-question reranking loss."""
    from src.evaluation.retrieval_analysis import analyze

    try:
        folder = analyze(indexes_dir, snapshot_id, gold_path, evaluation_report, output_dir)
    except (ValueError, FileNotFoundError, APIError) as exc:
        console.print(f"[red]Analysis failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"Retrieval analysis: {folder / 'report.md'}")


@app.command(name="diagnose-reranker")
def diagnose_reranker(
    gold_path: Path = typer.Argument(..., help="Human-labeled JSONL."),
    snapshot_id: str = typer.Option(..., help="Exact complete snapshot ID."),
    evaluation_report: Path = typer.Option(..., help="Completed evaluation report.json."),
    question_numbers: list[int] = typer.Option([], "--question-number", min=1,
                                               help="Repeat to diagnose selected questions; defaults to all."),
    indexes_dir: Path = typer.Option(Path("data/indexes"), help="Snapshot directory."),
    output_dir: Path = typer.Option(Path("data/evaluation/rerank-diagnostics"), help="Diagnostic output."),
) -> None:
    """Record numeric tournament scores without prompts, responses, or document text."""
    from src.evaluation.rerank_diagnostics import diagnose

    try:
        folder = diagnose(indexes_dir, snapshot_id, gold_path, evaluation_report,
                          question_numbers, output_dir)
    except (ValueError, FileNotFoundError, APIError) as exc:
        console.print(f"[red]Reranker diagnosis failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"Reranker diagnostics: {folder / 'rerank_diagnostics.md'}")


@app.command(name="audit-evaluation-answers")
def audit_evaluation_answers(
    gold_path: Path = typer.Argument(..., help="Human-labeled JSONL with reference answers."),
    snapshot_id: str = typer.Option(..., help="Exact complete snapshot ID."),
    evaluation_report: Path = typer.Option(..., help="Completed evaluation report.json."),
    indexes_dir: Path = typer.Option(Path("data/indexes"), help="Snapshot directory."),
    output_dir: Path = typer.Option(Path("data/evaluation/answer-audit"), help="Audit output."),
) -> None:
    """Supplement page metrics with a conservative exact-answer text audit."""
    from src.evaluation.answer_audit import audit

    try:
        folder = audit(indexes_dir, snapshot_id, gold_path, evaluation_report, output_dir)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]Answer audit failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"Answer audit: {folder / 'answer_audit.md'}")


@app.command(name="prepare-gold-review")
def prepare_gold_review(
    gold_path: Path = typer.Argument(..., help="Original benchmark gold JSONL."),
    question_numbers: list[int] = typer.Option(..., "--question-number", min=1,
                                               help="Repeat for each question to review."),
    output_path: Path = typer.Option(Path("data/benchmarks/mmlongbench/gold_review.jsonl"),
                                     help="Review template output; must not be the original gold file."),
) -> None:
    """Create an auditable review template without changing original gold labels."""
    from src.evaluation.gold_review import prepare

    try:
        if output_path.resolve() == gold_path.resolve():
            raise ValueError("Review template must not overwrite the original gold file")
        result = prepare(gold_path, question_numbers, output_path)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]Gold review preparation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"Gold review template: {result}")


@app.command(name="apply-gold-review")
def apply_gold_review(
    gold_path: Path = typer.Argument(..., help="Original benchmark gold JSONL (read only)."),
    review_path: Path = typer.Argument(..., help="Completed review JSONL."),
    evaluation_report: Path = typer.Option(..., help="Completed saved evaluation report.json."),
    snapshot_id: str = typer.Option(..., help="Exact snapshot represented by the evaluation report."),
    indexes_dir: Path = typer.Option(Path("data/indexes"), help="Snapshot directory."),
    output_dir: Path = typer.Option(Path("data/benchmarks/mmlongbench/reviewed_evaluation"),
                                    help="Derived gold and report output directory."),
) -> None:
    """Apply confirmed reviews to a derived gold file and rescore saved Top 5 offline."""
    from src.evaluation.gold_review import apply

    try:
        if output_dir.resolve() == gold_path.resolve().parent:
            raise ValueError("Use a dedicated output directory; original gold must remain untouched")
        folder = apply(gold_path, review_path, evaluation_report, indexes_dir, snapshot_id, output_dir)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]Gold review application failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    report = json.loads((folder / "gold_review_report.json").read_text(encoding="utf-8"))
    console.print(f"Reviewed gold: {folder / 'gold_reviewed.jsonl'}")
    console.print(f"Review report: {folder / 'gold_review_report.md'}")
    console.print(f"Reviewed records: {report['reviewed_records']}; applied changes: {report['applied_changes']}")


@app.command(name="prepare-mmlongbench")
def prepare_mmlongbench(
    source_dir: Path = typer.Option(Path("MMLongBench-Doc"), help="Downloaded benchmark repository."),
    output_dir: Path = typer.Option(Path("data/benchmarks/mmlongbench"), help="Isolated parsed files and labels."),
    max_documents: int = typer.Option(3, min=1, help="Deterministic number of PDFs to import."),
) -> None:
    """Parse a small benchmark subset and convert evidence pages to gold JSONL."""
    from src.evaluation.mmlongbench import prepare

    try:
        report = prepare(source_dir, output_dir, max_documents)
    except (ValueError, FileNotFoundError, OSError) as exc:
        console.print(f"[red]Benchmark import failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"Parsed documents: {report['parsed_documents']}; failed: {len(report['failed_documents'])}")
    console.print(f"Answerable labels: {report['answerable_questions']}; unanswerable: {report['unanswerable_questions']}")
    console.print(f"Gold: {report['gold_path']}")


@app.command(name="evaluate-local")
def evaluate_local(
    parsed_dir: Path = typer.Option(Path("data/benchmarks/mmlongbench/parsed"), help="Existing parsed artifacts."),
    gold_path: Path = typer.Option(Path("data/benchmarks/mmlongbench/gold.jsonl"), help="Human page labels."),
    output_dir: Path = typer.Option(Path("data/evaluation"), help="Offline report output."),
) -> None:
    """Compare fixed and structured chunks with offline BM25 only."""
    from src.evaluation.local_bm25 import run

    try:
        folder = run(parsed_dir, gold_path, output_dir)
    except (ValueError, FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        console.print(f"[red]Local evaluation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    for result in report["results"]:
        console.print(f"{result['chunk_mode']}: chunks={result['chunk_count']}; "
                      f"Recall@5={result['recall_at_k']:.4f}; "
                      f"Recall@20={result['candidate_recall']:.4f}; MRR@5={result['mrr_at_k']:.4f}")
    console.print(f"Offline report: {folder / 'report.md'}")


@app.command(name="describe-mmlongbench")
def describe_mmlongbench(
    source_dir: Path = typer.Option(Path("MMLongBench-Doc"), help="Downloaded benchmark repository."),
    parsed_dir: Path = typer.Option(Path("data/benchmarks/mmlongbench/parsed"), help="Original parsed artifacts."),
    gold_path: Path = typer.Option(Path("data/benchmarks/mmlongbench/gold.jsonl"), help="Human evidence-page labels."),
    output_dir: Path = typer.Option(Path("data/benchmarks/mmlongbench/parsed_vision"), help="New enriched artifacts."),
    max_pages: int | None = typer.Option(None, min=1, help="Optional pilot limit for unique evidence pages."),
) -> None:
    """Describe labeled evidence pages with MoArk Qwen3-VL in a new parsed directory."""
    from src.parser.vision import enrich_mmlongbench

    try:
        report = enrich_mmlongbench(source_dir, parsed_dir, gold_path, output_dir, max_pages=max_pages)
    except (ValueError, FileNotFoundError, OSError, APIError) as exc:
        console.print(f"[red]Vision enrichment failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"Vision pages: {report['described_pages']}/{report['requested_pages']}; "
                  f"failed: {report['failed_pages']}; output: {output_dir}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind address; use a LAN IP or 0.0.0.0 for trusted LAN access."),
    port: int = typer.Option(8000, min=1, max=65535, help="HTTP port."),
) -> None:
    """Serve the shared LAN web interface without login."""
    import uvicorn

    console.print("[yellow]No login: everyone who can reach this address can view and upload documents.[/yellow]")
    uvicorn.run("src.service.web:app", host=host, port=port)


@app.command(name="compare-chunks")
def compare_chunks(
    snapshot_ids: list[str] = typer.Argument(..., help="Two or more build snapshot IDs."),
    indexes_dir: Path = typer.Option(Path("data/indexes"), help="Snapshot directory."),
    output_dir: Path = typer.Option(Path("data/evaluation"), help="Comparison report directory."),
) -> None:
    """Compare chunk counts and lengths across saved snapshots without gold labels."""
    from src.evaluation.metrics import compare_chunk_snapshots, render_chunk_comparison

    try:
        results = compare_chunk_snapshots(indexes_dir, snapshot_ids)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]Comparison failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    folder = output_dir / run_id
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "chunk_comparison.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                                                   encoding="utf-8")
    (folder / "chunk_comparison.md").write_text(render_chunk_comparison(results), encoding="utf-8")
    console.print(f"Chunk comparison: {folder / 'chunk_comparison.md'}")


@app.command(name="all")
def run_all(
    question: str = typer.Argument(..., help="Question to answer after parse and build."),
    raw_dir: Path = typer.Option(Path("data/raw"), help="Input documents."),
    parsed_dir: Path = typer.Option(Path("data/parsed"), help="Parent directory for this run's parsed output."),
    indexes_dir: Path = typer.Option(Path("data/indexes"), help="Snapshot output directory."),
    chunk_mode: str = typer.Option("structured", help="structured or fixed."),
) -> None:
    """Run parse, build, retrieval, and cited answer in order."""
    from src.index.build import build as build_snapshot
    from src.qa.answer import answer_question

    if not raw_dir.is_dir() or not supported_files(raw_dir):
        console.print(f"[red]No supported documents in[/red] {raw_dir}")
        raise typer.Exit(code=1)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run_parsed_dir = parsed_dir / "_runs" / run_id
    parse(raw_dir=raw_dir, output_dir=run_parsed_dir, render_pages=False)
    try:
        folder = build_snapshot(run_parsed_dir, indexes_dir, chunk_mode)
        report = json.loads((folder / "build_report.json").read_text(encoding="utf-8"))
        if not report["complete"]:
            raise ValueError("Build is partial; inspect build_report.json and errors.jsonl")
        console.print(f"Parsed output: {run_parsed_dir}")
        console.print(f"Snapshot: {folder}")
        console.print(answer_question(indexes_dir, folder.name, question, Path("config.yaml"),
                                      on_step=lambda step: console.print(f"[cyan]{step}...[/cyan]")))
    except (ValueError, FileNotFoundError, RuntimeError, APIError) as exc:
        console.print(f"[red]All failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
