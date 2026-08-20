"""CLI entrypoint for dataset_db's 3 Parquet housekeeping jobs — see
infrastructure.datastore_engine.parquet_housekeeping's module docstring for what each one does and why
they're separate commands.

Examples:
    python -m presentation.datastore_engine.parquet_housekeeping_cli migrate
    python -m presentation.datastore_engine.parquet_housekeeping_cli migrate --dry-run
    python -m presentation.datastore_engine.parquet_housekeeping_cli fix-index
    python -m presentation.datastore_engine.parquet_housekeeping_cli fix-index --force
    python -m presentation.datastore_engine.parquet_housekeeping_cli compact --dry-run
"""

from __future__ import annotations

from pathlib import Path

import typer
from application.datastore_engine.parquet_housekeeping import run_compaction, run_index_repair, run_legacy_migration

app = typer.Typer(add_completion=False)


def _print_failures(header: str, failures: list[tuple[Path, str]]) -> None:
    if not failures:
        return
    print(f"[parquet_housekeeping_cli] {len(failures)} {header}:")
    for path, error in failures:
        print(f"  {path}: {error}")


def _print_batch_failures(header: str, failures: list[tuple[list[Path], str]]) -> None:
    if not failures:
        return
    print(f"[parquet_housekeeping_cli] {len(failures)} {header}:")
    for files, error in failures:
        print(f"  {files[0]}..{files[-1]} ({len(files)} file(s)): {error}")


@app.command()
def migrate(
    dry_run: bool = typer.Option(False, "--dry-run", help="report what would change without writing anything"),
) -> None:
    """Convert every remaining legacy Feather/ZSTD or CSV-zip cache file to Parquet/ZSTD."""
    summary = run_legacy_migration(dry_run=dry_run)
    verb = "would convert" if dry_run else "converted"
    print(f"[parquet_housekeeping_cli] {verb} {len(summary.legacy_files_converted)} legacy file(s) to Parquet")
    _print_failures("legacy conversion failure(s)", summary.legacy_files_failed)
    if not summary.ok:
        raise typer.Exit(code=1)


@app.command("fix-index")
def fix_index(
    dry_run: bool = typer.Option(False, "--dry-run", help="report what would change without writing anything"),
    force: bool = typer.Option(
        False, "--force", help="scan every Parquet file even if no repair was flagged by a real read failure"
    ),
) -> None:
    """Scan and repair Parquet files still carrying the date-as-index bug — skipped by default unless a
    real read already flagged one live (see --force)."""
    summary = run_index_repair(dry_run=dry_run, force=force)
    if not summary.ran:
        print("[parquet_housekeeping_cli] no repair flagged, nothing to do (pass --force to scan anyway)")
        return
    repair_verb = "would repair" if dry_run else "repaired"
    print(
        f"[parquet_housekeeping_cli] scanned {summary.parquet_files_scanned} Parquet file(s), "
        f"{repair_verb} {len(summary.parquet_files_repaired)}"
    )
    _print_failures("Parquet repair failure(s)", summary.parquet_files_failed)
    if not summary.ok:
        raise typer.Exit(code=1)


@app.command()
def compact(
    dry_run: bool = typer.Option(False, "--dry-run", help="report what would change without writing anything"),
) -> None:
    """Merge contiguous small per-window Parquet files into larger ~app_config.parquet_target_chunk_size_mb
    files."""
    summary = run_compaction(dry_run=dry_run)
    verb = "would merge" if dry_run else "merged"
    print(f"[parquet_housekeeping_cli] {verb} {len(summary.batches_merged)} batch(es)")
    _print_batch_failures("compaction failure(s)", summary.batches_failed)
    if not summary.ok:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
