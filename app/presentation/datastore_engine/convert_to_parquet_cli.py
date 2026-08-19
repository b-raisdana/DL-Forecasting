"""CLI entrypoint: convert every remaining legacy Feather/ZSTD or CSV-zip cache file under
<data>/dataset_db/ to Parquet/ZSTD, then scan every Parquet file there and repair any still written
with `date` (and, for multi_timeframe_* types, `timeframe`) set as the pandas index instead of a plain
column — see application.datastore_engine.convert_to_parquest.run_datastore_conversion and
infrastructure.datastore_engine.convert_to_parquest's module docstring for the bug this repairs.

Examples:
    python -m presentation.datastore_engine.convert_to_parquet_cli
    python -m presentation.datastore_engine.convert_to_parquet_cli --dry-run
"""

from __future__ import annotations

from pathlib import Path

import typer
from application.datastore_engine.convert_to_parquest import run_datastore_conversion

app = typer.Typer(add_completion=False)


def _print_failures(header: str, failures: list[tuple[Path, str]]) -> None:
    if not failures:
        return
    print(f"[convert_to_parquet_cli] {len(failures)} {header}:")
    for path, error in failures:
        print(f"  {path}: {error}")


@app.command()
def main(
    dry_run: bool = typer.Option(False, "--dry-run", help="report what would change without writing anything"),
) -> None:
    summary = run_datastore_conversion(dry_run=dry_run)
    verb = "would convert" if dry_run else "converted"
    repair_verb = "would repair" if dry_run else "repaired"
    print(f"[convert_to_parquet_cli] {verb} {len(summary.legacy_files_converted)} legacy file(s) to Parquet")
    print(
        f"[convert_to_parquet_cli] scanned {summary.parquet_files_scanned} Parquet file(s), "
        f"{repair_verb} {len(summary.parquet_files_repaired)}"
    )
    _print_failures("legacy conversion failure(s)", summary.legacy_files_failed)
    _print_failures("Parquet repair failure(s)", summary.parquet_files_failed)
    if not summary.ok:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
