from dataclasses import dataclass, field
from pathlib import Path

from helper.logging.do_log import log_i, log_w
from infrastructure.datastore_engine.convert_to_parquest import (
    convert_legacy_file,
    find_legacy_files,
    find_parquet_files,
    repair_parquet_file,
)

"""
Orchestrates infrastructure.datastore_engine.convert_to_parquest's two file-level jobs — legacy
(Feather/ZSTD, CSV-zip) -> Parquet conversion and repair of already-migrated Parquet files that were
written with `date` set as the pandas index instead of a plain column — as one explicit, resumable
batch run over dataset_db, driven by presentation.datastore_engine.convert_to_parquet_cli. One bad file
never aborts the run: every per-file failure is caught, logged, and reported in the returned summary
instead of raised.
"""

_PROGRESS_LOG_EVERY = 200


@dataclass(frozen=True)
class ConversionSummary:
    legacy_files_converted: list[Path] = field(default_factory=list)
    legacy_files_failed: list[tuple[Path, str]] = field(default_factory=list)
    parquet_files_scanned: int = 0
    parquet_files_repaired: list[Path] = field(default_factory=list)
    parquet_files_failed: list[tuple[Path, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.legacy_files_failed and not self.parquet_files_failed


def _convert_legacy_files(root: Path | None, dry_run: bool) -> tuple[list[Path], list[tuple[Path, str]]]:
    converted: list[Path] = []
    failed: list[tuple[Path, str]] = []
    legacy_files = find_legacy_files(root)
    for index, legacy_path in enumerate(legacy_files, start=1):
        if index % _PROGRESS_LOG_EVERY == 0:
            log_i(f"convert_to_parquest: converted {index}/{len(legacy_files)} legacy file(s) so far")
        if dry_run:
            converted.append(legacy_path)
            continue
        try:
            parquet_path = convert_legacy_file(legacy_path)
        except Exception as e:
            log_w(f"convert_to_parquest: failed converting {legacy_path}: {e}")
            failed.append((legacy_path, str(e)))
            continue
        if parquet_path is None:
            failed.append((legacy_path, "conversion did not produce a Parquet file"))
        else:
            converted.append(parquet_path)
    return converted, failed


def _repair_parquet_files(root: Path | None, dry_run: bool) -> tuple[int, list[Path], list[tuple[Path, str]]]:
    repaired: list[Path] = []
    failed: list[tuple[Path, str]] = []
    parquet_files = find_parquet_files(root)
    for index, parquet_path in enumerate(parquet_files, start=1):
        if index % _PROGRESS_LOG_EVERY == 0:
            log_i(f"convert_to_parquest: scanned {index}/{len(parquet_files)} Parquet file(s) so far")
        try:
            if repair_parquet_file(parquet_path, dry_run=dry_run):
                repaired.append(parquet_path)
        except Exception as e:
            log_w(f"convert_to_parquest: failed repairing {parquet_path}: {e}")
            failed.append((parquet_path, str(e)))
    return len(parquet_files), repaired, failed


def run_datastore_conversion(root: Path | None = None, dry_run: bool = False) -> ConversionSummary:
    """
    Full datastore_engine conversion/repair pass over dataset_db_root() (or `root`, for tests/a scoped
    rerun): convert every remaining legacy Feather/ZSTD or CSV-zip cache file to Parquet/ZSTD, then scan
    every Parquet file and repair any still carrying the date-as-index bug (see
    infrastructure.datastore_engine.convert_to_parquest module docstring). `dry_run=True` reports what
    would change without writing anything.
    """
    converted, legacy_failed = _convert_legacy_files(root, dry_run)
    scanned, repaired, parquet_failed = _repair_parquet_files(root, dry_run)
    verb = "would convert" if dry_run else "converted"
    repair_verb = "would repair" if dry_run else "repaired"
    log_i(
        f"convert_to_parquest: {verb} {len(converted)} legacy file(s) "
        f"({len(legacy_failed)} failed); scanned {scanned} Parquet file(s), {repair_verb} {len(repaired)} "
        f"({len(parquet_failed)} failed)"
    )
    return ConversionSummary(
        legacy_files_converted=converted,
        legacy_files_failed=legacy_failed,
        parquet_files_scanned=scanned,
        parquet_files_repaired=repaired,
        parquet_files_failed=parquet_failed,
    )
