from dataclasses import dataclass, field
from pathlib import Path

from helper.logging.do_log import log_i, log_w
from infrastructure.datastore_engine.parquet_housekeeping import (
    clear_repair_flag,
    convert_legacy_file,
    find_legacy_files,
    find_merge_batches,
    find_parquet_files,
    merge_batch,
    repair_parquet_file,
    repair_required,
)

"""
Orchestrates infrastructure.datastore_engine.parquet_housekeeping's 3 file-level jobs as 3 separate,
explicit, resumable batch runs over dataset_db, driven by
presentation.datastore_engine.parquet_housekeeping_cli's 3 subcommands (`migrate`, `fix-index`,
`compact`). One bad file/batch never aborts a run: every failure is caught, logged, and reported in the
returned summary instead of raised.
"""

_PROGRESS_LOG_EVERY = 200


@dataclass(frozen=True)
class MigrationSummary:
    legacy_files_converted: list[Path] = field(default_factory=list)
    legacy_files_failed: list[tuple[Path, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.legacy_files_failed


@dataclass(frozen=True)
class RepairSummary:
    ran: bool  # False when skipped because no repair was flagged/forced (see run_index_repair())
    parquet_files_scanned: int = 0
    parquet_files_repaired: list[Path] = field(default_factory=list)
    parquet_files_failed: list[tuple[Path, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.parquet_files_failed


@dataclass(frozen=True)
class CompactionSummary:
    batches_merged: list[Path] = field(default_factory=list)
    batches_failed: list[tuple[list[Path], str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.batches_failed


def run_legacy_migration(root: Path | None = None, dry_run: bool = False) -> MigrationSummary:
    """Convert every remaining legacy Feather/ZSTD or CSV-zip cache file under dataset_db_root() (or
    `root`, for tests/a scoped rerun) to Parquet/ZSTD (see infrastructure.datastore_engine.
    parquet_housekeeping module docstring § 1). `dry_run=True` reports what would change without
    writing anything."""
    converted: list[Path] = []
    failed: list[tuple[Path, str]] = []
    legacy_files = find_legacy_files(root)
    for index, legacy_path in enumerate(legacy_files, start=1):
        if index % _PROGRESS_LOG_EVERY == 0:
            log_i(f"parquet_housekeeping: converted {index}/{len(legacy_files)} legacy file(s) so far")
        if dry_run:
            converted.append(legacy_path)
            continue
        try:
            parquet_path = convert_legacy_file(legacy_path)
        except Exception as e:
            log_w(f"parquet_housekeeping: failed converting {legacy_path}: {e}")
            failed.append((legacy_path, str(e)))
            continue
        if parquet_path is None:
            failed.append((legacy_path, "conversion did not produce a Parquet file"))
        else:
            converted.append(parquet_path)
    verb = "would convert" if dry_run else "converted"
    log_i(f"parquet_housekeeping: {verb} {len(converted)} legacy file(s) ({len(failed)} failed)")
    return MigrationSummary(legacy_files_converted=converted, legacy_files_failed=failed)


def run_index_repair(root: Path | None = None, dry_run: bool = False, force: bool = False) -> RepairSummary:
    """Scan every Parquet file under dataset_db_root() (or `root`) and repair any still carrying the
    date-as-index bug (see infrastructure.datastore_engine.parquet_housekeeping module docstring § 2).
    A full scan is expensive at this repo's scale, so it only runs when repair_required() says a repair
    was actually flagged by a real read failure, or when `force=True` overrides that gate — otherwise
    this is a no-op reporting `ran=False`. `dry_run=True` reports what would change without writing
    anything and never clears the flag (so a real run still picks it up next time)."""
    if not force and not repair_required(root):
        log_i("parquet_housekeeping: no repair flagged, skipping scan (pass force=True to override)")
        return RepairSummary(ran=False)

    repaired: list[Path] = []
    failed: list[tuple[Path, str]] = []
    parquet_files = find_parquet_files(root)
    for index, parquet_path in enumerate(parquet_files, start=1):
        if index % _PROGRESS_LOG_EVERY == 0:
            log_i(f"parquet_housekeeping: scanned {index}/{len(parquet_files)} Parquet file(s) so far")
        try:
            if repair_parquet_file(parquet_path, dry_run=dry_run):
                repaired.append(parquet_path)
        except Exception as e:
            log_w(f"parquet_housekeeping: failed repairing {parquet_path}: {e}")
            failed.append((parquet_path, str(e)))
    repair_verb = "would repair" if dry_run else "repaired"
    log_i(
        f"parquet_housekeeping: scanned {len(parquet_files)} Parquet file(s), {repair_verb} "
        f"{len(repaired)} ({len(failed)} failed)"
    )
    if not dry_run and not failed:
        clear_repair_flag(root)
    return RepairSummary(
        ran=True, parquet_files_scanned=len(parquet_files), parquet_files_repaired=repaired, parquet_files_failed=failed
    )


def run_compaction(root: Path | None = None, dry_run: bool = False) -> CompactionSummary:
    """Merge every contiguous batch of small, single-calendar-window Parquet files under
    dataset_db_root() (or `root`) into larger ~app_config.parquet_target_chunk_size_mb files (see
    infrastructure.datastore_engine.parquet_housekeeping module docstring § 3). `dry_run=True` reports
    which batches would be merged without writing anything."""
    batches = find_merge_batches(root)
    merged: list[Path] = []
    failed: list[tuple[list[Path], str]] = []
    for data_frame_type, files in batches:
        if dry_run:
            merged.append(files[0].with_name(f"{data_frame_type}.<merged>.parquet"))
            continue
        try:
            merged_path = merge_batch(data_frame_type, files)
        except Exception as e:
            log_w(f"parquet_housekeeping: failed merging batch of {len(files)} file(s): {e}")
            failed.append((files, str(e)))
            continue
        if merged_path is None:
            failed.append((files, "merge did not produce a Parquet file"))
        else:
            merged.append(merged_path)
    verb = "would merge" if dry_run else "merged"
    log_i(f"parquet_housekeeping: {verb} {len(merged)} batch(es) ({len(failed)} failed)")
    return CompactionSummary(batches_merged=merged, batches_failed=failed)
