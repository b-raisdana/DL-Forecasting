from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd
from infrastructure.datastore_engine.disk_cache_layout import add_timeframe_index, index_by_date

"""
DuckDB-backed batched read for infrastructure.disk_cache's windowed cache (read_file_windowed()).
Given the on-disk Parquet files for a set of calendar windows that are already fully cached — the hot
path for repeated backtesting/training reads over historical data — this replaces the previous
one-pd.read_parquet-call-per-window loop with a single query, getting real row-group pruning on the
`date` column every write_data_file() call already writes (df.reset_index().to_parquet(...)).

Infrastructure-layer I/O adapter only, no calculation logic (project-decisions skill § code layers) —
mirrors disk_cache.py/disk_cache_layout.py's own split. Not a storage-format change: Parquet/ZSTD
stays the authoritative on-disk format; this module only reads what's already there. See
data/dataset_db/README.md and docs/infrastructure.md § DuckDB for the full design.
"""


def read_parquet_files(paths: list[Path], data_frame_type: str, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Read and concatenate `paths` (already-validated Parquet/ZSTD cache files, all the same
    data_frame_type) in one DuckDB query, trimmed to [start, end] via a real filter on the `date`
    column, then indexed exactly like disk_cache.read_by_date()/read_with_timeframe() would (so the
    result is indistinguishable from disk_cache's own per-file read path).

    `paths` is always an explicit file list, never a directory glob — read_file_windowed() only ever
    passes the exact canonical file for each already-cached window, so this can't accidentally pick up
    an unrelated legacy-range file sitting in the same data_frame_type directory.
    """
    if not paths:
        raise ValueError("read_parquet_files() requires at least one path")
    file_list = [str(p) for p in paths]
    con = duckdb.connect()
    try:
        df = con.execute(
            "SELECT * FROM read_parquet($files) WHERE date >= $start AND date <= $end",
            {"files": file_list, "start": start, "end": end},
        ).fetch_df()
    finally:
        con.close()
    df = index_by_date(df)
    return add_timeframe_index(df, data_frame_type)
