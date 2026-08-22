---
name: sqlalchemy-duckdb-orm
description: Use when adding a genuinely relational table (metadata, config, job/coverage tracking) backed by DuckDB — generate SQLAlchemy Core Tables/ORM classes from the existing Pandera DataFrameModel instead of hand-writing a second column definition, and wire Alembic against the generated MetaData.
---

# SQLAlchemy + DuckDB, ORM style

Trigger: a table that's genuinely relational (has real structure/relationships/queries-by-key), not a bulk time-series artifact. Example use case: a coverage ledger tracking which `(datastore_relative_path, timeframe, window)` tuples `duckdb_cache` has already generated (see [docs/todos/duckdb_cache_decorator.md](../../../docs/todos/duckdb_cache_decorator.md) § SQLAlchemy + DuckDB ORM) — optional, not required for that decorator's v1.

**Do not use this for bulk OHLCV/indicator/label rows.** Row-by-row ORM object mapping is a serious throughput regression against columnar Parquet/DuckDB bulk writes (project-decisions § vectorized-pandas-numpy, lib-first) — time-series data stays in Parquet, queried in bulk via `duckdb.connect().execute(...)`/`read_parquet(...)`, never inserted one `session.add(...)` at a time.

## Dependencies

`sqlalchemy[asyncio]==2.0.52` is already in `requirements.txt`. Add `alembic` and `duckdb-engine` (the SQLAlchemy dialect for DuckDB, connection string `duckdb:///path/to/file.duckdb`) the first time this pattern is actually used — not preemptively.

## Pandera is the source of truth for columns

Don't hand-write a second column/type definition in a declarative SQLAlchemy class. Generate a SQLAlchemy Core `Table` from the same `pandera.DataFrameModel` already used to validate the DataFrame version of this data, then imperatively map an ORM class onto that generated `Table`:

```python
def pandera_to_table(
    schema_cls: type[pa.DataFrameModel], table_name: str, metadata: MetaData,
    primary_key: str, foreign_keys: dict[str, str] | None = None,
) -> Table:
    schema = schema_cls.to_schema()
    columns = [
        Column(name, PANDAS_TO_SQLA.get(np.dtype(col.dtype.type), String),
               ForeignKey(foreign_keys[name]) if name in (foreign_keys or {}) else None,
               primary_key=(name == primary_key), nullable=col.nullable)
        for name, col in schema.columns.items()
    ]
    return Table(table_name, metadata, *columns)
```

Relationships/foreign keys are the one piece Pandera has no concept of — pass them explicitly to the generator (`foreign_keys={"customer_id": "customers.id"}`), then wire `relationship(...)` in `mapper_registry.map_imperatively(...)`. Everything else (column names, types, nullability, primary key) comes from the Pandera schema, once.

## Alembic

Point `alembic/env.py`'s `target_metadata` at the same `MetaData` object the generator populated — it works exactly like a hand-written declarative `Base.metadata`. Caveat: Alembic diffs whatever `metadata` currently holds, not the Pandera class definitions directly — if a Pandera schema changes, re-run the `pandera_to_table` generator (rebuild `metadata`) *before* `alembic revision --autogenerate`, or the migration diff will be against stale table shapes.

## Full worked example

`docs/todos/upgrade_duckdb_cache_to_use_sqlalchemy_and_alembic.md` has the complete version: schema definition, generator, imperative mapping with relationships, engine setup (`create_engine("duckdb:///...", poolclass=NullPool)`), CRUD, joins/aggregation via `select(...)`, a Pandera-validate-then-`session.merge()` row loop for small batches, and a SQL-level `INSERT ... ON CONFLICT DO UPDATE` for larger upserts. Read it before implementing a new table with this pattern.
