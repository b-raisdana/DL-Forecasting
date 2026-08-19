## Objective

A robust, efficient, and maintainable process for converting, organizing, and maintaining CSV/ZIP and Feather datasets as Parquet.

### 1. Conversion

* Convert all CSV/ZIP and Feather files to Parquet.
* Preserve data, types, timestamps, nulls, and relevant metadata.
* Log conversion failures and unsupported files.

### 2. Dataset Organization

* Group files by dataframe/dataset type into dedicated folders.
* Define consistent naming, directory, and partition conventions.
* Partition by time or other high-value query dimensions where appropriate.

### 3. File Size & Layout

* Target ~200 MB Parquet files; define acceptable min/max range.
* Keep records time-sequential where useful.
* Define appropriate row-group size.
* Consider sorting/clustering by commonly filtered columns.

### 4. Incremental & Late Data

Plan handling of:

* New data extending the latest range.
* Data filling historical gaps.
* Out-of-order data.
* Overlapping data.
* Corrections/deletions.

Compare append, new-file, rewrite, merge/re-split, and deferred-compaction approaches. Determine whether file size should influence the choice.

### 5. Deduplication

* Define duplicate keys/rules.
* Deduplicate across files and ingestion batches.
* Make repeated ingestion idempotent.
* Define when deduplication occurs: ingestion, compaction, or both.

### 6. Schema Evolution / migration mechanism

Handle added/removed/renamed columns, type changes, missing columns, incompatible schemas, and malformed values. Define when schemas can be merged versus requiring a new version.

### 7. Compaction & Reorganization

* Merge small files and split oversized files.
* Re-sort/repartition when beneficial.
* Define compaction triggers.
* Make compaction atomic and recoverable.

### 8. Query & Index Optimization

* Optimize partitioning for common filters.
* Use predicate/projection pushdown and Parquet statistics.
* Optimize row groups and min/max data skipping.
* Evaluate explicit/external indexes where beneficial.
* Benchmark different file/row-group sizes and layouts.

### 9. Housekeeping & Cleanup

* Remove temporary, orphaned, obsolete, superseded, and duplicate files safely.
* Define retention and cleanup policies.
* Detect and remediate small-file problems.
* Keep an audit trail for destructive operations.

### 10. Validation & Data Quality

* Validate schemas, row counts, types, nulls, timestamps, ordering, and duplicates.
* Compare source/Parquet aggregates or checksums where appropriate.
* Validate after conversion, merge, and compaction.

### 11. Metadata, Catalog & Lineage

Maintain a dataset/file manifest containing:

* File path, size, row count, time range.
* Partition/schema/version.
* Relevant column statistics.
* Source files, processing time, checksum, and status.
* Transformation and lineage information.

### 12. Lifecycle & Retention

no need at the moment / postponed

* Define hot/warm/archive storage where useful.
* Define retention and deletion policies.
* Handle immutable versus mutable historical data.
* Define archival and historical-version policies.

### 13. Reliability & Concurrency

* Use atomic writes and temporary staging files.
* Prevent concurrent conflicting updates.
* Ensure readers never see partial files.
* Make ingestion/compaction idempotent.
* Define failure detection, retry, recovery, and rollback.

### 14. Monitoring

Track:

* File count/size and small-file ratio.
* Dataset coverage/gaps.
* Duplicates and schema changes.
* Conversion/compaction failures.
* Processing latency and storage growth.
* Query/read performance.
* log all we did on the original

Define thresholds and alerts for abnormal conditions.

### 15. Security

no need at the moment / postponed

* Separate raw, staging, and processed data.
* Define read/write/delete permissions.
* Handle encryption and sensitive data appropriately.
* Prevent unauthorized modification/deletion.

### 16. Operational Procedures

in the service-application root folder readme.md Define procedures for:

* New dataset onboarding.
* Backfills and reprocessing.
* Late data.
* Schema migration.
* Deduplication.
* Compaction/repartitioning.
* Corrupt files.
* Rollback/recovery.
* Dataset archival/deletion.

### 17. Final Architecture

Define the recommended end-to-end flow:

**source → staging → conversion → validation → schema normalization → deduplication → sorting/partitioning → Parquet sizing → catalog update → incremental ingestion → compaction → validation → housekeeping**

For each major decision, document the **default approach, alternatives, triggers, trade-offs, and recovery strategy**.
