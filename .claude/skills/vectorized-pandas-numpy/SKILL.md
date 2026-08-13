---
name: vectorized-pandas-numpy
description: Use whenever writing or reviewing pandas/numpy code in this repo — dataset generation, indicator/label computation, scaling/normalization, OHLCV processing. Pushes toward fully vectorized (no Python-level per-row/per-sample loops) implementations for correctness at scale and speed. Trigger before writing new DataFrame/ndarray logic and again as a review pass on existing loops over rows, dates, or samples.
---

# Vectorized pandas/numpy

Goal: express row/sample-wise logic as array/DataFrame operations, not Python loops. A loop over rows or samples is a bug magnet and 10-1000x slower than the vectorized equivalent.

## Red flags to eliminate

- `for i in range(len(df))`, `for idx, row in df.iterrows()`, `df.apply(..., axis=1)` — almost always replaceable by column-wise ops, `np.where`, `np.select`, or boolean masks.
- `while remained > 0: ... .append(...)` sample-generation loops that recompute slices one at a time (see `train_data_of_mt_n_profit` in [training_datasets.py](app/ai_modelling/dataset_generator/training_datasets.py) for an existing instance) — prefer drawing all sample boundaries at once (vectorized `np.random.randint(size=n)`), building slices via `.loc` on a batch of index labels, or using numpy fancy indexing/stride tricks on the underlying array instead of repeating a single-sample code path in a loop.
- Repeated `np.array(df[cols])` conversions inside a hot loop — convert once, slice the ndarray afterward.
- Growing a Python list of DataFrames/arrays across a loop then `pd.concat`/`np.array` at the end: only when each iteration's boundaries are inherently sequential/random-dependent and can't be precomputed; if the iteration count is fixed and independent per-sample, generate all boundaries up front and slice in one vectorized pass.
- Scalar-by-scalar column assignment in a loop over columns (`for column in slc.columns: if column in X: ...`) — replace with `.isin()` masks and vectorized arithmetic across the selected columns at once, e.g. `df.loc[:, price_cols] = (df[price_cols] + shift) * scale`.

## Preferred patterns

- Boolean masking: `df.loc[mask, col] = value` instead of iterating and checking a condition per row.
- `np.where(cond, a, b)` / `np.select([cond1, cond2], [a, b], default=c)` for branching per-element logic.
- Broadcasting: align shapes so an operation applies across the whole array/DataFrame at once instead of per-row/per-column loops.
- Group-wise vectorized ops: `.groupby(...).transform(...)` or `.rolling(...).agg(...)` instead of manual index slicing per window.
- For sample/window extraction at many offsets: build all offsets as an array, use `pd.IndexSlice`/`.loc` with a batch of boundaries, or `numpy.lib.stride_tricks.sliding_window_view` for fixed-length windows over a contiguous array — avoid re-deriving one window at a time inside a `while`/`for`.
- Prefer `.to_numpy()` (not bare `.values`) when dropping to ndarray for a hot path.

## When a loop is legitimate

- Truly sequential/stateful logic (e.g., a walk-forward state machine where step `n` depends on the *computed result* of step `n-1`, not just on fixed input data) — vectorize as much of the per-step body as possible, but the outer step-to-step loop may stay.
- One-time setup/config code, not hot paths over data.

## Review checklist

1. Any `for`/`while` touching DataFrame rows, ndarray elements, or sample indices — can it be replaced by a mask, `np.where`/`np.select`, groupby/rolling, or a fully-vectorized index computation?
2. Any repeated `np.array(...)`/`.to_numpy()` conversion of the same columns inside a loop — hoist it outside.
3. Any column-by-column Python loop applying the same formula — collapse to one vectorized expression over the selected columns.
4. Confirm the vectorized version's shape/dtype matches what downstream code expects (dtype upcasting from mixed NaN/int, index alignment after `.loc` reindexing) before replacing the loop.
