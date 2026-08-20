# DataLoader-to-GPU feed path TODO

Open work for closing visibility and starvation gaps in the training data pipeline's producer (DataLoader/prefetcher) to consumer (GPU step) feed path. Assumption: six-timeframe fusion sample construction, 8GB VRAM ceiling, and per-sample cost make queue underruns and unbounded growth both realistic failure modes.

## GPU starvation risk

- Verify the DataLoader/prefetcher keeps enough batches pre-loaded so the GPU never idles between consecutive optimizer steps.
- Instrument end-to-end: DataLoader worker → collate/fetch → CUDA transfer → step start. The gap between step N's `loss.backward()` finishing and step N+1's forward pass starting must never exceed a threshold driven by actual step latency, not a guessed `prefetch_factor`.
- Tuning knob candidates: `prefetch_factor`, `num_workers`, `persistent_workers`, `pin_memory`, and whether a custom `IterDataPipe`/prefetch buffer (bounded, separate from PyTorch's internal queue) is needed because the default queue depth is opaque.

## Consumption-rate matching

- Monitor sample consumption rate (GPU steps/sec) vs. production rate (samples/sec produced by the loader) in real time.
- Ensure the feed keeps pace with consumption without unbounded queue growth. A producer that outruns the consumer wastes RAM that could otherwise hold larger batches or models under the 8GB VRAM cap.

## Metrics to instrument

- **Queue depth** — # pre-loaded batches (and samples) waiting at any instant. Should trend around a target steady-state; not zero (starvation) and not climbing without bound (backup).
- **Time-to-feed per batch** — producer-side latency: worker fetch + collate + optional transform + host-to-device copy. Baseline this once and flag regressions.
- **Time-to-consume per batch** — GPU step time: forward + backward + optimizer step. Should be stable relative to model architecture; spikes indicate host-side stalls.
- **GPU utilization % and idle/stall time** — nvidia-smi / DCGM / PyTorch profiler `Record#0` gaps. Target: near 100% utilization during training windows; any gap >2x median time-to-consume is suspicious.
- **DataLoader worker utilization / CPU-bound bottlenecks** — per-worker CPU time, blocked time, and context-switch rate. Six-timeframe fusion samples may spend meaningful CPU in alignment/resampling before the tensor ever reaches the GPU.
- **I/O wait time** — disk/network latency for multi-timeframe OHLCV window reads. If the cache is warm this should be near zero; any non-trivial iowait here points to a cache-miss or an unbounded window pull.

## Alerting / thresholds

- **Starved** — GPU idle waiting on data. Definition: consecutive step-start delays exceeding `k × median(time-to-consume)` where `k ≥ 2` (configurable). Trigger: log warning, increment starvation counter, optionally break-into next epoch early to refresh loader state. VRAM context: starvation is especially costly here because each wasted idle second is a second the 8GB card could have been crunching the already-allocated batch.
- **Backed up** — producer outrunning consumer, risking memory pressure. Definition: queue depth sustained above `prefetch_factor × num_workers` for more than `N` consecutive measurement windows, OR host memory RSS growth correlated with queue growth. Trigger: log warning, reduce `prefetch_factor`, or apply backpressure via a bounded semaphore in a custom fetch loop. VRAM context: queue growth doesn't hit VRAM directly, but host memory bloat from pinned/mmap'd OHLCV windows can push the process into swap, which then stalls the PCIe transfer thread and indirectly starves the GPU.
- **CPU bottleneck** — DataLoader worker CPU utilization consistently >90% across all workers. Trigger: reduce per-worker transform cost (vectorize/resample upstream, cache aligned windows to disk), or increase `num_workers` only if I/O is the actual bottleneck.
- **I/O stall** — single-fetch I/O wait time > threshold relative to cache-hit baseline. Trigger: cache warm-up failure, network filesystem hiccup, or an unexpectedly large window read (verify DuckDB Arrow slicing is active and not falling back to full-table reads).

## Implementation notes

- Prefer a lightweight, opt-in instrumentation module (e.g. `app/infrastructure/dataloader_profiler.py`) that wraps the existing DataLoader or monkey-patches `torch.utils.data.DataLoader` hooks, rather than modifying every call site. Output: structured logs + optional TensorBoard/Weights & Biases step-level scalars.
- Start with a characterization run on one model (tier1_000) to collect baseline distributions for all six metrics before tuning `prefetch_factor` or adding custom prefetch logic. Baselines are needed to set the `k` multiplier and queue-depth ceilings above.
