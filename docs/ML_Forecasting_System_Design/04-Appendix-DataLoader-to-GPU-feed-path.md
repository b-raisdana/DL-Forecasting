# DataLoader-to-GPU feed path

Assumption: six-timeframe fusion sample construction, 8GB VRAM ceiling, and per-sample cost make queue underruns and unbounded growth both realistic failure modes. This doc closes visibility and starvation gaps in the training data pipeline's producer (DataLoader/prefetcher) to consumer (GPU step) feed path.

## Problem statement

The default PyTorch DataLoader queue is opaque. Under the project's constraints — per-sample six-timeframe fusion, 8GB VRAM, host-memory pressure from pinned/mmap'd OHLCV windows — two failure modes are realistic:

- **GPU starvation**: the GPU idles between optimizer steps because the prefetcher hasn't queued the next batch in time.
- **Backup / memory pressure**: the producer outruns the consumer, queue depth grows without bound, and host memory bloat pushes the process into swap, stalling the PCIe transfer thread and indirectly starving the GPU.

Neither mode is visible today. This doc defines what to instrument, what counts as "starved" vs. "backed up," and how to close the gap.

## GPU starvation risk

The GPU must never idle between consecutive optimizer steps. The relevant gap is the wall-clock time between step N's `loss.backward()` finishing and step N+1's forward pass starting.

Current tuning knobs: `prefetch_factor`, `num_workers`, `persistent_workers`, `pin_memory`. If the default queue depth proves insufficient or opaque, a bounded custom `IterDataPipe` prefetch buffer separate from PyTorch's internal queue is the fallback.

**Decision**: instrument first, tune second. Profile baseline queue depth and step-start latency on one model before changing `prefetch_factor` or adding custom prefetch logic.

## Consumption-rate matching

Monitor sample consumption rate (GPU steps/sec) vs. production rate (samples/sec produced by the loader) in real time. The feed must keep pace with consumption without unbounded queue growth — host RAM consumed by a runaway queue is RAM unavailable for larger batches or a larger model under the 8GB VRAM cap.

## Metrics to instrument

- **Queue depth** — # pre-loaded batches (and samples) waiting at any instant. Target: steady-state above zero, trending flat — not climbing without bound.
- **Time-to-feed per batch** — producer-side latency: worker fetch + collate + optional transform + host-to-device copy. Baseline once; flag regressions.
- **Time-to-consume per batch** — GPU step time: forward + backward + optimizer step. Stable relative to model architecture; spikes indicate host-side stalls.
- **GPU utilization % and idle/stall time** — nvidia-smi / DCGM / PyTorch profiler `Record#0` gaps. Target: near 100% during training windows; any gap >2x median time-to-consume is suspicious.
- **DataLoader worker utilization / CPU-bound bottlenecks** — per-worker CPU time, blocked time, context-switch rate. Six-timeframe fusion samples may spend meaningful CPU in alignment/resampling before the tensor reaches the GPU.
- **I/O wait time** — disk/network latency for multi-timeframe OHLCV window reads. Near zero with a warm cache; non-trivial iowait points to a cache miss or an unbounded window pull.

## Alerting / thresholds

- **Starved** — GPU idle waiting on data. Definition: consecutive step-start delays exceeding `k × median(time-to-consume)` where `k ≥ 2` (configurable). Trigger: log warning, increment starvation counter, optionally break into the next epoch early to refresh loader state. VRAM context: each wasted idle second is a second the 8GB card could have been crunching the already-allocated batch.
- **Backed up** — producer outrunning consumer, risking memory pressure. Definition: queue depth sustained above `prefetch_factor × num_workers` for more than `N` consecutive measurement windows, OR host memory RSS growth correlated with queue growth. Trigger: log warning, reduce `prefetch_factor`, or apply backpressure via a bounded semaphore in a custom fetch loop. VRAM context: queue growth doesn't hit VRAM directly, but host memory bloat from pinned/mmap'd OHLCV windows can push the process into swap, which then stalls the PCIe transfer thread and indirectly starves the GPU.
- **CPU bottleneck** — DataLoader worker CPU utilization consistently >90% across all workers. Trigger: reduce per-worker transform cost (vectorize/resample upstream, cache aligned windows to disk), or increase `num_workers` only if I/O is the actual bottleneck.
- **I/O stall** — single-fetch I/O wait time > threshold relative to cache-hit baseline. Trigger: cache warm-up failure, network filesystem hiccup, or an unexpectedly large window read (verify DuckDB Arrow slicing is active and not falling back to full-table reads).

## Implementation approach

Prefer a lightweight, opt-in instrumentation module (e.g. `app/infrastructure/dataloader_profiler.py`) that wraps the existing DataLoader or monkey-patches `torch.utils.data.DataLoader` hooks, rather than modifying every call site. Output: structured logs + optional TensorBoard/Weights & Biases step-level scalars.

Start with a characterization run on one model (tier1_000) to collect baseline distributions for all six metrics before tuning `prefetch_factor` or adding custom prefetch logic. Baselines are needed to set the `k` multiplier and queue-depth ceilings above.
