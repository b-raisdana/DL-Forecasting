# Time-Series Foundation Models (TSFMs)

Excluded from [AI Trading System — Planning Notes](model-architecture-planning.md#model-architecture--selection) — that doc covers custom/from-scratch architecture design only; this file covers the pretrained-checkpoint alternative track.

Chronos, TimesFM, Moirai, Lag-Llama, PatchTST-based pretrained checkpoints

These are a different category entirely: not language models repurposed, but transformer/patching architectures pretrained from scratch on large corpora of numeric time series across many domains, then fine-tuned or used zero-shot on a new series. This is architecturally much closer to what's already in your candidate pool (decoder-only transformer over patched sequences) than to LLM-reprogramming — the "foundation model" part is about the pretraining corpus size/diversity, not about language.

full pretraining from scratch is out of scope (that's what makes them "foundation" models — large corpora, large compute, not a single-GPU exercise). But downloading a pretrained checkpoint and fine-tuning locally on your BTC/USDT + cross-pair data is plausible within your 8GB budget for the smaller Chronos/TimesFM variants, and zero-shot inference (no training at all) is cheap enough to run as a baseline comparison point. Two open questions your doc's own methodology already answers how to handle:

- (a) these models are pretrained mostly on non-financial series (retail demand, weather, web traffic, etc.) — whether that pretraining transfers to crypto price dynamics is exactly the kind of thing your MI/GBM screening and backtested-KPI discipline should settle, not assumption;
- (b) they typically output point/quantile forecasts of the raw series, not your TP/SL/action/MAE-OM label structure (see [training-data.md § model output targets](training-data.md#model-output-targets)) — you'd need a task-specific head on top regardless.
