# Prompt: Generate the Next Design Set

Run this prompt once per invocation. It emits exactly **one** design-set JSON file — the next one in priority order — then stops. Re-run to get the next.

A design set = one complete, standalone, runnable experiment specification: every axis a full comparable run needs, not just the candidate under test. Nothing may be omitted or left implicit; unused topics are still stated explicitly (see [Required topics](#required-topics), rule on `"n/a"`).

## Output

- Path: `docs/ML_Forecasting_System_Design/designsets/Tier-<n>_<idx>-<slug>.json`
- `<n>` = tier (`1`|`2`|`3`) from [04 § Decision Framework](../04-Experimentation, Evaluation & Optimization.md#decision-framework).
- `<idx>` = 3-digit, zero-padded, **restarts at `000` per tier** (`Tier-1_000`, `Tier-1_001`, …, `Tier-2_000`, …).
- `<slug>` = kebab-case candidate/axis name, e.g. `tcn-modern-tcn`, `atr-hybrid-norm`.
- One JSON object per file, valid JSON, no comments, no trailing placeholders.

## Selection algorithm

1. **Build the queue** from every tiered table in [04 § tiered candidates by layer](../04-Experimentation, Evaluation & Optimization.md#tiered-candidates-by-layer), taken in the doc's own layer order: normalization → embedding → local extraction → sequential encoding → attention → global representation → **current Stage-1 candidate set** (whole-architecture) → activation → combination strategy → fusion mechanism → multi-timeframe fusion → GBM-family. Each table row = one queue entry (`layer`, `candidate`, `tier`, `adjusted` score, `gated?`).
2. **Sort**: tier ascending (1→3) → layer order as listed above → `adjusted` score descending → row order within the table (tiebreak).
3. **Resume**: list existing files in this folder, parse `<n>`/`<idx>` from filenames, find the lowest tier with an unfilled `idx` gap (or the next `idx` after the last one present) and generate that queue entry. Never renumber or overwrite an existing file.
4. Score/tier/gate values in the design set's `prioritization` object must match the source table row verbatim — don't re-derive.

## Building a full experiment from one queue row

Every row names one axis+candidate, not a whole pipeline. Complete it into a runnable design:

- **Whole-architecture rows** (current Stage-1 candidate set): use that architecture's own `stage_config` and hyperparameter profile from [03 § architecture candidates](../03-Model & Architecture Engineering.md#architecture-candidates).
- **Sub-component rows** (embedding/local_extraction/sequential/attention/global_repr/activation/fusion): start from the [unified super-architecture skeleton](../03-Model & Architecture Engineering.md#unified-super-architecture-skeleton), set every stage to the project's current Tier-1 reference backbone (record which in `comparison_scope.backbone_ref`), and set only the stage under test to this row's candidate — the skeleton's own zeroing/numbering test.
- **Cross-cutting rows** (normalization, combination strategy, multi-timeframe fusion, GBM-family): hold the reference backbone fixed, vary only the named axis.
- **Every other topic** (data split, labels, sampling, seeds, evaluation, …) = the project's already-resolved default per [Required topics](#required-topics)'s doc-source column — never left blank, never invented.

This keeps the "one axis varied, rest held fixed" controlled-experiment discipline ([04 § Experiment Design](../04-Experimentation, Evaluation & Optimization.md#experiment-design), [§ cross-architecture fairness](../04-Experimentation, Evaluation & Optimization.md#cross-architecture-fairness)).

## Required topics

Every key below is mandatory in every design set. If a topic doesn't apply to this row's axis, set it to `"n/a"` with a one-line reason — never delete the key.

| JSON key                  | Captures                                                                                          | Doc source                                                                                                                                                       |
| -------------------------- | --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `designset_id`             | Filename stem                                                                                       | —                                                                                                                                                                    |
| `status`                   | `proposed`\|`queued`\|`running`\|`completed`\|`rejected`\|`superseded`                               | —                                                                                                                                                                    |
| `prioritization`           | `tier`, `raw`, `adjusted`, 6 scoring factors, risk/tooling modifiers, `gated`, `gate_reason`         | [04 § Decision Framework](../04-Experimentation, Evaluation & Optimization.md#decision-framework)                                                                  |
| `axis_under_test`          | `layer`, `candidate`, source table anchor                                                            | [04 § tiered candidates by layer](../04-Experimentation, Evaluation & Optimization.md#tiered-candidates-by-layer)                                                  |
| `comparison_scope`         | `backbone_ref`, `held_constant[]`, `baseline` (`{name, source}`), `hypothesis`                       | [04 § Experiment Design](../04-Experimentation, Evaluation & Optimization.md#experiment-design)                                                                    |
| `objective`                | Target(s), horizon, point vs probabilistic formulation                                               | [01-Objective.md](../01-Objective.md)                                                                                                                               |
| `data`                     | Symbol universe, snapshot policy, split scheme, purge/embargo, sample-uniqueness weighting           | [02 § training symbol universe](../02-Data, Label & Feature Engineering.md#training-symbol-universe-survivorship) / [§ validation & train/test splitting](../02-Data, Label & Feature Engineering.md#validation--traintest-splitting) / [§ overlapping labels](../02-Data, Label & Feature Engineering.md#overlapping-labels) |
| `features`                 | Candle feature schema version, feature set used, screening status                                    | [02 § candle feature schema](../02-Data, Label & Feature Engineering.md#candle-feature-schema)                                                                     |
| `labels`                   | Label scheme, horizon (min), `OM` threshold, class-imbalance handling                                | [02 § label design](../02-Data, Label & Feature Engineering.md#label-design) / [§ class imbalance handling](../02-Data, Label & Feature Engineering.md#class-imbalance-handling) |
| `normalization`            | Scheme + params                                                                                      | [02 § normalization strategy](../02-Data, Label & Feature Engineering.md#normalization-strategy)                                                                   |
| `window`                   | Per-tf length scheme (uniform/independent/tapering) + values                                         | [03 § multi-timeframe fusion](../03-Model & Architecture Engineering.md#multi-timeframe-fusion) → "per-tf window length"                                           |
| `architecture`             | `stage_config` (7 slots), backbone name, hyperparam profile (S1/S2/S3 or custom), param-count estimate, activation | [03 § architecture candidates](../03-Model & Architecture Engineering.md#architecture-candidates) / [§ design checklist](../03-Model & Architecture Engineering.md#design-checklist) |
| `combination`               | Combination strategy + fusion mechanism (or `"n/a — single-backend-wins"`)                           | [03 § combination strategy](../03-Model & Architecture Engineering.md#combination-strategy) / [§ fusion mechanism](../03-Model & Architecture Engineering.md#fusion-mechanism) |
| `training`                 | Strategy, batch size, epoch budget + early stop, per-head loss + weights, sampling, augmentation, seed count | [04 § Training Engineering](../04-Experimentation, Evaluation & Optimization.md#training-engineering)                                                              |
| `optimization`             | `fixed-config`\|`optuna-dimension`, search space if any, pruning, GPU budget                         | [04 § hyperparam search-space bounds](../04-Experimentation, Evaluation & Optimization.md#hyperparam-search-space-bounds) / [§ optimization strategy](../04-Experimentation, Evaluation & Optimization.md#optimization-strategy) |
| `evaluation`                | Per-head dev metrics, backtested KPIs (primary/guardrail/secondary), statistical-validity params, stopping criteria | [04 § per-head statistical metrics](../04-Experimentation, Evaluation & Optimization.md#per-head-statistical-metrics-dev-diagnostics) / [§ backtested trading KPIs](../04-Experimentation, Evaluation & Optimization.md#backtested-trading-kpis-final-selection) / [§ statistical validity of comparisons](../04-Experimentation, Evaluation & Optimization.md#statistical-validity-of-comparisons) |
| `alternatives_considered`  | `Alt:` list carried over from the source doc row                                                     | source table row                                                                                                                                                    |
| `known_risks`              | Relevant [99-Weakness Analysis.md](../99-Weakness Analysis.md) item IDs (e.g. `A12`, `A14`, `B5`)    | [99-Weakness Analysis.md](../99-Weakness Analysis.md)                                                                                                               |
| `doc_refs`                 | All doc anchors cited in this file                                                                   | —                                                                                                                                                                    |

## Skeleton

```json
{
  "designset_id": "Tier-1_000-<slug>",
  "status": "proposed",
  "prioritization": {"tier": 1, "raw": 0, "adjusted": 0, "evidence": 0, "dominance": 0, "modernity": 0, "resource_fit": 0, "domain_fit": 0, "impact_cost": 0, "risk_modifier": 0, "tooling_modifier": 0, "gated": false, "gate_reason": "n/a"},
  "axis_under_test": {"layer": "<layer>", "candidate": "<candidate>", "source_anchor": "04-....md#<anchor>"},
  "comparison_scope": {"backbone_ref": "<reference architecture>", "held_constant": ["<axis>", "..."], "baseline": {"name": "<baseline candidate>", "source": "<doc anchor>"}, "hypothesis": "<one line>"},
  "objective": {"targets": ["MAE", "OM", "MFE"], "horizon_min": 240, "formulation": "point|probabilistic"},
  "data": {"symbol_universe": "...", "snapshot_policy": "...", "split_scheme": "4-way", "purge_embargo_candles": 48, "uniqueness_weighting": true},
  "features": {"schema_version": "...", "feature_set": "...", "screening_status": "..."},
  "labels": {"scheme": "MFE/MAE/OM", "horizon_min": 240, "om_threshold": 1, "class_imbalance_handling": "..."},
  "normalization": {"scheme": "...", "params": {}},
  "window": {"scheme": "uniform|independent-per-tf|tapering", "per_tf_length": {}},
  "architecture": {"stage_config": {"embedding": 0, "local_extraction": 0, "sequential": 0, "attention": 0, "fusion": 0, "global_repr": 0, "heads": {}}, "backbone_name": "...", "hyperparam_profile": "S1|S2|S3|custom", "param_count_estimate": "...", "activation": "..."},
  "combination": {"strategy": "single-backend-wins", "fusion_mechanism": "n/a"},
  "training": {"strategy": "train-from-scratch", "batch_size": 0, "epoch_budget": 0, "early_stopping": "...", "loss_per_head": {}, "loss_weights": {}, "sampling": "uniform-random-shuffle", "augmentation": "none", "seed_count": 3},
  "optimization": {"mode": "fixed-config|optuna-dimension", "search_space": {}, "pruning": "...", "gpu_budget_hours": 0},
  "evaluation": {"dev_metrics": {}, "backtested_kpis": {"primary": "expectancy_per_trade", "guardrail": "max_dd", "secondary": "sortino"}, "statistical_validity": {"min_seeds": 3, "test": "paired t-test/Wilcoxon"}, "stopping_criteria": "..."},
  "alternatives_considered": ["..."],
  "known_risks": ["A12", "A14"],
  "doc_refs": ["04-....md#tiered-candidates-by-layer"]
}
```

## Rules

- One file per run. Don't batch-generate.
- Copy scores/values from the source doc verbatim; don't guess or round differently.
- If the source table itself is stale (recalibrated scores, new rows) by the time you run this, re-read the current table — it's the ground truth, this prompt only orders/schematizes it.
- If a topic's project-wide default has no doc-resolved value yet, use the doc's own stated placeholder/default and note it in `known_risks`, not a fabricated number.
