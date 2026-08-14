# Prompt: Generate the Next Design Set

Run this prompt once per invocation. It emits exactly **one** design-set JSON file — the next one in priority order — then stops. Re-run to get the next.

Two design-set sub-types exist:

- **single-candidate** — one complete, standalone, runnable experiment spec: every axis a full comparable run needs, not just the candidate under test. Nothing may be omitted or left implicit; unused topics are still stated explicitly (see [Required topics](#required-topics), rule on `"n/a"`).
- **parameter-search bundle** — one complete catalog of every feasible test option for a parameter/hyperparameter/sizing topic, partitioned into Tier-1/2/3 subsets inside a single JSON. Not a single run — a search-space spec consumed by many Optuna trials. See [Required topics](#required-topics) and [Bundle skeleton](#bundle-skeleton).

## Output

- `architecture-design` is the **core design** — its files sit directly under `docs/ML_Forecasting_System_Design/designsets/<name>.json`, no subfolder.
- `input-data-feature` and `outcome-label-target-head` are satellite categories — their files sit under `docs/ML_Forecasting_System_Design/designsets/<category>/<name>.json`.
- See [Category assignment](#category-assignment) for which category a given axis belongs to.
- One unified filename pattern for **both** sub-types: `<name>` = `Tier-<n>_<idx>.<slug>`:
  - `<n>` = tier (`1`\|`2`\|`3`) from [04 § Decision Framework](../04-Experimentation, Evaluation & Optimization.md#decision-framework).
  - `<idx>` = 3-digit, zero-padded, **restarts at `000` per tier, scoped to the candidate's location** (root for `architecture-design`, its subfolder for the other two).
  - `<slug>` = short, abbreviated snake_case, always ending `_sel` (selected) — pack in every distinguishing aspect of the candidate (technology/option chosen, and axis if not obvious from context) while staying readable, e.g. `atr_hyb_norm_sel`, `tcn_modern_tcn_sel`. Use the [abbreviation glossary](#abbreviation-glossary) so terms stay consistent across files instead of ad hoc per file.
  - Bundles are always the **first Tier-1 entry at their location** (`Tier-1_000.<topic-slug>_sel.json`) — see [Bundle topics](#bundle-topics) — since [step 0](#selection-algorithm) generates them before any other Tier-1 file there; single-candidate Tier-1 rows at that location start at `_001`.
- One JSON object per file, valid JSON, no comments, no trailing placeholders.

### Abbreviation glossary

Reuse these across every filename so slugs stay short and mutually consistent; extend the list (don't invent one-off abbreviations) when a new term needs one.

| Full term | Abbrev. | Full term | Abbrev. | Full term | Abbrev. |
| --- | --- | --- | --- | --- | --- |
| and | `n` | selected | `sel` | parameters | `params` |
| architecture | `arch` | sizing | `sz` | hyperparameters | `hparams` |
| normalization | `norm` | hybrid | `hyb` | training | `train` |
| window | `win` | length | `len` | feature | `feat` |
| label | `lbl` | threshold | `thr` | combination | `comb` |
| embedding | `embed` | extraction | `extr` | sequential | `seq` |
| attention | `attn` | representation | `repr` | fusion | `fus` |

## Category assignment

Every design set — single-candidate or bundle — belongs to exactly one category, chosen by what it varies:

| Category | Covers | Location | Source tables/sections in [04](../04-Experimentation, Evaluation & Optimization.md) |
| --- | --- | --- | --- |
| `architecture-design` | Model backbone/topology and how it's trained/sized — **the core design**: every satellite-category choice ultimately plugs into one of these as the reference backbone | `designsets/` root | [§ tiered candidates by layer](../04-Experimentation, Evaluation & Optimization.md#tiered-candidates-by-layer): input/feature embedding, local feature extraction, sequential encoding, attention/dependency, global representation, current Stage-1 candidate set, activation mechanisms, combination strategy, fusion mechanism, multi-timeframe fusion, GBM-family. Plus its bundle topic: learning rate, dropout, weight decay, optimizer, batch size, epoch/early-stopping budget — **not** model/backbone sizing, see [Bundles hold the search space, not today's value](#bundles-hold-the-search-space-not-todays-value) |
| `input-data-feature` | What goes into the model | `designsets/input-data-feature/` | [§ normalization strategy](../04-Experimentation, Evaluation & Optimization.md#normalization-strategy). Plus its bundle topic: per-tf window/sequence-length scheme, feature parameters |
| `outcome-label-target-head` | What the model is trained to predict and how heads are weighted/scored | `designsets/outcome-label-target-head/` | No scored candidate table exists yet in 04 (the [per-head statistical metrics](../04-Experimentation, Evaluation & Optimization.md#per-head-statistical-metrics-dev-diagnostics) table lists alternatives but isn't tiered — score it with the [Decision Framework](../04-Experimentation, Evaluation & Optimization.md#decision-framework) formula before queuing a row here). Plus its bundle topic: label parameters, threshold parameters (e.g. `OM` threshold), loss-function parameters, per-head loss-weight vector |

If a future candidate table doesn't fit a row above, classify it by the same rule (backbone/topology → `architecture-design`, model input → `input-data-feature`, model output/label/head → `outcome-label-target-head`) and add it to this table.

Every single-candidate design set, in any of the 3 categories, must cross-reference the specific bundle/sibling files it draws its bundle-owned held-constant topics from, via `related_designsets` — see [Required topics](#required-topics) and [Building a full experiment § bundle-owned topics](#building-a-full-experiment-from-one-single-candidate-queue-row).

## Selection algorithm

0. **Bundle prerequisite**: if any of the 3 bundle files ([Bundle topics](#bundle-topics)) is missing, generate the next missing one — fixed order `architecture-design` → `input-data-feature` → `outcome-label-target-head` — and stop. Bundles are referenced by every single-candidate row's `training`/`optimization` topic, so they're generated first. Once all 3 exist, proceed to step 1.
1. **Build the queue** from every tiered table in [04 § tiered candidates by layer](../04-Experimentation, Evaluation & Optimization.md#tiered-candidates-by-layer), taken in the doc's own layer order: normalization → embedding → local extraction → sequential encoding → attention → global representation → **current Stage-1 candidate set** (whole-architecture) → activation → combination strategy → fusion mechanism → multi-timeframe fusion → GBM-family. Each table row = one queue entry (`layer`, `candidate`, `tier`, `adjusted` score, `gated?`).
2. **Sort**: tier ascending (1→3) → `adjusted` score descending → layer order as listed above (tiebreak only, for equal scores) → row order within the table (tiebreak). Score genuinely drives order this way — sorting by layer order before score (an earlier version of this rule) would drain a whole layer's Tier-1 rows before ever reaching a higher-scoring row in a later layer, e.g. queuing embedding's Tier-1 rows (adjusted 10, 9) to completion before local extraction's TCN/ModernTCN row (adjusted 11, the doc's single highest Tier-1 score).
3. **Resume**: for each queue entry, its target location is fixed by [Category assignment](#category-assignment) (today: every layer above except normalization → `designsets/` root; normalization → `designsets/input-data-feature/`). List existing `Tier-*` files at that location, parse `<n>`/`<idx>`, find the lowest tier with an unfilled `idx` gap (or the next `idx` after the last one present, at that location) — walking the global queue from step 2 to find the first entry whose location+tier slot is still open — and generate that queue entry. Never renumber or overwrite an existing file.
4. Score/tier/gate values in the design set's `prioritization` object must match the source table row verbatim — don't re-derive.

## Building a full experiment from one single-candidate queue row

Every row names one axis+candidate, not a whole pipeline. Complete it into a runnable design:

- **Whole-architecture rows** (current Stage-1 candidate set): use that architecture's own `stage_config` and hyperparameter profile from [03 § architecture candidates](../03-Model & Architecture Engineering.md#architecture-candidates).
- **Sub-component rows** (embedding/local_extraction/sequential/attention/global_repr/activation/fusion): start from the [unified super-architecture skeleton](../03-Model & Architecture Engineering.md#unified-super-architecture-skeleton), set every stage to the project's current Tier-1 reference backbone (record which in `comparison_scope.backbone_ref`), and set only the stage under test to this row's candidate — the skeleton's own zeroing/numbering test.
- **Cross-cutting rows** (normalization, combination strategy, multi-timeframe fusion, GBM-family): hold the reference backbone fixed, vary only the named axis.
- **Every topic gets its actual value inline, always — no exceptions.** `window`, `features`, `labels`, `training`, `architecture.hyperparam_profile`, every required key: state the concrete value/choice used in *this* run, in the file itself. A reader must be able to understand the full experiment from this one JSON alone, never by chasing another file for what a field actually means. This holds even for topics a [bundle](#bundle-topics) also covers — the bundle documents the *full search space + rationale* for that topic (every feasible option, tiered), which is a different, richer job than this row's own few-word "what value does this run use." Never replace a field's value with a pointer like `"see Tier-1_000..."`; that breaks the "complete, standalone, runnable" requirement this whole prompt exists to enforce.
- **`related_designsets`** (required in every single-candidate row, any category): a *citation* list of the bundle/sibling files that informed this row's held-constant values — lineage, not content. It sits alongside the inline values above, never in place of them.

This keeps the "one axis varied, rest held fixed" controlled-experiment discipline ([04 § Experiment Design](../04-Experimentation, Evaluation & Optimization.md#experiment-design), [§ cross-architecture fairness](../04-Experimentation, Evaluation & Optimization.md#cross-architecture-fairness)).

## Bundle topics

Sourced from 04's own [§ Parameter Optimization](../04-Experimentation, Evaluation & Optimization.md#parameter-optimization) and [§ Hyperparameter Optimization](../04-Experimentation, Evaluation & Optimization.md#hyperparameter-optimization) bullet lists — each bullet assigned to whichever category owns it:

| Bundle file | Location | Parameter-optimization bullets it covers | Key doc anchors |
| --- | --- | --- | --- |
| `Tier-1_000.train_hparams_sel.json` | `designsets/` root (`architecture-design`) | Training parameters (batch size, epoch budget), learning rate, dropout, weight decay, optimizer. **Not** Model parameters (sizing) — see [Bundles hold the search space, not today's value](#bundles-hold-the-search-space-not-todays-value) | [§ hyperparam search-space bounds](../04-Experimentation, Evaluation & Optimization.md#hyperparam-search-space-bounds), [§ batch-size strategy](../04-Experimentation, Evaluation & Optimization.md#batch-size-strategy), [§ epoch / training-budget selection](../04-Experimentation, Evaluation & Optimization.md#epoch--training-budget-selection) |
| `Tier-1_000.win_len_n_feat_params_sel.json` | `designsets/input-data-feature/` | Window parameters (untried schemes only: independent-per-tf/tapering — **not** the current uniform-256/tf default), Feature parameters (candidate pool + screening method — **not** the current v1 schema) | [§ multi-timeframe fusion](../03-Model & Architecture Engineering.md#multi-timeframe-fusion) → "per-tf window length" |
| `Tier-1_000.lbl_thr_n_loss_params_sel.json` | `designsets/outcome-label-target-head/` | Label parameters, Threshold parameters (`OM` threshold), loss-function parameters, per-head loss-weight vector | [§ loss-weight selection](../04-Experimentation, Evaluation & Optimization.md#loss-weight-selection), [§ per-head statistical metrics](../04-Experimentation, Evaluation & Optimization.md#per-head-statistical-metrics-dev-diagnostics), [02 § label design](../02-Data, Label & Feature Engineering.md#label-design) |

Each bundle's `search_space` must state **every feasible option worth testing for that topic**, split into `tier_1`/`tier_2`/`tier_3` (same tiering discipline as [04 § Decision Framework](../04-Experimentation, Evaluation & Optimization.md#decision-framework): Tier 1 = doc-named default/primary, must-test; Tier 2 = doc-named alternative, secondary; Tier 3 = speculative/parked). Where 04 states a bound is profiler-derived rather than fixed (`profile_trial_cost()`/`max_trials_for_budget()`), record that mechanism as the option's actual value — never fabricate a number the doc doesn't give.

Every bundle also carries `related_designsets` with exactly one `input_data_feature` path and exactly one `outcome_label_target_head` path (see [Required topics § Bundle only](#bundle-only)) — a bundle whose own category is one of those two self-references (names its own file), since the 3 bundle filenames above are fixed and predictable, referencing a sibling bundle by that fixed name is valid even before it's physically generated.

### Bundles hold the search space, not today's value

Every design set already inlines its own current value for every topic (per [Building a full experiment](#building-a-full-experiment-from-one-single-candidate-queue-row)). That means a bundle's `tier_1`/`tier_2`/`tier_3` entries must never just restate "here's what's used today" — that's already sitting, mandatorily, in every referencing design set's own field. A bundle entry earns its place only by adding something no single design set already states: an untried alternative, or a shared methodology/process. Two concrete consequences already hit in practice:

- **Model/backbone sizing never appears in the `architecture-design` bundle**, even though 04 names it under "Model parameters." Sizing hyperparameter *names and values* are backbone-specific — TCN's `hidden_channels`/`kernel_size`/`num_dilated_levels` mean nothing to a Transformer's `d_model`/`num_heads` — so a shared bundle entry could only ever be vague filler or a growing per-backbone table duplicating what's already correct in each candidate's own `architecture.hyperparam_profile` (see e.g. `Tier-1_002.tcn_modern_tcn_sel.json`'s `hidden_channels S1:40/S2:96/S3:56, kernel_size S1:3/S2:3/S3:9, num_dilated_levels S1:10/S2:3/S3:6`). All of it is per-candidate; none of it belongs here.
- **The `input-data-feature` bundle's `window`/`feature` entries skip the current default** (uniform-256/tf window scheme, v1 feature schema) for the same reason — every design set already inlines those in its own `window`/`features` fields. The bundle covers only what isn't stated anywhere else: untried window schemes (independent-per-tf, tapering), not-yet-promoted candidate features, and the shared MI-screening methodology.

## Required topics

Distributed by design-set sub-type — a key only appears where that sub-type actually needs it.

### Core (every design set, both sub-types)

| JSON key | Captures | Doc source |
| --- | --- | --- |
| `designset_id` | Filename stem | — |
| `status` | `proposed`\|`queued`\|`running`\|`completed`\|`rejected`\|`superseded` | — |
| `category` | `architecture-design`\|`input-data-feature`\|`outcome-label-target-head` | [Category assignment](#category-assignment) |
| `known_risks` | Relevant [99-Weakness Analysis.md](../99-Weakness Analysis.md) item IDs (e.g. `A12`, `A14`, `B5`) | [99-Weakness Analysis.md](../99-Weakness Analysis.md) |
| `doc_refs` | All doc anchors cited in this file | — |

### Single-candidate only

Mandatory in every single-candidate design set. If a topic doesn't apply to this row's axis, set it to `"n/a"` with a one-line reason — never delete the key. "Primary in" names the category that typically varies that key; the other two categories still fill it in, held constant (see [Building a full experiment](#building-a-full-experiment-from-one-single-candidate-queue-row)).

| JSON key | Captures | Doc source | Primary in |
| --- | --- | --- | --- |
| `prioritization` | `tier`, `raw`, `adjusted`, 6 scoring factors, risk/tooling modifiers, `gated`, `gate_reason` | [04 § Decision Framework](../04-Experimentation, Evaluation & Optimization.md#decision-framework) | all |
| `axis_under_test` | `layer`, `candidate`, source table anchor | [04 § tiered candidates by layer](../04-Experimentation, Evaluation & Optimization.md#tiered-candidates-by-layer) | all |
| `comparison_scope` | `backbone_ref`, `held_constant[]`, `baseline` (`{name, source}`), `hypothesis` | [04 § Experiment Design](../04-Experimentation, Evaluation & Optimization.md#experiment-design) | all |
| `related_designsets` | Paths to the specific sibling design-set/bundle files (any category, including the row's own) this row's bundle-owned topics are drawn from | — | all |
| `objective` | Target(s), horizon, point vs probabilistic formulation | [01-Objective.md](../01-Objective.md) | `outcome-label-target-head` |
| `data` | Symbol universe, snapshot policy, split scheme, purge/embargo, sample-uniqueness weighting | [02 § training symbol universe](../02-Data, Label & Feature Engineering.md#training-symbol-universe-survivorship) / [§ validation & train/test splitting](../02-Data, Label & Feature Engineering.md#validation--traintest-splitting) / [§ overlapping labels](../02-Data, Label & Feature Engineering.md#overlapping-labels) | `input-data-feature` |
| `features` | Candle feature schema version, feature set used, screening status | [02 § candle feature schema](../02-Data, Label & Feature Engineering.md#candle-feature-schema) | `input-data-feature` |
| `labels` | Label scheme, horizon (min), `OM` threshold, class-imbalance handling | [02 § label design](../02-Data, Label & Feature Engineering.md#label-design) / [§ class imbalance handling](../02-Data, Label & Feature Engineering.md#class-imbalance-handling) | `outcome-label-target-head` |
| `normalization` | Scheme + params | [02 § normalization strategy](../02-Data, Label & Feature Engineering.md#normalization-strategy) | `input-data-feature` |
| `window` | Per-tf length scheme (uniform/independent/tapering) + values | [03 § multi-timeframe fusion](../03-Model & Architecture Engineering.md#multi-timeframe-fusion) → "per-tf window length" | `input-data-feature` |
| `architecture` | `stage_config` (7 slots), backbone name, hyperparam profile (S1/S2/S3 or custom), param-count estimate, activation | [03 § architecture candidates](../03-Model & Architecture Engineering.md#architecture-candidates) / [§ design checklist](../03-Model & Architecture Engineering.md#design-checklist) | `architecture-design` |
| `combination` | Combination strategy + fusion mechanism (or `"n/a — single-backend-wins"`) | [03 § combination strategy](../03-Model & Architecture Engineering.md#combination-strategy) / [§ fusion mechanism](../03-Model & Architecture Engineering.md#fusion-mechanism) | `architecture-design` |
| `training` | Strategy, batch size, epoch budget + early stop, per-head loss + weights, sampling, augmentation, seed count | [04 § Training Engineering](../04-Experimentation, Evaluation & Optimization.md#training-engineering) | `architecture-design` |
| `optimization` | `fixed-config`\|`optuna-dimension`, search space if any, pruning, GPU budget | [04 § hyperparam search-space bounds](../04-Experimentation, Evaluation & Optimization.md#hyperparam-search-space-bounds) / [§ optimization strategy](../04-Experimentation, Evaluation & Optimization.md#optimization-strategy) | `architecture-design` |
| `evaluation` | Per-head dev metrics, backtested KPIs (primary/guardrail/secondary), statistical-validity params, stopping criteria | [04 § per-head statistical metrics](../04-Experimentation, Evaluation & Optimization.md#per-head-statistical-metrics-dev-diagnostics) / [§ backtested trading KPIs](../04-Experimentation, Evaluation & Optimization.md#backtested-trading-kpis-final-selection) / [§ statistical validity of comparisons](../04-Experimentation, Evaluation & Optimization.md#statistical-validity-of-comparisons) | `outcome-label-target-head` |
| `alternatives_considered` | `Alt:` list carried over from the source doc row | source table row | all |

### Bundle only

Mandatory in every parameter-search bundle instead of the single-candidate table above.

| JSON key | Captures | Doc source |
| --- | --- | --- |
| `type` | Literal `"parameter-search-bundle"` | — |
| `bundle_scope` | `topic`, `parameter_optimization_source[]` (bullets from 04's Parameter/Hyperparameter Optimization lists), `applies_to` (which trials/queue this bundle governs) | [04 § Parameter Optimization](../04-Experimentation, Evaluation & Optimization.md#parameter-optimization) / [§ Hyperparameter Optimization](../04-Experimentation, Evaluation & Optimization.md#hyperparameter-optimization) |
| `related_designsets` | Exactly one `input_data_feature` path + exactly one `outcome_label_target_head` path (strings, not arrays) — the canonical file in each of the other two categories this bundle assumes; a bundle self-references its own category's slot | [Bundle topics](#bundle-topics) |
| `search_space` | `tier_1[]`/`tier_2[]`/`tier_3[]`, each entry `{parameter, options[], rationale, source_anchor}` | [Bundle topics](#bundle-topics) |
| `optimization_mechanics` | Search method, pruning, budget control | [04 § optimization strategy](../04-Experimentation, Evaluation & Optimization.md#optimization-strategy) |

## Single-candidate skeleton

```json
{
  "designset_id": "Tier-1_001.<abbrev_slug>_sel",
  "status": "proposed",
  "category": "architecture-design|input-data-feature|outcome-label-target-head",
  "prioritization": {"tier": 1, "raw": 0, "adjusted": 0, "evidence": 0, "dominance": 0, "modernity": 0, "resource_fit": 0, "domain_fit": 0, "impact_cost": 0, "risk_modifier": 0, "tooling_modifier": 0, "gated": false, "gate_reason": "n/a"},
  "axis_under_test": {"layer": "<layer>", "candidate": "<candidate>", "source_anchor": "04-....md#<anchor>"},
  "comparison_scope": {"backbone_ref": "<reference architecture>", "held_constant": ["<axis>", "..."], "baseline": {"name": "<baseline candidate>", "source": "<doc anchor>"}, "hypothesis": "<one line>"},
  "related_designsets": {"architecture_design": ["Tier-1_000.train_hparams_sel.json"], "input_data_feature": ["input-data-feature/Tier-1_000.win_len_n_feat_params_sel.json"], "outcome_label_target_head": ["outcome-label-target-head/Tier-1_000.lbl_thr_n_loss_params_sel.json"]},
  "objective": {"targets": ["MAE", "OM", "MFE"], "horizon_min": 240, "formulation": "point|probabilistic"},
  "data": {"symbol_universe": "...", "snapshot_policy": "...", "split_scheme": "4-way", "purge_embargo_candles": 48, "uniqueness_weighting": true},
  "features": {"schema_version": "...", "feature_set": "...", "screening_status": "..."},
  "labels": {"scheme": "MFE/MAE/OM", "horizon_min": 240, "om_threshold": 5, "class_imbalance_handling": "..."},
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

## Bundle skeleton

```json
{
  "designset_id": "Tier-1_000.<abbrev_topic_slug>_sel",
  "status": "proposed",
  "category": "architecture-design|input-data-feature|outcome-label-target-head",
  "type": "parameter-search-bundle",
  "bundle_scope": {
    "topic": "<topic name>",
    "parameter_optimization_source": ["<bullet from 04 § Parameter/Hyperparameter Optimization>", "..."],
    "applies_to": "<which trials/queue this governs, e.g. every Optuna trial across all architecture-design candidates>"
  },
  "related_designsets": {"input_data_feature": "input-data-feature/Tier-1_000.win_len_n_feat_params_sel.json", "outcome_label_target_head": "outcome-label-target-head/Tier-1_000.lbl_thr_n_loss_params_sel.json"},
  "search_space": {
    "tier_1": [{"parameter": "<name>", "options": ["<value|range|mechanism>", "..."], "rationale": "<why must-test>", "source_anchor": "04-....md#<anchor>"}],
    "tier_2": [{"parameter": "<name>", "options": [], "rationale": "", "source_anchor": ""}],
    "tier_3": [{"parameter": "<name>", "options": [], "rationale": "", "source_anchor": ""}]
  },
  "optimization_mechanics": {"search_method": "Optuna TPE + Hyperband pruning", "pruning": "...", "budget_control": "estimate_total_budget()/max_trials_for_budget(), per-trial profile_trial_cost()"},
  "known_risks": ["..."],
  "doc_refs": ["04-....md#parameter-optimization", "04-....md#hyperparameter-optimization"]
}
```

## Rules

- One file per run. Don't batch-generate.
- Copy scores/values from the source doc verbatim; don't guess or round differently.
- If the source table itself is stale (recalibrated scores, new rows) by the time you run this, re-read the current table — it's the ground truth, this prompt only orders/schematizes it.
- If a topic's project-wide default has no doc-resolved value yet, use the doc's own stated placeholder/default and note it in `known_risks`, not a fabricated number.
- A bundle's `search_space` must list every feasible option 04 names for that topic (not just the current default) — the point of a bundle is completeness, unlike a single-candidate row which isolates one axis.
- `related_designsets` paths must point to files that already exist — this is why [step 0](#selection-algorithm) generates all 3 bundles before any `architecture-design` row that would need to cite them.
- When a candidate's `hyperparam_profile`/`searchable_architecture_parameter_sets` states a base numeric value with implied search bounds, step those bounds by ×4 (not ×2) from the base in each direction — deliberately coarse now, to be fine-tuned with smaller steps later.
