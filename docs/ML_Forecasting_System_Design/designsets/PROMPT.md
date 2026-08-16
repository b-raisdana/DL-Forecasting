# Prompt: Generate the Next Design Set

Run this prompt once per invocation. It emits exactly **one** design-set JSONC file — the next one in priority order — then stops. Re-run to get the next.

Every design-set file is a complete, standalone, buildable engineering spec for exactly one [category](#category-assignment) (`architecture-design` / `input-data-feature` / `outcome-label-target-head`). Each category has its own fixed JSON shape (see [Required topics](#required-topics)) — used both by that category's hand-authored idx-`000` reference file and by every auto-generated idx-`001`+ single-axis test file; the difference between them is authorship and scope (see [Selection algorithm](#selection-algorithm)), not JSON shape.

## No separate bookkeeping object

A design-set file is a pure engineering spec — it does not wrap the content in a tracking/bookkeeping object (`designset_id`, `status`, `prioritization`, `axis_under_test`, `comparison_scope`, `related_designsets`, `known_risks`, `doc_refs`). Each of those needs is covered elsewhere instead:

- **which candidate/axis this file tests, and why** → `metadata.description` (prose, not structured fields)
- **tier/score** → stays in [05](../05-Prioritization Framework.md)'s own tables; cite it in `metadata.description` if useful, don't duplicate it as JSON
- **which category** → implied by the file's location (see [Category assignment](#category-assignment)), not a restated key
- **which sibling files this row assumes** → `preprocessed_input` / `outcome_heads` (architecture-design files only) — direct pointers, not a citation object
- **known risks / doc anchors** → not tracked per file; [99-Weakness Analysis.md](../99-Weakness Analysis.md) and this prompt's own cross-references remain the source of truth

**Never edit the three `.hand-made.*` files** — they're the fixed reference every generated file is compared against.

## Output

- `architecture-design` is the **core design** — its files sit directly under `docs/ML_Forecasting_System_Design/designsets/<name>.jsonc`, no subfolder.
- `input-data-feature` and `outcome-label-target-head` are satellite categories — their files sit under `docs/ML_Forecasting_System_Design/designsets/<category>/<name>.jsonc`.
- See [Category assignment](#category-assignment) for which category a given axis belongs to.
- Filename pattern: `<name>` = `Tier-<n>_<idx>.<slug>`:
  - `<n>` = tier (`1`\|`2`\|`3`) from [05 § Decision Framework](../05-Prioritization Framework.md#decision-framework).
  - `<idx>` = 3-digit, zero-padded, **restarts at `000` per tier, scoped to the candidate's location** (root for `architecture-design`, its subfolder for the other two).
  - `<slug>`: idx `000` is always `hand-made` — the category's hand-authored reference (see [step 0](#selection-algorithm)), never produced by this prompt. Every other idx uses a short, abbreviated snake_case slug always ending `_sel` (selected) — pack in every distinguishing aspect of the candidate (technology/option chosen, and axis if not obvious from context) while staying readable, e.g. `atr_hyb_norm_sel`, `tcn_dilated_sel`. Use the [abbreviation glossary](#abbreviation-glossary) so terms stay consistent across files instead of ad hoc per file.
- `.jsonc` extension (JSON with `//` comments) — see [Rules](#rules).

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
| dilated | `dil` | convolution | `conv` | baseline | `base` |

## Category assignment

Every design set belongs to exactly one category, chosen by what it varies:

| Category | Covers | Location | Source tables/sections in [05](../05-Prioritization Framework.md) |
| --- | --- | --- | --- |
| `architecture-design` | Model backbone/topology and how it's trained/sized — **the core design**: every satellite-category choice ultimately plugs into one of these as the reference backbone | `designsets/` root | [§ tiered candidates by layer](../05-Prioritization Framework.md#tiered-candidates-by-layer): input/feature embedding, local feature extraction, sequential encoding, attention/dependency, global representation, current Stage-1 candidate set, activation mechanisms, combination strategy, fusion mechanism, multi-timeframe fusion, GBM-family. Plus its embedded search space (see [Embedded search space](#embedded-search-space)): learning rate, dropout, weight decay, optimizer, batch size, epoch/early-stopping budget — **not** model/backbone sizing, see [Embedded search space](#embedded-search-space) |
| `input-data-feature` | What goes into the model | `designsets/input-data-feature/` | [§ normalization strategy](../05-Prioritization Framework.md#normalization-strategy). Plus its embedded search space: per-tf window/sequence-length scheme, feature parameters |
| `outcome-label-target-head` | What the model is trained to predict and how heads are weighted/scored | `designsets/outcome-label-target-head/` | No scored candidate table exists yet in 05 (the [per-head statistical metrics](../04-Experimentation, Evaluation & Optimization.md#per-head-statistical-metrics-dev-diagnostics) table in 04 lists alternatives but isn't tiered — score it with the [Decision Framework](../05-Prioritization Framework.md#decision-framework) formula before queuing a row here). Plus its embedded search space: label parameters, threshold parameters (e.g. `OM` threshold), loss-function parameters, per-head loss-weight vector |

If a future candidate table doesn't fit a row above, classify it by the same rule (backbone/topology → `architecture-design`, model input → `input-data-feature`, model output/label/head → `outcome-label-target-head`) and add it to this table.

Every architecture-design file names the exact `input-data-feature`/`outcome-label-target-head` files it assumes via `preprocessed_input`/`outcome_heads` (see [Required topics](#required-topics)) — no separate citation object needed now that search-space content lives inside each category's own reference file instead of standalone bundle files.

## Selection algorithm

0. **Category-reference prerequisite**: each category's idx-`000` file (`designsets/Tier-1_000.hand-made.jsonc`, `designsets/input-data-feature/Tier-1_000.hand-made.input.jsonc`, `designsets/outcome-label-target-head/Tier-1_000.hand-made.outcome.jsonc`) is that category's reference — today's resolved default for every axis in the category, plus the category's own embedded search space (see [Embedded search space](#embedded-search-space)). It's authored directly, not produced by walking steps 1–3 below: assembling a coherent, buildable whole-architecture design (or a satellite file's full variation set) out of per-axis winners is a holistic judgment call, not a mechanical queue pop. **If a category's reference file doesn't exist yet, stop and say so** rather than attempting to generate one via the steps below. All 3 reference files already exist, so this step is satisfied — proceed straight to step 1, and only revisit step 0 if a reference file is ever superseded by a new one (never by editing the old one in place, see [Rules](#rules)).
1. **Build the queue** from every tiered table in [05 § tiered candidates by layer](../05-Prioritization Framework.md#tiered-candidates-by-layer), taken in the doc's own layer order: normalization → embedding → local extraction → sequential encoding → attention → global representation → **current Stage-1 candidate set** (whole-architecture) → activation → combination strategy → fusion mechanism → multi-timeframe fusion → GBM-family. Each table row = one queue entry (`layer`, `candidate`, `tier`, `adjusted` score, `gated?`).
2. **Sort**: tier ascending (1→3) → `adjusted` score descending → layer order as listed above (tiebreak only, for equal scores) → row order within the table (tiebreak). Score genuinely drives order this way — sorting by layer order before score would drain a whole layer's Tier-1 rows before ever reaching a higher-scoring row in a later layer, e.g. queuing embedding's Tier-1 rows (adjusted 10, 9) to completion before local extraction's TCN/ModernTCN row (adjusted 11, the doc's single highest Tier-1 score).
3. **Resume**: for each queue entry in sorted order, its target location is fixed by [Category assignment](#category-assignment) (today: every layer above except normalization → `designsets/` root; normalization → `designsets/input-data-feature/`).
   - **Skip a queue entry whose candidate is already the exact value the category reference (idx `000`) embeds for that axis** — no controlled variation would result from generating it (e.g. the local-feature-extraction table's `TCN / ModernTCN` row is already the reference's own `local_extraction` choice; the GBM-family table's `LightGBM` row is already the reference's own `auxiliary_model` choice; the sequential-encoding table's `LSTM` row and the attention table's `GQA/MQA` row are likewise already embedded). Skipped entries don't consume an idx slot — keep walking the sorted queue.
   - Otherwise: list existing `Tier-*` files at that location, parse `<n>`/`<idx>`, find the lowest tier with an unfilled `idx` gap (or the next `idx` after the last one present, at that location), and generate that queue entry. Never renumber or overwrite an existing file, and never touch a `.hand-made.*` reference file — it's authored by hand, per step 0.
4. When a design set's `metadata.description` cites a score/tier from [05](../05-Prioritization Framework.md), copy it verbatim — don't re-derive or re-round it.

## Building a full experiment from one queue row

Every row names one axis+candidate, not a whole pipeline. Complete it into a runnable design, in the schema for its category (see [Required topics](#required-topics)):

- **Whole-architecture rows** (current Stage-1 candidate set): use that architecture's own `stage_config` and hyperparameter profile from [03 § architecture candidates](../03-Model & Architecture Engineering.md#architecture-candidates) — a full `the_model` tree built from that candidate's own stages, not a one-stage diff against the category reference. When the queue row's name is a merged `X/Y` pair (per [05's tool-identity test](../05-Prioritization Framework.md#tool-identity-test-when-a-xy-grouping-stays-one-row)) and only one of `X`/`Y` has its own doc-published S1/S2/S3 profile in [03 § architecture candidates](../03-Model & Architecture Engineering.md#architecture-candidates) — e.g. "TCN / ModernTCN" merges into one scored row in 05, but only plain TCN has a published profile there; ModernTCN is discussed only as a config-level upgrade path in [03 § local feature extraction](../03-Model & Architecture Engineering.md#local-feature-extraction), with no S1/S2/S3 of its own — instantiate the variant that actually has published numbers. Never invent a new hyperparameter/config axis (e.g. a categorical switch between the two) that no doc names for it; copy verbatim per [Rules](#rules) instead. The copied profile's depth/kernel-size/structural hyperparameters stay verbatim, but its width hyperparameters still need fitting to the hardware budget before the file is finalized — see [Memory sizing convention § fitting the budget](#memory-sizing-convention): a source doc's S1/S2/S3 numbers are a starting shape, not a target VRAM utilization.
- **Sub-component rows** (embedding/local_extraction/sequential/attention/global_repr/activation/fusion): start from the category reference's `the_model` tree, keep every stage identical to it, and change only the stage under test to this row's candidate — name the reference file and the swapped stage in `metadata.description` (e.g. "`Tier-1_000` reference backbone with `dependency_modeling.attention` swapped from GQA to MLA"). This is the [unified super-architecture skeleton](../03-Model & Architecture Engineering.md#unified-super-architecture-skeleton)'s own zeroing/numbering test, expressed as a full standalone file rather than a `stage_config` diff.
- **Cross-cutting rows** (normalization, combination strategy, multi-timeframe fusion, GBM-family): same pattern — hold the reference fixed, vary only the named axis, describe the swap in `metadata.description`.
- **State the actual value inline for anything specific to this candidate** — `the_model`, `searchable_architecture_parameter_sets`, `complete_flow`, `visualized`: never a pointer like "see Tier-1_000..." for something that belongs to this run. Project-wide constants that don't vary per design set — data split scheme, symbol universe, purge/embargo, Optuna/Hyperband mechanics, trading-KPI definitions — live once in [02](../02-Data, Label & Feature Engineering.md)/[04](../04-Experimentation, Evaluation & Optimization.md) and aren't restated per file; `preprocessed_input`/`outcome_heads` (pointers, not inlined copies) and this prompt's own doc cross-references cover that instead.
- **`preprocessed_input` / `outcome_heads`** (architecture-design files only): paths to the specific `input-data-feature` / `outcome-label-target-head` files this row assumes — almost always the category references (idx `000`) unless the row under test is itself in one of those categories.

This keeps the "one axis varied, rest held fixed" controlled-experiment discipline ([04 § Experiment Design](../04-Experimentation, Evaluation & Optimization.md#experiment-design), [§ cross-architecture fairness](../04-Experimentation, Evaluation & Optimization.md#cross-architecture-fairness)).

## Embedded search space

Each category's parameter/window/label search space lives inside its own category reference file (idx `000`) — there's no separate bundle file type:

- `designsets/Tier-1_000.hand-made.jsonc` embeds the architecture/training/hyperparameter search space in `searchable_architecture_parameter_sets` — one `"<n>-base"` block per named base config, each parameter's line commented with its search bounds (see [Precise sizing convention](#precise-sizing-convention)).
- `designsets/input-data-feature/Tier-1_000.hand-made.input.jsonc` embeds the window/feature-parameter search space as numbered `variations` (each a complete alternative scheme) under one shared `base_definitions` block.
- `designsets/outcome-label-target-head/Tier-1_000.hand-made.outcome.jsonc` embeds the label/loss-parameter search space the same way: numbered top-level entries, each a complete alternative head config.

A future single-axis row (e.g. testing one specific window-length variation, or one specific normalization scheme) still owes a **complete, standalone value** for its own axis in its own file — it doesn't need to re-embed the whole search space again, only state which reference variation/entry it's testing against, per [Building a full experiment](#building-a-full-experiment-from-one-queue-row).

Whether a reference file's variation `"1"` is today's already-resolved default or the first rung of a not-yet-adopted incremental ladder isn't a fixed rule — match whatever [02](../02-Data, Label & Feature Engineering.md)/[03](../03-Model & Architecture Engineering.md) actually establish for that axis. The input reference's `variations.1` is today's resolved default (256/tf uniform) because the architecture-design schema doesn't restate `window`/`normalization` inline anywhere else — the input file is the *only* place that value is documented, so it has to be one of the numbered entries, not just the untried alternatives. The outcome reference's two entries are instead both rungs of the not-yet-adopted probabilistic-head ladder (mean+std, then +skew+kurtosis, per [02 § model output targets](../02-Data, Label & Feature Engineering.md#model-output-targets)) — the point-estimate baseline needs no search-space entry at all, since "not adopted" is itself the answer.

An embedded search-space entry earns its place only by adding something no single design-set file already states inline — an untried alternative, or a shared methodology — never by restating an already-inlined value a second time. Concrete consequence: **model/backbone sizing stays out of any shared/cross-candidate list** — sizing hyperparameter names and values are backbone-specific (ModernTCN's `ModernTCN_kernel_size`/`ModernTCN_depth`/`ModernTCN_channels` mean nothing to a Transformer's `d_model`/`num_heads`), so each candidate's own `searchable_architecture_parameter_sets` carries its own sizing entries and nothing shared duplicates them.

## Required topics

Three fixed schemas, one per category — every file in a category uses its schema regardless of idx.

### architecture-design files

| JSON key | Captures | Doc source |
| --- | --- | --- |
| `metadata.design_set` | Sequential ID, `DS-<nn>`. Only `architecture-design` files carry `metadata` at all — the two satellite schemas have no `metadata` object (see [input-data-feature files](#input-data-feature-files)/[outcome-label-target-head files](#outcome-label-target-head-files)), so the counter only ever counts `architecture-design` files. `DS-01` = `Tier-1_000.hand-made.jsonc`; increment from the highest `DS-<nn>` already present among existing `architecture-design` files | — |
| `metadata.name` / `version` / `description` | Short human title; semver-ish version (`1.0.0` for a first cut); prose naming every distinguishing technology/option choice vs. the category reference | — |
| `preprocessed_input` | Path to the `input-data-feature` file this design assumes | [Category assignment](#category-assignment) |
| `outcome_heads` | Path to the `outcome-label-target-head` file this design assumes | [Category assignment](#category-assignment) |
| `auxiliary_features` | `source` / `role` of the flattened last-candle snapshot feeding the MLP head's optional input and the GBM specialist | [03 § auxiliary tabular models (GBM-family)](../03-Model & Architecture Engineering.md#auxiliary-tabular-models-gbm-family) |
| `the_model.per_timeframe_processing` | One branch per input tf (5m/15m/1H/4H/1D/1W — see [02 § glossary](../02-Data, Label & Feature Engineering.md#glossary) tf-ordered-list), each an ordered `temporal_encoder` stage list matching this candidate's `stage_config` — a stage this candidate zeroes at the per-branch level (`embedding`, `sequential`, `local_extraction_post`) is simply not listed, per the hand-made reference's own convention (it omits `embedding` entirely for its ModernTCN+LSTM branches, no placeholder entry); explain the zeroing in `metadata.description` if it's worth calling out, not with an inline placeholder value | [03 § unified super-architecture skeleton](../03-Model & Architecture Engineering.md#unified-super-architecture-skeleton) |
| `the_model.fusion` | Multi-timeframe fusion block and its own sub-config | [03 § multi-timeframe fusion](../03-Model & Architecture Engineering.md#multi-timeframe-fusion) |
| `the_model.dependency_modeling.attention` | Post-fusion attention block — omit the key entirely if this candidate zeroes the attention stage (same "not listed" convention as `per_timeframe_processing` above; a `//` comment near the omission is fine, a placeholder JSON value like `"n/a"` is not — the hand-made reference never uses one) | [03 § attention / dependency](../03-Model & Architecture Engineering.md#attention--dependency) |
| `the_model.dependency_modeling.regularization_and_stabilization` | Normalization placement, residual connections, dropout — state exactly what the candidate's own pseudocode in [03 § architecture candidates](../03-Model & Architecture Engineering.md#architecture-candidates) shows; omit `normalization` entirely (don't invent a placeholder) if the source pseudocode has no explicit normalization layer | [03 § design layers to pass](../03-Model & Architecture Engineering.md#design-layers-to-pass) step 4 |
| `the_model.dependency_modeling.global_representation` | Pooling stage | [03 § global representation](../03-Model & Architecture Engineering.md#global-representation) |
| `the_model.dependency_modeling.prediction_head` | Shared MLP head shape (held constant across candidates) + pointer to the outcome file's head definitions | [02 § model output targets](../02-Data, Label & Feature Engineering.md#model-output-targets) |
| `the_model.dependency_modeling.auxiliary_model` | GBM specialist config (held constant unless GBM-family is the axis under test) | [03 § auxiliary tabular models (GBM-family)](../03-Model & Architecture Engineering.md#auxiliary-tabular-models-gbm-family) |
| `the_model.ensemble` | Deep-model + GBM combination (held constant unless this row tests it) | [03 § combination strategy](../03-Model & Architecture Engineering.md#combination-strategy) |
| `the_model.training.optimization` | Optimizer/scheduler/AMP/grad-clip shape (held constant — values come from `searchable_architecture_parameter_sets`) | [04 § Training Engineering](../04-Experimentation, Evaluation & Optimization.md#training-engineering) |
| `experiment_controller.hyperparameter_optimization` | Optuna pruning config (held constant) | [04 § optimization strategy](../04-Experimentation, Evaluation & Optimization.md#optimization-strategy) |
| `searchable_architecture_parameter_sets` | `"<n>-base"` block(s): every architecture-specific + shared hyperparameter, base value + `//` search-bound comment, per [Precise sizing convention](#precise-sizing-convention) | [03 § architecture candidates](../03-Model & Architecture Engineering.md#architecture-candidates) |
| `memory_budget` | VRAM/RAM estimate for this candidate's `1-base` config, against the hardware budget: static `model_parameters` (weights + gradients + AdamW optimizer state — param-count-scaled, independent of batch size), per-step `transient_activation_memory` (forward/backward activations — batch×seq×d_model-scaled), `preloaded_input_label_cache` (RAM-resident windowed dataset, sized from this row's `input_set`/`outcome_set` selection), `other_memory_consumers` (CUDA/driver context, staging buffers, fragmentation/OS headroom, GBM specialist) | [03 § hardware constraints](../03-Model & Architecture Engineering.md#hardware-constraints) / [§ vram/ram budget split](../03-Model & Architecture Engineering.md#vramram-budget-split); computed per [Memory sizing convention](#memory-sizing-convention) |
| `complete_flow` | Ordered pipeline-stage name list, matching this candidate's actual non-zeroed stages | — |
| `visualized` | ASCII pipeline diagram, matching `complete_flow` | — |

### input-data-feature files

| JSON key | Captures | Doc source |
| --- | --- | --- |
| `base_definitions` | Shared terms/derivations this file's `variations` build on (extremum/nearest/`plus2TF`/`plus3TF`/ATR/`candle_dataset` field formulas) — copy verbatim from [02 § candle feature schema](../02-Data, Label & Feature Engineering.md#candle-feature-schema)/[§ glossary](../02-Data, Label & Feature Engineering.md#glossary) unless the axis under test changes a formula itself | [02 § candle feature schema](../02-Data, Label & Feature Engineering.md#candle-feature-schema) |
| `variations` | Numbered (`"1"`, `"2"`, ...) complete alternatives for the **one** axis this file covers (window-length scheme, or normalization scheme, or feature-set variant — a new axis gets a new file, not a new key alongside existing variations) — include today's resolved default as one entry if this file is the only place that default is documented (see [Embedded search space](#embedded-search-space)) | [03 § multi-timeframe fusion](../03-Model & Architecture Engineering.md#multi-timeframe-fusion) → "per-tf window length" / [02 § normalization strategy](../02-Data, Label & Feature Engineering.md#normalization-strategy) |

### outcome-label-target-head files

| JSON key | Captures | Doc source |
| --- | --- | --- |
| `"<n>"` (numbered top-level entries) | One complete alternative head config per entry — `action_head` (`classes`/`task`/`activation`/`loss`/`loss_weight`) plus the price-level head group (`mean_std_pairs` / `mean_std_skew_kurtosis_pairs` / a future point-estimate or quantile variant), each with `heads[]`, `task`, `activation`, `loss`, `loss_weight` | [02 § model output targets](../02-Data, Label & Feature Engineering.md#model-output-targets) |

## Skeletons

### architecture-design skeleton

```jsonc
{
  "metadata": {
    "design_set": "DS-<nn>",
    "name": "<short human title>",
    "version": "1.0.0",
    "description": "<what this candidate is and why, vs. the Tier-1_000 reference — name the swapped stage/axis and the source-doc candidate/score if citing one>"
  },
  "preprocessed_input": "input-data-feature/Tier-1_000.hand-made.input.jsonc",
  "outcome_heads": "outcome-label-target-head/Tier-1_000.hand-made.outcome.jsonc",
  "auxiliary_features": {
    "source": "preprocessed_input.candle_dataset, LAST candle only per timeframe branch (not the full window), flattened across all 6 branches",
    "role": "shared tabular snapshot feeding prediction_head.MLP's optional_auxiliary_features input and the LightGBM specialist"
  },
  "the_model": {
    "per_timeframe_processing": {
      "architecture": "independent_parallel_branches",
      "branches": {
        "5m_encoded_sequence": { "temporal_encoder": [ { "<stage_name>": "<block>" } ] },
        "15m_encoded_sequence": { "temporal_encoder": [ "...same shape as 5m..." ] },
        "1H_encoded_sequence": { "temporal_encoder": [ "..." ] },
        "4H_encoded_sequence": { "temporal_encoder": [ "..." ] },
        "1D_encoded_sequence": { "temporal_encoder": [ "..." ] },
        "1W_encoded_sequence": { "temporal_encoder": [ "..." ] }
      }
    },
    "fusion": { "multitimeframe_fusion": { "<block>": { "...": "..." } } },
    "dependency_modeling": {
      // omit "attention" entirely here if stage_config.attention = 0 for this candidate — no placeholder key/value
      "attention": { "<block>": { "...": "..." } },
      // omit "normalization" entirely if the candidate's own pseudocode has no explicit normalization layer
      "regularization_and_stabilization": { "training_stability": { "normalization": {}, "residual_connections": {}, "dropout": {} } },
      "global_representation": { "pooling": "<value, or \"searchable\">" },
      "prediction_head": {
        "MLP": {
          "fusion_concatenation": { "inputs": ["deep_temporal_representation", "optional_auxiliary_features"] },
          "each_layer": [ { "Dense": "searchable", "activation": "searchable" }, { "Dropout": "searchable" } ],
          "MLP_depth": "searchable",
          "MLP_width": "searchable"
        },
        "output_heads": { "defined_in": "Tier-1_000.hand-made.outcome.jsonc" }
      },
      "auxiliary_model": { "tabular_model": { "GBM": { "LightGBM": { "inputs": "auxiliary_features", "quantile_regression_loss": "pinball", "outputs": ["q10", "q50", "q90"], "integration": { "external_ensemble": { "inside_tensorflow_graph": false } } } } } }
    },
    "ensemble": { "model_combination": { "deep_model_plus_GBM": { "components": ["Keras_temporal_model", "LightGBM_tabular_model"], "combination": { "calibrated_ensemble": { "weighting": "searchable" } } } } },
    "training": { "optimization": { "optimizer": "searchable", "scheduler": { "learning_rate_schedule": "searchable" }, "mixed_precision": "enabled", "gradient_clipping": { "enabled": true, "norm": "searchable" } } }
  },
  "experiment_controller": { "hyperparameter_optimization": { "Optuna": { "pruning": { "enabled": true, "role": "terminate_unpromising_trials_early" } } } },
  "searchable_architecture_parameter_sets": {
    // stepping convention (base * 4 / base / 4 per level, with documented exceptions): see PROMPT.md § Precise sizing convention
    "1-base": {
      "input_set": 1, // 2, 3, 4 — selects a variation from preprocessed_input's "variations"
      "<arch_specific_hparam>": "<base value>, // <low>, <high>",
      "MLP_depth": 4, // 2, 8 - intentionally is not stepped by 4.
      "MLP_width": 512, // 128, 2048
      "dropout": 0.1, // 0.0, 0.5
      "activation": "GELU", // ReLU, LeakyReLU, ELU
      "pooling_method": "searchable", // mean, max, attention, last_token
      "batch_size": 128, // 32, 512
      "learning_rate": 0.0003, // 0.00001, 0.001
      "weight_decay": 0.0001, // 0.0, 0.01
      "optimizer": "AdamW", // SGD, Adam, RMSprop
      "scheduler": "cosine", // step, exponential
      "gradient_clip_norm": 1.0, // 0.1, 5.0
      "outcome_set": 1, // 2 — selects a set from outcome_heads
      "ensemble_weighting": "validation_optimized_scalar" // fixed_equal_weight, stacked_meta_learner
    }
  },
  "memory_budget": {
    // order-of-magnitude estimate for this file's "1-base" config, per Memory sizing convention — not profiler-measured, profile_trial_cost() is ground truth
    "hardware_reference": "RTX 4060 Laptop GPU, 8GB VRAM (8188 MiB), 64GB RAM — per 03-Model & Architecture Engineering.md#hardware-constraints",
    "model_parameters": { "vram_mb": "<value>", "basis": "<per-stage param-count formula and rollup, see Memory sizing convention>" },
    "transient_activation_memory": { "vram_mb": "<value>", "basis": "<batch_size × seq_len × d_model × depth × bytes, per stage>" },
    "preloaded_input_label_cache": { "ram_gb": "<value or 'capacity-bound, see basis'>", "basis": "<bytes/sample from this row's input_set/outcome_set, sized against the RAM budget's cache slice>" },
    "other_memory_consumers": { "vram_mb": "<value>", "ram_gb": "<value>", "basis": "<CUDA/staging/fragmentation/OS overhead + GBM specialist, per vram/ram budget split>" },
    "estimated_total": { "vram_mb": "<value>", "vram_pct_of_8gb_card": "<value>", "flag": "<fits/negligible/needs profiling — name the dominant term>" }
  },
  "complete_flow": ["preprocessed_inputs", "...", "Optuna_hyperparameter_optimization"],
  "visualized": ["INPUT", "..."]
}
```

### input-data-feature skeleton

```jsonc
{
  "base_definitions": {
    "<term>": "<definition, copied verbatim from 02 § candle feature schema / § glossary unless this axis changes it>"
  },
  "variations": {
    "1": { "<tf_minutes>": "<value for this variation>" },
    "2": { "<tf_minutes>": "<value for this variation>" }
  }
}
```

### outcome-label-target-head skeleton

```jsonc
{
  "1": {
    "action_head": { "classes": ["long", "short", "none"], "task": "multiclass_classification", "activation": "softmax", "loss": "categorical_crossentropy", "loss_weight": 1.0 },
    "<price_level_head_group_name>": { "heads": ["mfe", "rer"], "task": "<...>", "activation": "<...>", "loss": "<...>", "loss_weight": 1.0 }
  }
}
```

## Precise sizing convention

Referenced from [03 § architecture candidates](../03-Model & Architecture Engineering.md#architecture-candidates) (the plain-conv candidate's `conv_layers` list) — this section is that link's target, covering how `searchable_architecture_parameter_sets` states both a value and its search bounds.

**Which S1/S2/S3 profile to base a candidate on, when nothing else in the row decides it:** the S1/S2/S3 labels (depth-heavy/width-heavy/context-heavy, per [03 § architecture candidates](../03-Model & Architecture Engineering.md#architecture-candidates)) name a *general per-architecture-family* emphasis, not a guaranteed ranking on every derived property for every architecture — verify against the candidate's own numbers rather than assuming the label. Worked example: TCN's receptive field ≈ `1 + (kernel_size − 1) × (2^num_dilated_levels − 1)`. Plugging in 03's own S1/S2/S3 numbers (`hidden_channels`/`kernel_size`/`num_dilated_levels` = 40/3/10, 96/3/3, 56/9/6) gives S1 ≈ 2047, S2 ≈ 15, S3 ≈ 505 — S1 dominates despite carrying the "depth-heavy" label rather than "context-heavy." For a candidate whose zeroed stages (e.g. no `sequential`/`attention`) leave `local_extraction` as the sole source of long-range context, that computed number — not the label — is what should drive the profile choice. State whichever reasoning applies in `metadata.description`.

**State an explicit list, never a formula.** A per-layer/per-block hyperparameter (conv channel counts, kernel sizes, a growth schedule) is written as one entry per layer, in order (e.g. `conv_layers = [(64,3), (128,5), (192,7), (256,9)]`), never compressed into a depth-count + growth-formula pair (`depth=4, growth=1.5x`) — a formula makes the reader compute actual values themselves, which breaks the "state the actual value inline" rule in [Building a full experiment](#building-a-full-experiment-from-one-queue-row).

**Step numeric search bounds by ×4 from the base value, by default** — one level down (`base / 4`) and one level up (`base × 4`), recorded as a `//` comment on that parameter's line, e.g. `"LSTM_hidden_units": 256, // 64, 1024`. This is deliberately coarse now, to be fine-tuned with smaller steps later. Three documented exceptions, all present in the hand-made reference — use them, don't invent new ones:

- **Small bounded integer counts** (layer/depth counts like `LSTM_layers`, `MLP_depth`) — ×4 would blow past any realistic value (`4 × 4 = 16` layers). Mark the line `// intentionally is not stepped by 4` and give a small custom range instead (the hand-made reference uses `4, // 1` for `LSTM_layers` and `4, // 2, 8` for `MLP_depth`).
- **Cross-parameter validity constraints** — when raw ×4 would violate a stated invariant (e.g. `Perceiver_latent_dim % GQA_heads == 0`, `GQA_heads % GQA_kv_heads == 0`, `GQA_kv_heads < GQA_heads`), clamp the offending bound to the nearest value that keeps the invariant, and leave the constraint itself as a `//` comment next to the parameters it governs (as the hand-made reference does directly above `GQA_layers`).
- **Log-scale or domain-bounded parameters** (`learning_rate`, `dropout`, `weight_decay`, `gradient_clip_norm`) — ×4 arithmetic doesn't track how these are actually searched (log-uniform for LR, `[0, 1)` for dropout). Use the field's own conventional search range instead (e.g. `learning_rate: 0.0003, // 0.00001, 0.001`; `dropout: 0.1, // 0.0, 0.5`).
- **Categorical parameters** (`optimizer`, `scheduler`, `activation`, `pooling_method`, `ensemble_weighting`, and selector fields like `input_set`/`outcome_set`) — no numeric bounds; list the alternative named options as the `//` comment instead (e.g. `"optimizer": "AdamW", // SGD, Adam, RMSprop`).

Recompute against the actual base value each time — don't copy another candidate's bound comment onto a different base number.

## Memory sizing convention

`architecture-design` files only — the two satellite categories are search-space definitions with no single model attached, so there's no VRAM/model-params number to compute for them; a satellite file's window/label sizes feed into whichever architecture-design file's `memory_budget` selects them via `input_set`/`outcome_set`, not into a budget of their own. Extends [03 § hardware constraints](../03-Model & Architecture Engineering.md#hardware-constraints)'s per-candidate requirement ("total-parameter estimate against this budget... flag if it's not obviously negligible") and [§ vram/ram budget split](../03-Model & Architecture Engineering.md#vramram-budget-split) from a generic rough prior into this specific candidate's own worked numbers — order-of-magnitude, not profiler-measured, same epistemic status as 03's own worked example (`profile_trial_cost()` remains ground truth).

**The four buckets, and why split this way:** `model_parameters` and `transient_activation_memory` are split by what they scale with, not by VRAM-vs-RAM — weights/gradients/optimizer-state scale with param count only (fixed once the architecture and `1-base` hyperparameters are chosen, independent of `batch_size`); activations scale with `batch_size × seq_len` and are reallocated every training step. That distinction is what "model-params" vs "transient-params" means here. `preloaded_input_label_cache` and `other_memory_consumers` are the remaining two named buckets from [Required topics](#required-topics)' `memory_budget` row.

- **`model_parameters` (VRAM, static).** Per-stage param-count formulas (state which stage(s) a candidate actually has, per its `stage_config` — omit zeroed stages, same convention as `the_model` itself):
  - Dense/Linear: `in_dim × out_dim` (bias negligible, omit).
  - LSTM layer: `4 × (input_size + hidden_size + 1) × hidden_size` (4 gates); stack layers with layer 1's `input_size` = the prior stage's output width, layers 2+ use `hidden_size` for both.
  - TCN dilated-conv residual block: `2 × kernel_size × channels²` (two stacked conv1d layers per block, standard TCN residual design) `+ channels²` only if that level's channel count changes from the previous one (1×1 residual-projection conv).
  - ModernTCN block (large-kernel/grouped-conv, per [03 § local feature extraction](../03-Model & Architecture Engineering.md#local-feature-extraction)): `kernel_size × channels` (depthwise conv) `+ 2 × channels × (4 × channels)` (ConvNeXt-style inverted-bottleneck ConvFFN, 4× expansion).
  - Multi-head / GQA attention layer: `Q_proj + K_proj + V_proj + O_proj`, where `Q_proj = O_proj = d_model²`, and `K_proj = V_proj = d_model × (kv_heads × head_dim)` with `head_dim = d_model / heads` (GQA's KV-head reduction shows up here; plain MHA is the `kv_heads = heads` special case) `+ FFN: 2 × d_model × (4 × d_model)` (standard 4× Transformer FFN expansion).
  - Perceiver cross-attention: same attention formula, `Q_proj` sized off `latent_dim`, `K_proj`/`V_proj` sized off the source sequence's own dim (per this file's own `pre_fusion` embedding note, not necessarily `latent_dim`) — plus the learnable latent-token bank (`latent_tokens × latent_dim`, usually negligible).
  - Roll up: `weight_bytes ≈ total_param_count × 16 bytes` (fp32 master weights + fp32 gradient + AdamW's 2 fp32 moment buffers — 4 param-sized copies, matching [03's worked example](../03-Model & Architecture Engineering.md#hardware-constraints) "~4× params, fp32" convention exactly, so the same shortcut is reusable here without re-deriving it).
- **`transient_activation_memory` (VRAM, per-step).** Per 03's own framing ("activation memory (batch × seq × d_model)" — [§ capacity/sizing layer](../03-Model & Architecture Engineering.md#design-layers-to-pass), [§ hardware constraints worked example](../03-Model & Architecture Engineering.md#hardware-constraints)): for each stage, `batch_size × seq_len × d_model_at_that_stage × depth_at_that_stage × 2 bytes` (fp16/bf16 under AMP — `mixed_precision: enabled` is this file's own `the_model.training.optimization` setting, so activations are stored at 2 bytes, not 4). Sum across stages; note which stage's `seq_len` is the full concatenated multi-tf window (pre-fusion) vs. a compressed token count (e.g. a Perceiver latent bottleneck reduces the sequence a downstream attention stage sees from "all timeframes' raw window" to `Perceiver_latent_tokens`) — this is usually where the dominant term lives, per 03's finding that activation memory, not param count, is the binding VRAM constraint.
- **`preloaded_input_label_cache` (RAM).** `bytes_per_sample = (Σ over this row's `input_set` variation's per-tf window lengths) × feature_count_per_candle × 4 bytes` (float32) `+ label_bytes_per_sample` (from this row's `outcome_set`: one-hot/scalar size per head, typically tens of bytes, negligible next to the input side). `feature_count_per_candle` = count the fields actually enumerated in `preprocessed_input`'s `base_definitions.candle_dataset` (commented-out fields like `tf_minutes`/`age_minutes`/`candle_offset` don't count). Total training-set sample count isn't a doc-resolved number (the training symbol universe is open-ended — every Binance USDT pair ever listed, per [02 § training symbol universe](../02-Data, Label & Feature Engineering.md#training-symbol-universe-survivorship)) — don't fabricate one; instead size `bytes_per_sample` against [03's RAM budget split](../03-Model & Architecture Engineering.md#vramram-budget-split) `~65%` in-memory-cache slice (`0.65 × 64GB`) and state the resulting sample-count *capacity*, flagged as capacity-bound rather than total-bound.
- **`other_memory_consumers` (VRAM + RAM).** Pipeline-level overhead that's roughly fixed regardless of which candidate this is — reuse [03's vram/ram budget split](../03-Model & Architecture Engineering.md#vramram-budget-split) percentages verbatim rather than re-deriving per file: VRAM side = CUDA/driver context (`~5%`) + pinned CPU→GPU transfer buffer (`~10%`) + fragmentation headroom (`~10%`) of the 8GB card; RAM side = OS/Python/dataloader-worker overhead (`~10%`) + pinned staging buffers (`~15%`) + headroom (`~10%`) of 64GB. Also note the LightGBM specialist's own training-time memory here (histogram bins etc.) if `auxiliary_model` is present — order tens of MB at this feature scale, negligible next to the cache slice, so it doesn't need its own bucket.
- **`estimated_total` / flag.** Sum the VRAM buckets, state `%` of the 8GB card, and name whichever single term dominates (per-candidate, this varies — a narrow-channel TCN baseline is typically overhead-dominated; a wide hybrid with full cross-attention is typically activation-dominated). This is the file's own instantiated number — it supersedes 03's generic rough-prior percentages for this specific candidate, but both remain subordinate to an actual `profile_trial_cost()` run.

**Fitting the budget, not just measuring it.** A freshly copied S1/S2/S3 profile (per [Building a full experiment](#building-a-full-experiment-from-one-queue-row)) is a *shape*, not a config already sized for this hardware — compute `memory_budget` once from the profile's literal numbers, check `estimated_total.vram_pct_of_8gb_card` against the target band below, and resize before finalizing the file if it's well outside that band, not after.

- **Target band: ~80–85% of the 8GB card's total estimate.** Below it, the row leaves the hardware this repo is scoped to ([03 § hardware constraints](../03-Model & Architecture Engineering.md#hardware-constraints)) unused; above it, there's not enough headroom left to cover this being an unverified order-of-magnitude number rather than a `profile_trial_cost()` measurement. Every row should land in the same band regardless of how cheap its own `stage_config` naturally makes it — a candidate with fewer active stages (no attention, no sequential) needs a proportionally larger scale-up on the stages it does have — so every candidate makes comparable use of the same fixed hardware when it's its turn to train, not so a structurally cheaper architecture family is permanently left under-using the card. `other_memory_consumers` stays the fixed ~2048MB slice from [§ vram/ram budget split](../03-Model & Architecture Engineering.md#vramram-budget-split) throughout the resize — it's `model_parameters` + `transient_activation_memory` that has to grow or shrink to hit the band.
- **What to scale: width only.** Identify the row's own width hyperparameters — channel counts, hidden units, `d_model`-equivalents, latent dims/tokens — and scale *all of them* by one common factor `k`, so every ratio between them is preserved exactly. Never scale one in isolation, and never touch: depth/layer counts or kernel sizes (already their own small-bounded-integer convention above, and often load-bearing for a receptive-field/profile-choice rationale already cited in `metadata.description` — leave that reasoning intact and say so explicitly if it depends only on the untouched parameters); `batch_size` (stays Optuna-searched per [04 § hyperparam search-space bounds](../04-Experimentation, Evaluation & Optimization.md#hyperparam-search-space-bounds), not maximized to fit a budget); or `MLP_width`/`MLP_depth` (the prediction head's shape is held constant across candidates, per [Required topics](#required-topics)'s `prediction_head` row — its input dimension still grows naturally when a scaled-up backbone feeds it a wider `deep_temporal_representation`, only the head's own Dense width/depth stay put).
- **Solve `k` empirically, round to clean numbers, recompute for real.** Order-of-magnitude work, same epistemic status as the rest of this section — a couple of trial values of `k` run back through the formulas above is enough, no closed-form solve needed. Round each scaled value to a number that still satisfies the row's own cross-parameter constraints (e.g. `Perceiver_latent_dim % GQA_heads == 0`), then recompute `memory_budget` from the *rounded* values, not the theoretical `k`-scaled ones — the two differ slightly, and the file's stated numbers must match what's actually in `searchable_architecture_parameter_sets`.
- **Document the scale.** State the factor and which hyperparameters it touched in `metadata.description` (e.g. "widths scaled ×1.5 from the original 128/256/64/512 to reach the target VRAM band"), and recompute every `//` search-bound comment on a scaled parameter from its new base value per [Precise sizing convention](#precise-sizing-convention) — don't leave a bound comment computed off the old, pre-scaling base.

## Rules

- One file per run. Don't batch-generate.
- `.jsonc` extension, JSON-with-comments — `//` line comments are expected (search-bound annotations, inline clarifications like `rer = risk_edge_ratio = mae/(mfe-mae)`), not forbidden; keep every object otherwise valid JSON (no trailing commas outside what a JSONC parser accepts, no unresolved placeholder text).
- Never edit a `.hand-made.*` reference file (idx `000`) — it's the hand-authored ground truth every other file in its category is compared against; if a stage's resolved default genuinely changes, add a new higher-idx file, don't rewrite the reference in place.
- Copy scores/values from [05](../05-Prioritization Framework.md) verbatim when citing them in `metadata.description`; don't guess or re-round.
- If the source table itself is stale (recalibrated scores, new rows) by the time you run this, re-read the current table — it's the ground truth, this prompt only orders/schematizes it.
- If a topic's project-wide default has no doc-resolved value yet, use the doc's own stated placeholder/default and say so in `metadata.description`, not a fabricated number.
- A category's embedded search space (`searchable_architecture_parameter_sets` / `variations` / numbered head entries) must list every feasible option [03](../03-Model & Architecture Engineering.md)/[05](../05-Prioritization Framework.md) name for that axis, not just the value in use today — see [Embedded search space](#embedded-search-space).
- `preprocessed_input` / `outcome_heads` paths must point to files that already exist — this is why [step 0](#selection-algorithm) treats the 3 category references as a hard prerequisite.
- See [Precise sizing convention](#precise-sizing-convention) for how to state a numeric hyperparameter's value and search bounds.
- See [Memory sizing convention](#memory-sizing-convention) for how to compute `memory_budget`'s four buckets (`architecture-design` files only) — including the "fitting the budget" procedure a whole-architecture row's width hyperparameters must go through before the file is finalized, not left at a copied profile's literal, hardware-unaware numbers.
