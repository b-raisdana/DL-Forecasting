# Prompt: Generate the Next Design Set

Run this prompt once per invocation. It emits exactly **one** design-set JSONC file — the next one in priority order — then stops. Re-run to get the next.

Every design-set file is a complete, standalone, buildable engineering spec for exactly one [category](#category-assignment) (`architecture-design` / `input-data-feature` / `outcome-label-target-head`). Each category has its own fixed JSON shape (see [Required topics](#required-topics)) — used both by that category's hand-authored idx-`000` reference file and by every auto-generated idx-`001`+ single-axis test file; the difference between them is authorship and scope (see [Selection algorithm](#selection-algorithm)), not JSON shape. See [Format retired from earlier revisions](#format-retired-from-earlier-revisions) if you're recalling an older, more heavily-bookkept version of this format.

## Format retired from earlier revisions

An earlier revision of this prompt wrapped every design set in tracking/bookkeeping fields — `designset_id`, `status`, `category`, `prioritization`, `axis_under_test`, `comparison_scope`, `related_designsets`, `known_risks`, `doc_refs` — on top of the engineering content itself, and produced separate `Tier-1_000.<topic>_sel.json` "bundle" files for each category's parameter search space. The hand-made reference sets (`Tier-1_000.hand-made.jsonc` and its two satellite files) carry none of that: they're pure, lean engineering specs, and their embedded search space replaced the separate bundle files (see [Embedded search space](#embedded-search-space)). This prompt no longer requires the old bookkeeping fields or the bundle file type. **Never edit the three `.hand-made.*` files** — they're the fixed reference every generated file is compared against.

What replaces each retired field:

- **which candidate/axis this file tests, and why** → `metadata.description` (prose, not structured fields)
- **tier/score** → stays in [05](../05-Prioritization Framework.md)'s own tables; cite it in `metadata.description` if useful, don't duplicate it as JSON
- **which category** → implied by the file's location (see [Category assignment](#category-assignment)), not a restated key
- **which sibling files this row assumes** → `preprocessed_input` / `outcome_heads` (architecture-design files only) — direct pointers, not a `related_designsets` citation object
- **known risks / doc anchors** → not tracked per file; [99-Weakness Analysis.md](../99-Weakness Analysis.md) and this prompt's own cross-references remain the source of truth

Match this leaner shape, not the old wrapped one, even where an older (now-deleted) file in this repo might be remembered as precedent.

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
2. **Sort**: tier ascending (1→3) → `adjusted` score descending → layer order as listed above (tiebreak only, for equal scores) → row order within the table (tiebreak). Score genuinely drives order this way — sorting by layer order before score (an earlier version of this rule) would drain a whole layer's Tier-1 rows before ever reaching a higher-scoring row in a later layer, e.g. queuing embedding's Tier-1 rows (adjusted 10, 9) to completion before local extraction's TCN/ModernTCN row (adjusted 11, the doc's single highest Tier-1 score).
3. **Resume**: for each queue entry in sorted order, its target location is fixed by [Category assignment](#category-assignment) (today: every layer above except normalization → `designsets/` root; normalization → `designsets/input-data-feature/`).
   - **Skip a queue entry whose candidate is already the exact value the category reference (idx `000`) embeds for that axis** — no controlled variation would result from generating it (e.g. the local-feature-extraction table's `TCN / ModernTCN` row is already the reference's own `local_extraction` choice; the GBM-family table's `LightGBM` row is already the reference's own `auxiliary_model` choice; the sequential-encoding table's `LSTM` row and the attention table's `GQA/MQA` row are likewise already embedded). Skipped entries don't consume an idx slot — keep walking the sorted queue.
   - Otherwise: list existing `Tier-*` files at that location, parse `<n>`/`<idx>`, find the lowest tier with an unfilled `idx` gap (or the next `idx` after the last one present, at that location), and generate that queue entry. Never renumber or overwrite an existing file, and never touch a `.hand-made.*` reference file — it's authored by hand, per step 0.
4. When a design set's `metadata.description` cites a score/tier from [05](../05-Prioritization Framework.md), copy it verbatim — don't re-derive or re-round it.

## Building a full experiment from one queue row

Every row names one axis+candidate, not a whole pipeline. Complete it into a runnable design, in the schema for its category (see [Required topics](#required-topics)):

- **Whole-architecture rows** (current Stage-1 candidate set): use that architecture's own `stage_config` and hyperparameter profile from [03 § architecture candidates](../03-Model & Architecture Engineering.md#architecture-candidates) — a full `the_model` tree built from that candidate's own stages, not a one-stage diff against the category reference.
- **Sub-component rows** (embedding/local_extraction/sequential/attention/global_repr/activation/fusion): start from the category reference's `the_model` tree, keep every stage identical to it, and change only the stage under test to this row's candidate — name the reference file and the swapped stage in `metadata.description` (e.g. "`Tier-1_000` reference backbone with `dependency_modeling.attention` swapped from GQA to MLA"). This is the [unified super-architecture skeleton](../03-Model & Architecture Engineering.md#unified-super-architecture-skeleton)'s own zeroing/numbering test, expressed as a full standalone file rather than a `stage_config` diff.
- **Cross-cutting rows** (normalization, combination strategy, multi-timeframe fusion, GBM-family): same pattern — hold the reference fixed, vary only the named axis, describe the swap in `metadata.description`.
- **State the actual value inline for anything specific to this candidate** — `the_model`, `searchable_architecture_parameter_sets`, `complete_flow`, `visualized`: never a pointer like "see Tier-1_000..." for something that belongs to this run. Project-wide constants that don't vary per design set — data split scheme, symbol universe, purge/embargo, Optuna/Hyperband mechanics, trading-KPI definitions — live once in [02](../02-Data, Label & Feature Engineering.md)/[04](../04-Experimentation, Evaluation & Optimization.md) and aren't restated per file; `preprocessed_input`/`outcome_heads` (pointers, not inlined copies) and this prompt's own doc cross-references cover that instead. This is narrower than an earlier revision of this rule, which required inlining even project-wide constants — the hand-made reference sets don't do that (see [Format retired from earlier revisions](#format-retired-from-earlier-revisions)), and matching their leanness is the point.
- **`preprocessed_input` / `outcome_heads`** (architecture-design files only): paths to the specific `input-data-feature` / `outcome-label-target-head` files this row assumes — almost always the category references (idx `000`) unless the row under test is itself in one of those categories.

This keeps the "one axis varied, rest held fixed" controlled-experiment discipline ([04 § Experiment Design](../04-Experimentation, Evaluation & Optimization.md#experiment-design), [§ cross-architecture fairness](../04-Experimentation, Evaluation & Optimization.md#cross-architecture-fairness)).

## Embedded search space

Earlier revisions of this prompt kept each category's parameter/window/label search space in a separate `Tier-1_000.<topic>_sel.json` bundle file. The hand-made reference sets fold that content into the category reference itself instead — there's no separate bundle file type anymore:

- `designsets/Tier-1_000.hand-made.jsonc` embeds the architecture/training/hyperparameter search space in `searchable_architecture_parameter_sets` — one `"<n>-base"` block per named base config, each parameter's line commented with its search bounds (see [Precise sizing convention](#precise-sizing-convention)).
- `designsets/input-data-feature/Tier-1_000.hand-made.input.jsonc` embeds the window/feature-parameter search space as numbered `variations` (each a complete alternative scheme) under one shared `base_definitions` block.
- `designsets/outcome-label-target-head/Tier-1_000.hand-made.outcome.jsonc` embeds the label/loss-parameter search space the same way: numbered top-level entries, each a complete alternative head config.

A future single-axis row (e.g. testing one specific window-length variation, or one specific normalization scheme) still owes a **complete, standalone value** for its own axis in its own file — it doesn't need to re-embed the whole search space again, only state which reference variation/entry it's testing against, per [Building a full experiment](#building-a-full-experiment-from-one-queue-row).

Whether a reference file's variation `"1"` is today's already-resolved default or the first rung of a not-yet-adopted incremental ladder isn't a fixed rule — match whatever [02](../02-Data, Label & Feature Engineering.md)/[03](../03-Model & Architecture Engineering.md) actually establish for that axis. The input reference's `variations.1` is today's resolved default (256/tf uniform) because the new architecture-design schema no longer restates `window`/`normalization` inline anywhere else — the input file is now the *only* place that value is documented, so it has to be one of the numbered entries, not just the untried alternatives. The outcome reference's two entries are instead both rungs of the not-yet-adopted probabilistic-head ladder (mean+std, then +skew+kurtosis, per [02 § model output targets](../02-Data, Label & Feature Engineering.md#model-output-targets)) — the point-estimate baseline needs no search-space entry at all, since "not adopted" is itself the answer.

The same discipline that governed bundles still applies to what belongs in a reference file's embedded search space: an entry earns its place only by adding something no single design-set file already states inline — an untried alternative, or a shared methodology — never by restating an already-inlined value a second time. One concrete consequence, unchanged from before: **model/backbone sizing stays out of any shared/cross-candidate list** — sizing hyperparameter names and values are backbone-specific (ModernTCN's `ModernTCN_kernel_size`/`ModernTCN_depth`/`ModernTCN_channels` mean nothing to a Transformer's `d_model`/`num_heads`), so each candidate's own `searchable_architecture_parameter_sets` carries its own sizing entries and nothing shared duplicates them.

## Required topics

Three fixed schemas, one per category — every file in a category uses its schema regardless of idx.

### architecture-design files

| JSON key | Captures | Doc source |
| --- | --- | --- |
| `metadata.design_set` | Sequential ID, `DS-<nn>` (`DS-01` = the hand-made reference; increment across the whole `designsets/` tree, not per category) | — |
| `metadata.name` / `version` / `description` | Short human title; semver-ish version (`1.0.0` for a first cut); prose naming every distinguishing technology/option choice vs. the category reference | — |
| `preprocessed_input` | Path to the `input-data-feature` file this design assumes | [Category assignment](#category-assignment) |
| `outcome_heads` | Path to the `outcome-label-target-head` file this design assumes | [Category assignment](#category-assignment) |
| `auxiliary_features` | `source` / `role` of the flattened last-candle snapshot feeding the MLP head's optional input and the GBM specialist | [03 § auxiliary tabular models (GBM-family)](../03-Model & Architecture Engineering.md#auxiliary-tabular-models-gbm-family) |
| `the_model.per_timeframe_processing` | One branch per input tf (5m/15m/1H/4H/1D/1W — see [02 § glossary](../02-Data, Label & Feature Engineering.md#glossary) tf-ordered-list), each an ordered `temporal_encoder` stage list matching this candidate's `stage_config` | [03 § unified super-architecture skeleton](../03-Model & Architecture Engineering.md#unified-super-architecture-skeleton) |
| `the_model.fusion` | Multi-timeframe fusion block and its own sub-config | [03 § multi-timeframe fusion](../03-Model & Architecture Engineering.md#multi-timeframe-fusion) |
| `the_model.dependency_modeling.attention` | Post-fusion attention block — or the key present with an explicit `"n/a"` + one-line reason if this candidate zeroes the attention stage | [03 § attention / dependency](../03-Model & Architecture Engineering.md#attention--dependency) |
| `the_model.dependency_modeling.regularization_and_stabilization` | Normalization placement, residual connections, dropout — state exactly what the candidate's own pseudocode in [03 § architecture candidates](../03-Model & Architecture Engineering.md#architecture-candidates) shows; don't add a normalization layer the source pseudocode doesn't have | [03 § design layers to pass](../03-Model & Architecture Engineering.md#design-layers-to-pass) step 4 |
| `the_model.dependency_modeling.global_representation` | Pooling stage | [03 § global representation](../03-Model & Architecture Engineering.md#global-representation) |
| `the_model.dependency_modeling.prediction_head` | Shared MLP head shape (held constant across candidates) + pointer to the outcome file's head definitions | [02 § model output targets](../02-Data, Label & Feature Engineering.md#model-output-targets) |
| `the_model.dependency_modeling.auxiliary_model` | GBM specialist config (held constant unless GBM-family is the axis under test) | [03 § auxiliary tabular models (GBM-family)](../03-Model & Architecture Engineering.md#auxiliary-tabular-models-gbm-family) |
| `the_model.ensemble` | Deep-model + GBM combination (held constant unless this row tests it) | [03 § combination strategy](../03-Model & Architecture Engineering.md#combination-strategy) |
| `the_model.training.optimization` | Optimizer/scheduler/AMP/grad-clip shape (held constant — values come from `searchable_architecture_parameter_sets`) | [04 § Training Engineering](../04-Experimentation, Evaluation & Optimization.md#training-engineering) |
| `experiment_controller.hyperparameter_optimization` | Optuna pruning config (held constant) | [04 § optimization strategy](../04-Experimentation, Evaluation & Optimization.md#optimization-strategy) |
| `searchable_architecture_parameter_sets` | `"<n>-base"` block(s): every architecture-specific + shared hyperparameter, base value + `//` search-bound comment, per [Precise sizing convention](#precise-sizing-convention) | [03 § architecture candidates](../03-Model & Architecture Engineering.md#architecture-candidates) |
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
      "attention": { "<block, or \"n/a\" + reason if zeroed>": { "...": "..." } },
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

**State an explicit list, never a formula.** A per-layer/per-block hyperparameter (conv channel counts, kernel sizes, a growth schedule) is written as one entry per layer, in order (e.g. `conv_layers = [(64,3), (128,5), (192,7), (256,9)]`), never compressed into a depth-count + growth-formula pair (`depth=4, growth=1.5x`) — a formula makes the reader compute actual values themselves, which breaks the "state the actual value inline" rule in [Building a full experiment](#building-a-full-experiment-from-one-queue-row).

**Step numeric search bounds by ×4 from the base value, by default** — one level down (`base / 4`) and one level up (`base × 4`), recorded as a `//` comment on that parameter's line, e.g. `"LSTM_hidden_units": 256, // 64, 1024`. This is deliberately coarse now, to be fine-tuned with smaller steps later. Three documented exceptions, all present in the hand-made reference — use them, don't invent new ones:

- **Small bounded integer counts** (layer/depth counts like `LSTM_layers`, `MLP_depth`) — ×4 would blow past any realistic value (`4 × 4 = 16` layers). Mark the line `// intentionally is not stepped by 4` and give a small custom range instead (the hand-made reference uses `4, // 1` for `LSTM_layers` and `4, // 2, 8` for `MLP_depth`).
- **Cross-parameter validity constraints** — when raw ×4 would violate a stated invariant (e.g. `Perceiver_latent_dim % GQA_heads == 0`, `GQA_heads % GQA_kv_heads == 0`, `GQA_kv_heads < GQA_heads`), clamp the offending bound to the nearest value that keeps the invariant, and leave the constraint itself as a `//` comment next to the parameters it governs (as the hand-made reference does directly above `GQA_layers`).
- **Log-scale or domain-bounded parameters** (`learning_rate`, `dropout`, `weight_decay`, `gradient_clip_norm`) — ×4 arithmetic doesn't track how these are actually searched (log-uniform for LR, `[0, 1)` for dropout). Use the field's own conventional search range instead (e.g. `learning_rate: 0.0003, // 0.00001, 0.001`; `dropout: 0.1, // 0.0, 0.5`).
- **Categorical parameters** (`optimizer`, `scheduler`, `activation`, `pooling_method`, `ensemble_weighting`, and selector fields like `input_set`/`outcome_set`) — no numeric bounds; list the alternative named options as the `//` comment instead (e.g. `"optimizer": "AdamW", // SGD, Adam, RMSprop`).

Recompute against the actual base value each time — don't copy another candidate's bound comment onto a different base number.

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
