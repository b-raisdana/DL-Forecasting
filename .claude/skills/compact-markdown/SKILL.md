---
name: compact-markdown
description: Use whenever creating or finishing an edit to ANY markdown (.md) file in this repo (docs, README, specs, planning notes). Keeps prose token-efficient and compact while preserving every fact, decision, number, and caveat. Trigger before writing new markdown content and again as a pass before ending edits to existing markdown files.
---

# Compact Markdown

Goal: fewest tokens, zero information loss.

## Rules

- Cut filler and throat-clearing: "It's important to note that", "In order to", "As we can see", intros that restate the heading.
- Prefer bullets/tables over prose paragraphs when listing 3+ comparable items.
- One idea per line; no padding with connective prose.
- Merge sections that repeat the same point instead of loosely cross-referencing them.
- Shortest correct word: "use" not "utilize", "to" not "in order to", "can" not "is able to".
- Drop hedging ("generally speaking", "it should be noted").
- Headers only where they aid navigation; don't nest past 3 levels.
- Don't restate the file's own title/purpose in the opening paragraph.
- Code/config snippets: show only the minimal illustrative piece, not a full file, unless the full file is the point.
- Glossary/definition-list sections: for each entry, confirm the term is actually used elsewhere — in this file, or (for a shared glossary linked from other docs, e.g. `02-Data, Label & Feature Engineering.md#glossary`) in any file that links to it. Drop entries for terms that appear nowhere else.

## Never compact away

- Explicit decisions and their rationale (the "why")
- Numbers, thresholds, formulas, dates, named parameters
- Caveats, edge cases, known limitations
- Anything the user wrote as a direct quote or hard requirement

## Before finishing an edit

Re-read the diff. If a sentence can be deleted without losing a fact, decision, or caveat, delete it. If two sentences say the same thing from different angles, keep the sharper one. Check bullet-for-bullet against the pre-edit version that nothing substantive was lost.
