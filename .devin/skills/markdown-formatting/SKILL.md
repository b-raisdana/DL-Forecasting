---
name: markdown-formatting
description: Use whenever creating or finishing an edit to ANY markdown (.md) file in this repo (docs, README, specs, planning notes). Covers source-line wrapping and heading/list conventions — no hard line-wraps inside prose (editing happens with the IDE's word-wrap feature on), no numbering on headings, and numbered lists reserved for sequences where the order itself is meaningful. Trigger before writing new markdown content and again as a pass before ending edits to existing markdown files.
---

# Markdown Formatting

## Only use a newline where it's meaningful

A newline separates one paragraph from the next, or marks another meaningful boundary (list item, heading, table row, fenced-code line) — never insert one just to keep a source line short. Write each paragraph, list item, and table row as a single source line, however long.

Why: editing happens with the IDE's word-wrap (soft-wrap) feature on, so long lines already display wrapped while editing. A manual mid-sentence break looks fine there but shows up as a broken sentence anywhere soft-wrap isn't active — GitHub's diff view, `git blame`, other editors, terminals — and makes the paragraph unreflowable without hunting down every embedded break first.

- Blank lines between paragraphs/list items are structural breaks, not wraps — they stay.
- Line breaks inside a fenced code block or between table rows are structural too — leave them as-is.
- Editing a file with existing mid-paragraph breaks: join them into single lines rather than adding more wrapped lines around them.

## Never number headings

No numbering at any level — not the `#` title, not `##`/`###` sections, nothing like `## 2. Output`. A numbered heading is a second ordering system on top of markdown's own hierarchy, hand-renumbered on every add/remove/reorder — heading levels plus the editor's outline/TOC already show structure without it.

## Numbered lists only when order is meaningful

Default to `-` bullets. Use `1.`/`2.`/... only when the sequence itself carries meaning the reader must follow — sequential steps, a ranked/priority order, a chronology. If reordering the items wouldn't change their meaning, they're not a numbered list.

Example: [PROMPT.md § Selection algorithm](../../../docs/ML_Forecasting_System_Design/designsets/PROMPT.md#selection-algorithm) numbers its steps because each depends on the previous one running first; the same file's requirement tables use bullets/table rows since those topics have no required order.
