---
name: markdown-formatting
description: Use whenever creating or finishing an edit to ANY markdown (.md) file in this repo (docs, README, specs, planning notes). Covers source-line wrapping and heading/list conventions — no hard line-wraps inside prose (editing happens with the IDE's word-wrap feature on), no numbering on headings, and numbered lists reserved for sequences where the order itself is meaningful. Trigger before writing new markdown content and again as a pass before ending edits to existing markdown files.
---

# Markdown Formatting

## Only use a newline where it's meaningful

A newline means something: it separates one paragraph from the next, or marks another logically meaningful boundary (a list item, a heading, a table row, a fenced code line). Never insert one just to keep a source line short — write each paragraph, list item, and table row as a single source line, however long.

Why: editing happens with the IDE's word-wrap (soft-wrap) feature on, so long lines already display wrapped on screen while editing. A manual mid-sentence line break looks fine in that view but shows up as a broken sentence anywhere soft-wrap isn't active — GitHub's diff view, `git blame`, other editors, terminals — and makes the paragraph unreflowable without hunting down and removing every embedded break first.

- One line per paragraph, per bullet, per table row — no exceptions for length.
- The blank line between paragraphs/list items is a real structural break, not a wrap, and stays.
- Line breaks that are part of the content's own syntax (fenced code block contents, table row boundaries) are structural too — leave them as-is.
- When editing a file that already has mid-paragraph line breaks, join them back into single lines as part of the edit rather than adding more wrapped lines around them.

## Never number headings

No numbering on headings at any level — not the document's `#` title, not `##`/`###` sections, nothing like `## 2. Output` or `### 3.1 Setup`.

Why: a numbered heading is a second ordering system layered on top of markdown's own heading hierarchy, and it has to be hand-renumbered every time a section is added, removed, or reordered. The heading levels themselves, plus the editor's outline/TOC, already show structure without it.

## Numbered lists only when order is meaningful

Default to `-` bullets. Use `1.`/`2.`/`3.`/... only when the sequence itself carries meaning the reader must follow in that order — steps executed in sequence, a ranked/priority order, a chronological sequence. If reordering the items wouldn't change their meaning, they're not a numbered list.

Example in this repo: [PROMPT.md § Selection algorithm](../../../docs/ML_Forecasting_System_Design/designsets/PROMPT.md#selection-algorithm) numbers its steps because each depends on the previous one running first; the same file's requirement tables use bullets/table rows, not numbers, because those topics have no required order.
