# position-finding for labels — open issues

Scope: open issues in
[training-data.md § where can be a position?](training-data.md#where-can-be-a-position),
[§ targeting bid price](training-data.md#targeting-bid-price), and
[§ TP / MAE label](training-data.md#tp--mae-label) — the logic that
decides, using FUTURE knowledge, whether a NOW candle gets labelled Long /
Short / None. Grounded against [current-code.md](current-code.md), which
shows what's actually implemented today.

No open issues currently. The last one (single collapsed label vs.
two-headed continuous signal) was decided in favor of a single label; the
code has not been updated to match yet — tracked as a TODO in
[current-code.md](current-code.md#gaps-vs-the-plan-planingmd--training-datamd).
