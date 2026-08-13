# **Future Work Concept: AI Market Movement Tokenization**

**Status:** Concept / future research — not currently planned for implementation

## **1. Objective**

Design a future AI subsystem that converts raw market price action into structured, machinereadable “market movement tokens.” The purpose is to give downstream trading models a higher-level representation of what the market is doing, in addition to raw candles and technical indicators.

The system would describe movements using semantic terms such as Rally, Pullback, Reversal, Breakout, Breakdown, Consolidation, Acceleration, Exhaustion, and similar concepts, while also attaching quantitative information such as start/end time, duration, amplitude, strength, volatility, velocity, and structure.

## **2. Core Concept**

Instead of treating every candle as an isolated observation, segment price action into meaningful movements and represent each movement as a structured object.

Example:

RALLY → PULLBACK → BREAKOUT → ACCELERATION → EXHAUSTION → CONSOLIDATION

Each movement would have both categorical and continuous attributes.

## **3. Proposed Movement Vocabulary**

Initial vocabulary should remain relatively small and objective.

Directional:

- Rally

- Decline

- Pullback

- Rebound

Structural / event:

- Breakout - Breakdown

- Reversal - Continuation

Non-directional:

- Consolidation

- Range - Compression

Extreme behavior:

- Spike

- Crash

- Capitulation

- Exhaustion

Traditional chart patterns such as Head & Shoulders, Flags, Pennants, and Triangles should initially be treated as higher-level interpretations rather than primary ground-truth labels.

## **4. Movement Representation**

A movement token could contain:

Identity: - movement_id - timeframe - type - direction

- phase

- market structure

Time:

- start_timestamp - end_timestamp - duration_candles - duration_minutes - time_since_previous_movement

Price / magnitude:

- start_price - end_price - net_price_change - percentage_change - amplitude - ATR-normalized amplitude

Behavior: - strength - velocity - acceleration - directional efficiency - volatility

- volume strength

- persistence

Risk / excursion:

- MFE

- MAE

- maximum drawdown during movement

- maximum extension

Structure:

- starting structure

- ending structure

- HH / HL / LH / LL relationships

- breakout/retest information

- reversal information

Quality:

- label confidence

- segmentation confidence

## **5. Labeling Philosophy**

Labels should be generated from deterministic, measurable rules rather than subjective visual interpretation.

The future implementation should define:

- when a movement starts

- when it ends

- minimum movement magnitude

- minimum duration

- reversal threshold

- breakout threshold

- consolidation criteria

- strength calculation

- movement classification rules

ATR- or volatility-adaptive thresholds are preferable to fixed percentage thresholds so that the same labeling framework can operate across different market regimes and instruments.

## **6. Hierarchical Labels**

Avoid creating one enormous categorical label such as:

BULLISH_STRONG_PULLBACK_AFTER_BREAKOUT

Instead use independent dimensions:

Direction: - UP - DOWN - NEUTRAL

Phase: - IMPULSE - CORRECTION - CONSOLIDATION

- REVERSAL

Event: - PULLBACK - BREAKOUT - BREAKDOWN - REBOUND - SPIKE - EXHAUSTION - etc.

Structure: - HH - HL - LH - LL - RANGE

This keeps the label space manageable and allows new attributes to be added independently.

## **7. Continuous Measurements**

Important movement properties should remain continuous rather than being unnecessarily converted into categories.

Examples: - strength = 0.73 - amplitude = 2.4 ATR - velocity = 0.30% per hour - duration = 8 hours - directional efficiency = 0.82

Human-readable categories such as Weak / Medium / Strong can later be derived from continuous values when useful.

## **8. Training-Label Generation**

For future supervised training, historical future candles may be used to determine what movement actually occurred after a reference candle.

Example:

Reference candle → future candles → completed movement → training label

The future information is allowed when constructing historical training labels, but must never be supplied as an input feature at inference time.

This distinction must be maintained rigorously to prevent target leakage.

## **9. Multi-Timeframe Representation**

Movement tokenization should potentially operate independently on multiple timeframes, for example:

1H:  PULLBACK / DOWN / strength 0.62 4H:  CONSOLIDATION / NEUTRAL / strength 0.31

- 1D:  UPTREND / UP / strength 0.78

1W:  UPTREND / UP / strength 0.91

The downstream model could therefore understand the current movement simultaneously at several temporal scales.

## **10. Proposed Future Architecture**

Potential future architecture:

Raw OHLCV + technical features

↓

Multi-timeframe representation

↓

Swing / pivot detection

↓ Movement segmentation

↓ Movement classification + quantitative measurements

↓ Structured movement tokens

↓

Trading / prediction model

A separate Movement Annotator could be responsible for learning to infer movement tokens from information available at inference time. A Trading Model could then consume those movement representations together with raw market features.

## **11. Movement Sequence / Market Grammar**

A later research direction is to treat movements as a sequence or graph rather than isolated events.

Example:

RALLY ↓ PULLBACK ↓ BREAKOUT ↓ ACCELERATION ↓ EXHAUSTION

The system could eventually learn transition relationships between movements, creating a kind of “market grammar.” This could be useful for sequence modeling and regime/context recognition.

## **12. Relationship to the Existing Trading AI**

This concept should complement, not immediately replace, the existing trading prediction approach.

The movement layer would represent market behavior.

The trading layer would answer questions such as:

- Long / Short / None - expected return

- expected MFE

- expected MAE

- TP achievement probabilities

- SL probability

Keeping market-movement understanding separate from trade decision-making provides a cleaner architecture and makes the movement representation potentially reusable across different trading strategies.

## **13. Important Research Questions**

Before implementation, future research should investigate:

1. What objective segmentation algorithm produces the most useful movement boundaries?

2. Should pivots be based on ATR, volatility, fractals, or another adaptive method?

3. How should overlapping movements at different timeframes be represented?

4. How much label granularity is useful before labels become noisy?

5. Which movement attributes actually improve out-of-sample trading performance?

6. Should movement tokens be predicted independently or jointly?

7. Can movement sequences improve MFE/MAE and trade-direction prediction?

8. How stable are movement definitions across different instruments?

9. Can the representation generalize from BTC/USDT to other assets?

10. How should ambiguous movement boundaries be handled?

11. How should confidence or uncertainty be represented?

12. Does movement tokenization improve performance enough to justify its additional complexity?

## **14. Key Design Principle**

The primary objective should not be to reproduce traditional technical-analysis terminology. The objective is to create an objective, quantitative representation of market behavior.

Traditional names are useful as semantic descriptions, but the underlying representation should be based on measurable properties such as direction, amplitude, duration, volatility, velocity, acceleration, structure, persistence, and excursions.

## **15. Future Implementation Scope**

This document intentionally describes a research concept rather than an implementation plan.

When this idea is revisited, the first practical deliverable should be a formal “Market Movement Label Specification” defining:

- movement taxonomy

- segmentation rules

- label-generation algorithm

- quantitative feature definitions

- multi-timeframe rules

- leakage controls

- dataset format

- evaluation methodology

Only after that specification is stable should model architecture and implementation be considered.
