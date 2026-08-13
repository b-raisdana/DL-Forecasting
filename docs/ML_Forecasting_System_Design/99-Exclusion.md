# Excluded Topics

The following areas are explicitly outside the current scope of the project.

**Current data assumption:** Locally cached Binance market data → feature/label engineering → ML/DL research → evaluation/backtesting.

## Self-Supervised / Contrastive Pretraining — Excluded

Idea: Pretrain the model on the project's large pool of unlabeled OHLCV before supervised training.
Methods: Masked reconstruction, contrastive learning, TS2Vec / TimeMAE-style pretraining.
Rationale: Could exploit unlabeled candles when clean OM > 1 action labels are scarce.
Pipeline: Unlabeled OHLCV → Self-supervised pretraining → Fine-tuning → Trading heads
Decision: Excluded from current architecture scope.
Reason: Adds significant complexity and an additional training stage; not considered necessary for the current design.

## Market Movement Tokenization

AI subsystem that converts raw market price action into structured, machinereadable "market movement tokens."
Instead of treating every candle as an isolated observation, segment price action into meaningful movements and represent each movement as a structured object.

## Security

- ML security
- Adversarial attacks
- Data poisoning
- Model extraction
- Prompt injection
- Cybersecurity

## MLOps / Infrastructure

- Kubernetes
- Cloud infrastructure
- CI/CD
- Distributed serving infrastructure
- Infrastructure orchestration
- Production infrastructure engineering

## Reinforcement Learning

- Q-learning
- DQN
- Policy gradients
- Actor-critic
- PPO
- Offline RL
- Multi-agent RL

## Production Decision Layer

- Production trading execution
- Broker/exchange execution architecture
- Operational position management
- Production risk-control infrastructure

## System-Level Architecture

- Large-scale distributed AI architecture
- Production system architecture
- Microservices architecture
- Infrastructure architecture

## Meta-Optimization / AutoML

- Automated feature engineering
- Neural architecture search
- Automated model selection
- Fully automated experiment generation

## Final Production Validation

- Production acceptance testing
- Canary deployment
- Blue/green deployment
- Production A/B testing
- Long-term production certification

## General AI / LLM Topics

- LLMs
- Tokenization
- Prompt engineering
- RAG
- Agents
- Fine-tuning of language models
- Multimodal AI
- Image models
- Audio models
- Video models
- General text processing

## Irrelevant Data Modalities

- Images
- Audio
- Video
- Generic sensor data
- Generic text datasets

## Alternative Market Data Acquisition

- Generic web scraping
- Generic external data acquisition
- Multiple unrelated market-data providers

## Deployment, Monitoring & Continuous Learning

- Live/production deployment
- Production monitoring and alerting
- Online/continuous learning pipelines

## Decision/System Architecture

- End-to-end trading system architecture
- Automated decision-making layer built on model output

## Deferred (project-specific)

- **transaction costs/spread/slippage/latency**: matters for sub-4H scalping, not addressed now. Revisit before live/paper trading — cost-free backtest overstates real perf.
- **risk/position sizing beyond TP targets**: handled manually via existing procedure, not by model. No AI work needed now.
- **market regime robustness/retraining cadence**: not addressed now. Revisit once live a while — crypto regime shifts (trend/range/vol), untouched model can decay silently.
- **backtest module design (concurrent-position capital/margin allocation, equity-curve compounding, fill logic, vectorized vs event-driven)**: signals can fire every 5 min with a 240-min hold, so many trades can be open simultaneously; [04's backtested KPIs](<04-Experimentation, Evaluation & Optimization.md#backtested-trading-kpis-final-selection>) (expectancy/trade, max-DD, Sortino) are defined per-trade with no spec for allocation across concurrent positions, so max-DD/Sortino are undefined without it. The "backtest module" is referenced ~6x in 04 as build-later infra but has no design yet. Known gap, deferred — not to be reported as a weakness in reviews.
- **perp-specific data channels (funding rate, open interest)**: [02's candidate feature pool](02-Data, Label & Feature Engineering.md#candidate-feature-pool) covers OBV/Ichimoku/MACD/ADX/VWAP/vol-regime/session/cross-symbol/structural but no funding-rate or OI channel. Out of scope for now — not to be reported as a weakness in reviews.
