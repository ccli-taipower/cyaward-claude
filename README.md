# cyaward-claude

MLB Cy Young Award voter-share regression model.

**Phase 1 (current):** train + backtest model on 2015-2024 BBWAA results.
**Phase 2 (deferred):** live 2026 dashboard. See [design spec](docs/superpowers/specs/2026-05-11-cyaward-design.md).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage (Phase 1)

```bash
python -m src.cli.build_training_data    # ~5-10 min, hits FanGraphs
python -m src.cli.train                   # < 1 min
python -m src.cli.backtest                # ~5-10 min
```

Results in `reports/backtest_v1.md`.
