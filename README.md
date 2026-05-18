# LineageEvo

LineageEvo is a research prototype for **Lineage-Conditioned Search Priors for LLM-Based Evolutionary Factor Mining**.

The framework evolves quantitative factor expressions with an LLM while keeping the safety-critical parts deterministic:

- the LLM generates one candidate factor at a time;
- deterministic code validates the expression DSL and adapts it to Qlib;
- Qlib evaluates IC and ICIR on train/validation splits;
- a lineage DAG records mutation and crossover ancestry;
- an LLM-assisted prior rewriter maintains structured lineage/global priors;
- final selection evaluates top validation factors on the test split and can run a Qlib backtest.

## Repository Status

This repository is intended to be runnable out of the box in two modes:

- **Real experiment mode**: `qlib-smoke-run` uses an OpenAI-compatible LLM endpoint and local Qlib data by default.
- **Offline smoke mode**: `smoke-run` uses mock LLM components and does not call any external API.

Mock components are intentionally kept for tests, CI, and reproducible debugging.

## Installation

Create and activate a Python environment. The project was developed with Python 3.10+.

```powershell
conda create -n lineageevo python=3.10
conda activate lineageevo
pip install -e ".[test]"
```

For real Qlib evaluation, install Qlib as well:

```powershell
pip install -e ".[qlib]"
```

If your Qlib installation is already available in another environment, use that environment as long as it can import this package and `qlib`.

## Configuration

Copy the example environment file and fill in local secrets and paths:

```powershell
copy .env.example .env
```

Required LLM variables for real experiments:

```text
LINEAGEEVO_LLM_BASE_URL=https://api.openai.com/v1
LINEAGEEVO_LLM_API_KEY=your_api_key
LINEAGEEVO_LLM_MODEL=your_model_name
```

Required Qlib variables if they differ from `configs/default.toml`:

```text
LINEAGEEVO_QLIB_PROVIDER_URI=C:/path/to/qlib_data/cn_data
LINEAGEEVO_QLIB_REGION=cn
LINEAGEEVO_QLIB_MARKET=csi500
```

Do **not** commit `.env`. It is ignored by `.gitignore`.

Experiment settings such as market, train/validation/test dates, search budget, final selection, and backtest parameters live in:

```text
configs/default.toml
```

By default, the real experiment entry uses:

- market: `csi500`
- benchmark: `SH000905`
- train: 2018-01-02 to 2018-06-30
- validation: 2018-07-01 to 2018-12-31
- test: 2019-01-01 to 2019-12-31
- candidate LLM: `openai-compatible`
- prior rewrite LLM: `openai-compatible`

## Run a Real Experiment

After configuring `.env` and making sure Qlib can read your local data:

```powershell
python -m lineage_evo.cli qlib-smoke-run --config configs/default.toml --print-llm-io --verbose
```

Useful smaller real run:

```powershell
python -m lineage_evo.cli qlib-smoke-run --config configs/default.toml --seed-count 3 --target-valid 5 --final-top-k 3 --print-llm-io --verbose
```

The command name `qlib-smoke-run` is kept for compatibility, but with the default config it is the main real-data experiment entry.

To force mock LLM components while still using Qlib evaluation:

```powershell
python -m lineage_evo.cli qlib-smoke-run --candidate-llm mock --prior-llm mock --target-valid 5
```

## Offline Smoke Run

This path never calls a real LLM API:

```powershell
python -m lineage_evo.cli smoke-run
```

It is useful for checking that the search loop, DAG, logging, prior manager, and final output writers still work.

## LLM Dry Run

Use dry runs to check prompt formatting and API configuration without running search:

```powershell
python -m lineage_evo.cli llm-dry-run --kind candidate --llm openai-compatible --print-llm-io
python -m lineage_evo.cli llm-dry-run --kind prior-rewrite --llm openai-compatible --print-llm-io
```

## Outputs

Each run creates a unique directory under `runs/`. The key files are:

- `config_snapshot.json`: merged configuration and component names;
- `candidate_log.jsonl`: raw LLM candidate output and generation/validation/evaluation status;
- `prior_rewrite_log.jsonl`: old prior, raw rewritten prior, accepted prior, schema status, fallback status;
- `dag_events.jsonl`: mutation and crossover DAG events;
- `summary_results.csv`: generated/evaluated/failure counters;
- `final_factor_pool.csv`: active pool at the end of search;
- `selected_factors.csv`: top validation factors selected for test;
- `test_ic_results.csv`: test IC/ICIR for selected factors;
- `backtest_summary.csv` and `backtest_daily_report.csv`: Qlib backtest results when enabled.

`runs/` is ignored by Git.

## Tests

Run the test suite:

```powershell
pytest
```

The test suite keeps real network calls out of automated tests. Mock components are used for deterministic regression coverage.

## Notes

- LLMs do not perform syntax validation, numerical validation, IC computation, or backtesting.
- Invalid candidates are logged but do not enter the DAG and do not trigger prior rewriting.
- Crossover children keep two parent edges but inherit only the primary lineage, chosen by higher validation ICIR.
- The current expression interface follows an AlphaPROBE-style operator DSL and is deterministically adapted to Qlib expressions.
