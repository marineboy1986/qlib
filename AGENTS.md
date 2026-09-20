# AGENTS.md — Qlib

Qlib is Microsoft's quantitative investment platform: a Python library for building the full quant research workflow (data → features → models → portfolios → backtest → experiment tracking). `README.md` is the user-facing entry point; this file is the working agreement for agents editing this repository.

## Repository map

```
qlib/            the shipped Python package (this is what `pyqlib` installs)
  data/          data providers, expressions/operators, cache, storage backends, datasets, processors
  model/         model base classes, trainer, ensembling, meta-learning, interpretation, risk models
  strategy/      base strategy classes; concrete strategies are under contrib/strategy/
  backtest/      exchange, account, executor, position, decision, reporting, high-freq backtest
  workflow/      experiment management (`R`), record templates, task generation, online serving
  rl/            reinforcement learning for order execution (Linux-only tests)
  contrib/       model zoo and contributed models/strategies — not core
  utils/         Serializable, `init_instance_by_config`, config helpers, data utilities
  cli/           `qrun` workflow runner and the `qlib.cli.data` downloader
  tests/         test-side helpers (`qlib.tests.config`, `qlib.tests.data`) shared by `tests/`
tests/           the pytest suite (runs with `cd tests && python -m pytest .`)
docs/            Sphinx documentation (component/, advanced/, developer/, start/)
examples/        runnable benchmarks and tutorials, driven by YAML configs
scripts/         data collectors and `get_data.py` downloaders
Reference/       personal, non-upstream demos (see "Reference demos" below)
```

## Environment setup

Python 3.8–3.12; CI runs all five minor versions on Linux, Windows and macOS. Use a virtualenv or conda environment — there is no committed environment file, and the system interpreter in a fresh checkout has none of the dependencies installed.

```bash
python -m pip install numpy && python -m pip install --upgrade cython   # build prerequisites
make install          # == `make prerequisite dependencies`
# `make dev` additionally installs every optional extra; `make ci-install` mirrors CI
pip install -e ".[dev]"   # the usual contributor install
```

- The Cython extensions `qlib.data._libs.rolling` / `expanding` come from `setup.py` (`pip install` builds them), and `make prerequisite` compiles them in place from `qlib/data/_libs/*.pyx` when they are missing. Until they exist, `qlib/data/ops.py` raises `ImportError` — so a source checkout is not importable before an install or `make prerequisite`.
- Optional extras (`pyproject.toml`): `dev`, `lint`, `docs`, `test`, `analysis`, `client`, `rl`, `package`. Prefer the smallest extra that covers your change.
- For local installs that must match CI, export `PIP_CONSTRAINT="$(pwd)/.github/ci/constraints.txt"`; the bounds there (`mlflow<3.13`, `filelock>=3.16.0,<3.30`, `plotly<7`, `black<26.1`, …) are compatibility fixes with documented reasons in `.github/ci/README.md` and are asserted against `pyproject.toml` by `tests/test_ci_configuration.py`. **When you change a bound, change it in both files and run the offline check below.**

## Commands

| Task | Command |
| --- | --- |
| Lint (black + pylint + flake8 + mypy + nbqa) | `make lint` |
| Format check only | `make black` (black `-l 120`) |
| Unit tests (fast) | `cd tests && python -m pytest . -m "not slow"` |
| Unit tests (slow) | `cd tests && python -m pytest . -m "slow"` |
| End-to-end pipeline | `python -m pytest tests/test_all_pipeline.py` |
| Workflow smoke test | `python qlib/cli/run.py examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml` |
| CI policy check | `python -m unittest discover -s tests -p test_ci_configuration.py` |
| Docs build | `make docs-gen` |
| Build wheel | `make build` |

Notes:
- Run pytest **from `tests/`** (`tests/pytest.ini` defines the `slow` marker, and that is exactly what CI does). Tests are `unittest`-style and many require downloaded provider data.
- `mypy` excludes `qlib/data`, `qlib/model`, `qlib/strategy`, `qlib/workflow`, `qlib/contrib`, `qlib/utils`, `qlib/tests`, `qlib/config.py`, `qlib/log.py` and `qlib/__init__.py` (`.mypy.ini`), so type checking covers only a small surface — do not read a green `make mypy` as whole-repo verification.
- `make lint` runs black, pylint, flake8, mypy and nbqa; the exact arguments live in the `Makefile` and are the source of truth (`docs/developer/code_standard_and_dev_guide.rst` shows an older flake8 ignore list).
- Pre-commit hooks run black (on `qlib`) and flake8: `pre-commit install`.

## Required data

Provider data is not committed. Default provider URI is `~/.qlib/qlib_data/cn_data` (`qlib/config.py`); US data goes to `~/.qlib/qlib_data/us_data` and is selected with `region=REG_US` from `qlib.constant`.

```bash
python scripts/get_data.py qlib_data --name qlib_data_simple --target_dir ~/.qlib/qlib_data/cn_data --interval 1d --region cn
python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn        # full daily data
```

The official dataset is currently disabled; the README documents a community mirror (`chenditc/investment_data` releases). Tests inherit `qlib.tests.TestAutoData`, which downloads `qlib_data_simple` into `~/.qlib/qlib_data/cn_data_simple` (and full/1min data when asked) through `qlib.tests.data.GetData` — so the suite needs network access on a cold machine and some test classes will fail offline.

## Extension points

Everything in a workflow YAML resolves by import path, not by registry: `qlib.utils.init_instance_by_config` / `get_callable_kwargs` (`qlib/utils/mod.py`) read a `{"class": ..., "module_path": ..., "kwargs": {...}}` block, import the module, and call the class. `qrun` (`qlib/cli/run.py`) loads the YAML, runs `qlib.init` from its `qlib_init` section, resolves `task.model` / `task.dataset`, trains through `qlib.model.trainer.task_train`, and records results under `qlib.workflow.R`.

Practical consequences:
- To add a model, dataset, handler, strategy or executor, implement the class and reference it from config — there is no central registry to update, but the dotted path must be importable from the installed package.
- `qrun` renders Jinja2 templates in the YAML (so `${VAR}`-style environment substitution works), merges a `BASE_CONFIG_PATH` overlay, and applies any `sys.path` / `sys.rel_path` entries before importing your class. Instantiating a `record` resolves names against `qlib.workflow.record_temp` by default.
- Config keys are **not** schema-validated: `Config.validate` checks only `provider_uri` and `region`, and unknown keys merely log `Unrecognized config`. A typo in a workflow YAML fails silently or much later, so verify behavior rather than assuming the key took effect.
- Operators are the exception to path-based resolution: `qlib.data.ops` uses explicit registration (`register_all_ops`, plus `custom_ops` passed to `qlib.init`) because expressions are parsed, not imported.
- A new requirement that only some users need should become an optional extra in `pyproject.toml` plus a lazily imported module, never a hard import from the data/backtest spine.

## Conventions

- Formatting: black with `-l 120`; flake8 with the Makefile ignore list; pylint with the Makefile disable list (add `# pylint: disable=...` inline for genuine exceptions). Docstrings: numpydoc style.
- Commit and PR titles must follow Conventional Commits — `feat`, `fix`, `ci`, `docs`, `refactor`, `chore`, `test`, `perf`, `build`, `style`, `revert`, `Release-As`, max 100 chars (`.commitlintrc.js`, enforced by `.github/workflows/lint_title.yml`).
- Keep changes scoped to the layer that owns the behavior. The data/backtest/workflow spine must stay importable without the optional heavy stacks: `torch` is imported only by `qlib/rl/**` and specific `qlib/contrib/**` models, and importing `qlib.contrib.model` eagerly loads LightGBM (called out in CI). New experimental models and strategies belong in `qlib/contrib/`.
- New features need tests under `tests/` (or an existing relevant subdirectory) and, for user-visible behavior, a docs update under `docs/`.
- Public API changes should preserve `qlib.utils.serial.Serializable` compatibility (handler/dataset/model state is pickled to disk; see `docs/advanced/serial.rst`).
- `examples/**` is excluded from some tooling (`.deepsource.toml`) and its results vary slightly by OS.

## Reference demos

`Reference/` is **not** upstream qlib code — it holds personal demos written in Chinese (`demo_factor_ic`, `demo_xgboost`). Leave them alone unless a task explicitly targets them; the upstream project lives at <https://github.com/microsoft/qlib>.

## Pitfalls

- `qlib.init()` must run before any data access — provider wrappers (`D`, `Cal`, `Inst`, `FeatureD`, …) raise until it does. Calling it again while an experiment is active raises `RecorderInitializationError`, because that would move where results are stored; pass `skip_if_reg=True` when re-initializing is genuinely intended (notebooks, repeated runs in one process).
- `qlib.init()` also registers the experiment manager and installs a `sys.excepthook` / `atexit` hook that finalizes the active recorder; `QSETTINGS` (`qlib/config.py`) additionally reads `QLIB_*` environment variables.
- Data is split by region and interval; `provider_uri` and `region` must agree (CN vs US calendars differ).
- Cache providers can mask stale data after an edit: `qlib.init(default_conf="client")` (the default) leaves `dataset_cache` off, while `server` mode and the built-in `HIGH_FREQ_CONFIG` enable `DiskExpressionCache` / `DiskDatasetCache`. Pass `None` explicitly when a test must read the provider untouched.
- Concrete portfolio strategies live in `qlib/contrib/strategy/` (`TopkDropoutStrategy`, `TWAPStrategy`, …) while `qlib/strategy/` holds only the base classes, and `qlib/contrib/model/__init__.py` swallows `ModuleNotFoundError` per model so an uninstalled optional dependency only prints a skip message — a missing library can silently remove a model from the registry.
- In `qlib/contrib/strategy/signal_strategy.py`, passing both `model` and `dataset` to a strategy is deprecated in favor of `signal`.
- `tests/conftest.py` skips everything under `rl/` on non-Linux platforms.
- `qlib/_version.py` is generated by `setuptools-scm` and is git-ignored — never edit it.
