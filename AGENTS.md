# Repository Guidelines

## Project Structure & Module Organization
Source code lives in `source/g1_custom_tutorial/g1_custom_tutorial`, which is the Python package loaded by Isaac Lab. Task definitions are split under `tasks/direct` and `tasks/manager_based`, while UI helpers sit in `ui_extension_example.py`. Configuration presets (YAML, JSON, USD) belong in `source/g1_custom_tutorial/config`, and user-facing docs or diagrams go in `source/g1_custom_tutorial/docs`. Automation scripts (training, eval, utilities) live in `scripts/`, grouped by RL backend (`rl_games`, `rsl_rl`) plus helper agents (`list_envs.py`, `random_agent.py`, `zero_agent.py`).

## Build, Test, and Development Commands
- `python -m pip install -e source/g1_custom_tutorial` — install the package in editable mode using the Isaac Lab interpreter.
- `python scripts/list_envs.py` — confirm that your tasks register; update the search token if you rename Template tasks.
- `python scripts/rsl_rl/train.py --task=Template-Direct-v0 --num_envs=16` — launch RSL-RL training with explicit task selection.
- `python scripts/rl_games/train.py --task=Template-Manager-v0 --max_iter=5` — quick RL-Games sanity run.
- `python scripts/zero_agent.py --task=Template-Direct-v0` — deterministic smoke test; use `random_agent.py` for stochastic stress.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indents, type hints on public APIs, and descriptive docstrings for tasks and policies. Name tasks `Template-<Mode>-v#` so discovery scripts keep working. Keep modules small: task logic in `tasks/<mode>`, shared utilities in package-level helpers. Run `pre-commit run --all-files` before pushing; configure hooks locally via `.pre-commit-config.yaml` if you need extras.

## Testing Guidelines
Each new task should ship with a smoke scenario by extending `scripts/zero_agent.py` or adding a deterministic policy under `scripts/`. Prefer pytest-based unit tests under `source/g1_custom_tutorial/tests` (create the folder if missing) for math/util pieces, and keep names `test_<feature>.py`. Aim for coverage of reward shaping, resets, and asset loading; add regression cases whenever you touch config schemas.

## Commit & Pull Request Guidelines
History currently uses short, imperative subjects (e.g., "Initial commit"), so continue with one-line commands under 72 chars followed by context in the body as needed. Reference issue IDs in the footer (`Refs #123`) and mention affected tasks/configs explicitly. PRs must include: purpose, testing evidence (command output or screenshots of Omniverse UI), config or asset updates, and any migration steps for downstream users. Tag reviewers responsible for the touched subsystem (tasks, UI, or training scripts) to keep reviews focused.

## Omniverse & Configuration Tips
Keep `.vscode/.python.env` up to date so Pylance resolves Isaac modules; rerun the `setup_python_env` VS Code task after every Isaac upgrade. When adding new USD assets, store relative paths under `config` to avoid broken extension loads.
