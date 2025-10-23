# Repository Guidelines

## Project Structure & Module Organization

- `src/`: Main code. Key modules: `search/` (engines + extractors), `prompts/`, `evaluate/`, `utils/`, `browser_cookies/`.
- `src/run.py`: Entry point for report generation and deep web exploration.
- `data/`, `outputs/`, `logs/`, `cache/`: Input, artifacts, run logs, and cached search/url content.
- `.env`: Local secrets and runtime config (not committed).

## Build, Test, and Development Commands

- Environment: `conda create -n webthinker python=3.11 && conda activate webthinker`
- Install: `pip install -r requirements.txt`
- Run (single question):
  - `python src/run.py --single_question "Your question" --search_engine "serper|cookie_google" --api_base_url "..." --model_name "QwQ-32B"`
- Evaluate reports:
  - `python src/evaluate/evaluate_report.py --api-base-url "..." --api-key "..." --models "deepseek/deepseek-r1" --model-to-test-dir "outputs/your_model"`

## Coding Style & Naming Conventions

- Python, 4-space indent, PEP 8 naming: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`.
- Use `loguru` for logging; avoid `print`.
- Type hints welcome but not mandatory where they impede iteration.
- Paths are relative to repo root; prefer `pathlib.Path`.

## Testing Guidelines

- No formal test harness yet. Validate changes with small end-to-end runs using `src/run_backend.py` and targeted utilities (e.g., `src/utils/test_webparser.py`).
- Add lightweight tests next to the code under `src/**/test_*.py` and keep them deterministic (mock APIs when possible).
- When adding tests, ensure they run without network by default or guard with flags.

## Commit & Pull Request Guidelines

- Commits: concise, present tense, scope first when relevant (e.g., `search: normalize marker handling`). Multi-line details welcome.
- Include rationale when changing model prompts, token logic, or search flows.
- PRs must include: purpose, high-level changes, how to run, config used (`.env` keys), and sample output/logs. Link related issues. Add screenshots for report diffs when helpful.

## Security & Configuration Tips

- Never commit `.env`, cookies, or API keys. Example env vars: `WEBTHINKER_USER_AGENT`, `CHROME_PROFILE`, service API keys.
- Prefer cookie-backed search only with explicit consent; redact personal paths in logs.
- Large runs write to `outputs/` and `logs/`; clean sensitive artifacts before sharing.

## Architecture Overview

- Core flow: Reasoning LLM plans → searches (`search_engine_utils`) → fetches pages → extracts snippets → auxiliary LLM drafts/edits → report saved in `outputs/` with trace in `logs/` and caches in `cache/`.
