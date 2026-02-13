---
name: bc-repo-standards
description: Standardize Bot Cripto repository structure, linting, typing, and baseline project conventions. Use when bootstrapping or enforcing quality gates and coding standards.
---

# bc-repo-standards

1. Verify expected folders exist: `src/`, `tests/`, `scripts/`, `docker/`, `k8s/`, `docs/`.
2. Ensure `pyproject.toml` defines dependencies, ruff, black, mypy, pytest.
3. Keep `README.md` and `.env.example` aligned with runtime config.
4. Enforce quality commands before closing work:
   - `ruff check src tests`
   - `black --check src tests`
   - `mypy src/bot_cripto`
   - `pytest tests -v`
5. Avoid hardcoded secrets and ensure `.gitignore` excludes `.env`.
