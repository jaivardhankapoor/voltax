"""Run every ```python block of each docs page, in order, in one namespace."""

import importlib
import re
from pathlib import Path

import pytest

DOCS = Path(__file__).parent.parent / "docs"
PAGES = sorted(p for p in DOCS.rglob("*.md") if "```python" in p.read_text())


@pytest.mark.extended
@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(DOCS)))
def test_docs_snippets_run(page):
    if 'vx.va.load("bsim4"' in page.read_text():
        try:  # BSIM4 is not bundled (CC BY-NC); downloaded on first use
            importlib.import_module("voltax.va.fetch").resolve("bsim4")
        except (RuntimeError, FileNotFoundError) as e:
            pytest.skip(f"BSIM4 Verilog-A unavailable: {e}")
    blocks = re.findall(r"```python\n(.*?)```", page.read_text(), re.DOTALL)
    namespace: dict = {}
    for i, code in enumerate(blocks):
        if code.lstrip().startswith("--8<--"):
            continue  # a whole example script, smoke-run by test_examples.py
        try:
            exec(compile(code, f"{page.name}[block {i}]", "exec"), namespace)
        except Exception as e:  # pragma: no cover - reported by pytest
            raise AssertionError(f"block {i} of {page.name} failed:\n{code}") from e
