"""Validate notebook JSON and Python code-cell syntax without executing cells."""

from __future__ import annotations

import ast
import json
from pathlib import Path


def main() -> None:
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "llm_finetuning_benchmark.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    for index, cell in enumerate(code_cells):
        ast.parse("".join(cell["source"]), filename=f"notebook-code-cell-{index}")
    print(f"Notebook JSON valid; {len(notebook['cells'])} cells; {len(code_cells)} code cells parse.")


if __name__ == "__main__":
    main()
