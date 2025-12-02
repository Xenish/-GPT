"""
Check that strategies don't import forbidden modules.

Strategies should only generate signals, not execute orders or access live trading.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import List, Set


# Forbidden imports for strategies
FORBIDDEN_MODULES = {
    "finantradealgo.execution",
    "finantradealgo.live_trading",
    "finantradealgo.live",
    "finantradealgo.api",
}


class ImportVisitor(ast.NodeVisitor):
    """AST visitor to extract import statements."""

    def __init__(self):
        self.imports: Set[str] = set()
        self.from_imports: Set[str] = set()

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.add(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            self.from_imports.add(node.module)


def check_file(file_path: Path) -> List[str]:
    """
    Check a single Python file for forbidden imports.

    Returns:
        List of error messages (empty if no violations)
    """
    errors = []

    try:
        with file_path.open("r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(file_path))

        visitor = ImportVisitor()
        visitor.visit(tree)

        # Check forbidden modules
        for forbidden in FORBIDDEN_MODULES:
            for import_name in visitor.imports | visitor.from_imports:
                if import_name.startswith(forbidden):
                    errors.append(
                        f"{file_path}: Forbidden import '{import_name}' found"
                    )

    except SyntaxError as e:
        errors.append(f"{file_path}: Syntax error - {e}")

    return errors


def check_strategies() -> int:
    """
    Check all files in finantradealgo/strategies/.

    Returns:
        Exit code (0 = success, 1 = violations found)
    """
    strategies_dir = Path("finantradealgo/strategies")

    if not strategies_dir.exists():
        print(f"[OK] Strategies directory not found")
        return 0

    python_files = [
        f for f in strategies_dir.glob("*.py")
        if not f.name.startswith("_")
    ]

    if not python_files:
        print(f"[OK] No strategy files found")
        return 0

    print(f"Checking {len(python_files)} strategy files...")
    print()

    all_errors = []
    for file_path in python_files:
        errors = check_file(file_path)
        all_errors.extend(errors)

    if all_errors:
        print("[FAIL] VIOLATIONS FOUND:")
        print()
        for error in all_errors:
            print(f"  {error}")
        print()
        print("Strategies must NOT import execution/live trading modules!")
        return 1
    else:
        print("[PASS] All checks passed!")
        print("  Strategies are properly isolated from execution layer")
        return 0


if __name__ == "__main__":
    sys.exit(check_strategies())
