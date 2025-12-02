"""
Check that research service doesn't import live trading modules.

This script validates that the research service maintains isolation from
live trading components, preventing accidental real order execution.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import List, Set


# Forbidden imports for research service
FORBIDDEN_MODULES = {
    "finantradealgo.execution.exchange_client",
    "finantradealgo.execution.binance_client",
    "finantradealgo.live_trading",
    "finantradealgo.live",
}

FORBIDDEN_PATTERNS = [
    "BinanceFuturesClient",
    "LiveEngine",
    "OrderExecutor",
    "LiveTradingEngine",
]


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
            if forbidden in visitor.imports or forbidden in visitor.from_imports:
                errors.append(
                    f"{file_path}: Forbidden import '{forbidden}' found"
                )

        # Check import patterns
        for pattern in FORBIDDEN_PATTERNS:
            if pattern in visitor.imports:
                errors.append(
                    f"{file_path}: Forbidden pattern '{pattern}' in imports"
                )

    except SyntaxError as e:
        errors.append(f"{file_path}: Syntax error - {e}")

    return errors


def check_research_service() -> int:
    """
    Check all files in services/research_service/.

    Returns:
        Exit code (0 = success, 1 = violations found)
    """
    research_service_dir = Path("services/research_service")

    if not research_service_dir.exists():
        print(f"[OK] Research service directory not found (OK if not yet created)")
        return 0

    python_files = list(research_service_dir.glob("**/*.py"))

    if not python_files:
        print(f"[OK] No Python files found in research service")
        return 0

    print(f"Checking {len(python_files)} files in services/research_service/...")
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
        print("Research service must NOT import live trading modules!")
        return 1
    else:
        print("[PASS] All checks passed!")
        print("  Research service is properly isolated from live trading")
        return 0


if __name__ == "__main__":
    sys.exit(check_research_service())
