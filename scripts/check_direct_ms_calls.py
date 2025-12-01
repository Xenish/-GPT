"""
Entry-point enforcement checker.

Task CRITICAL-5: Verify that code uses single entry-points instead of direct calls.

This script checks that:
1. Market structure features are accessed via compute_market_structure_df()
2. Microstructure features are accessed via compute_microstructure_df()
3. No direct imports/calls to internal MS/Micro functions

Violations indicate code that bypasses the single entry-point pattern and may:
- Break output contract guarantees
- Skip input validation
- Lead to inconsistent feature computation

Usage:
    python scripts/check_direct_ms_calls.py
    python scripts/check_direct_ms_calls.py --path finantradealgo/
    python scripts/check_direct_ms_calls.py --strict  # Exit with error code on violations
"""
import argparse
import ast
import logging
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


# Internal functions that should NOT be called directly
MARKET_STRUCTURE_INTERNALS = [
    "detect_swings",
    "detect_impulse",
    "detect_exhaustion",
    "detect_burst",
    "compute_fvg",
    "detect_trend",
]

MICROSTRUCTURE_INTERNALS = [
    "compute_imbalance",
    "detect_liquidity_sweep",
    "compute_chop_index",
    "detect_volatility_regime",
    "detect_burst_move",
    "detect_exhaustion_move",
    "detect_parabolic",
]

# Allowed entry-point functions
ALLOWED_ENTRY_POINTS = [
    "compute_market_structure_df",
    "compute_microstructure_df",
]

# Modules to check
TARGET_MODULES = [
    "finantradealgo/features/",
    "finantradealgo/system/",
    "scripts/",
]

# Modules to exclude (tests, internal implementations)
EXCLUDE_PATTERNS = [
    "test_",
    "__pycache__",
    ".pyc",
    "finantradealgo/market_structure/",  # Internal implementation is allowed to call internals
    "finantradealgo/microstructure/",  # Internal implementation is allowed to call internals
]


class DirectCallChecker(ast.NodeVisitor):
    """AST visitor to detect direct calls to internal functions."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.violations = []
        self.imported_names = {}  # Track imports

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Track 'from X import Y' statements."""
        if node.module:
            # Check if importing from internal modules
            if "market_structure" in node.module or "microstructure" in node.module:
                for alias in node.names:
                    name = alias.name
                    as_name = alias.asname or name

                    # Check if importing internal function
                    if name in MARKET_STRUCTURE_INTERNALS:
                        self.violations.append({
                            "type": "import",
                            "function": name,
                            "module": "market_structure",
                            "line": node.lineno,
                            "message": f"Direct import of internal function '{name}' from {node.module}",
                        })
                        self.imported_names[as_name] = name

                    elif name in MICROSTRUCTURE_INTERNALS:
                        self.violations.append({
                            "type": "import",
                            "function": name,
                            "module": "microstructure",
                            "line": node.lineno,
                            "message": f"Direct import of internal function '{name}' from {node.module}",
                        })
                        self.imported_names[as_name] = name

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """Check function calls."""
        func_name = None

        # Extract function name from Call node
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name:
            # Check if calling an imported internal function
            if func_name in self.imported_names:
                original_name = self.imported_names[func_name]
                module = "market_structure" if original_name in MARKET_STRUCTURE_INTERNALS else "microstructure"
                self.violations.append({
                    "type": "call",
                    "function": original_name,
                    "module": module,
                    "line": node.lineno,
                    "message": f"Direct call to internal function '{func_name}()' (imported from {module})",
                })

            # Also check direct calls to known internal functions
            elif func_name in MARKET_STRUCTURE_INTERNALS:
                self.violations.append({
                    "type": "call",
                    "function": func_name,
                    "module": "market_structure",
                    "line": node.lineno,
                    "message": f"Direct call to internal function '{func_name}()'",
                })

            elif func_name in MICROSTRUCTURE_INTERNALS:
                self.violations.append({
                    "type": "call",
                    "function": func_name,
                    "module": "microstructure",
                    "line": node.lineno,
                    "message": f"Direct call to internal function '{func_name}()'",
                })

        self.generic_visit(node)


def should_check_file(file_path: Path) -> bool:
    """
    Determine if file should be checked for violations.

    Args:
        file_path: Path to Python file

    Returns:
        True if file should be checked
    """
    # Use forward slashes for consistent matching across platforms
    file_str = str(file_path).replace("\\", "/")

    # Skip excluded patterns
    for pattern in EXCLUDE_PATTERNS:
        pattern_normalized = pattern.replace("\\", "/")
        if pattern_normalized in file_str:
            return False

    return True


def check_file(file_path: Path) -> List[Dict]:
    """
    Check a single Python file for direct calls to internal functions.

    Args:
        file_path: Path to Python file

    Returns:
        List of violation dictionaries
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source, filename=str(file_path))
        checker = DirectCallChecker(str(file_path))
        checker.visit(tree)

        return checker.violations

    except SyntaxError as e:
        logger.warning(f"Syntax error in {file_path}: {e}")
        return []
    except Exception as e:
        logger.error(f"Error checking {file_path}: {e}")
        return []


def check_directory(dir_path: Path) -> Dict[str, List[Dict]]:
    """
    Check all Python files in directory for violations.

    Args:
        dir_path: Directory path to check

    Returns:
        Dictionary mapping file paths to lists of violations
    """
    violations_by_file = {}

    # Find all Python files
    for py_file in dir_path.rglob("*.py"):
        if should_check_file(py_file):
            violations = check_file(py_file)
            if violations:
                violations_by_file[str(py_file)] = violations

    return violations_by_file


def generate_report(violations_by_file: Dict[str, List[Dict]]) -> str:
    """
    Generate human-readable report of violations.

    Args:
        violations_by_file: Dictionary mapping file paths to violations

    Returns:
        Report string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("ENTRY-POINT ENFORCEMENT REPORT")
    lines.append("=" * 80)
    lines.append("")

    total_violations = sum(len(v) for v in violations_by_file.values())
    total_files = len(violations_by_file)

    lines.append("SUMMARY")
    lines.append("-" * 80)
    lines.append(f"Total violations: {total_violations}")
    lines.append(f"Files with violations: {total_files}")
    lines.append("")

    if total_violations == 0:
        lines.append("OK - No entry-point violations detected!")
        lines.append("")
        lines.append("All code is correctly using single entry-points:")
        lines.append("  - compute_market_structure_df() for market structure features")
        lines.append("  - compute_microstructure_df() for microstructure features")
    else:
        lines.append("VIOLATIONS DETECTED")
        lines.append("-" * 80)
        lines.append("")

        # Group violations by module
        ms_violations = []
        micro_violations = []

        for file_path, violations in sorted(violations_by_file.items()):
            for v in violations:
                if v["module"] == "market_structure":
                    ms_violations.append((file_path, v))
                else:
                    micro_violations.append((file_path, v))

        # Report Market Structure violations
        if ms_violations:
            lines.append("MARKET STRUCTURE VIOLATIONS:")
            lines.append("")
            for file_path, v in ms_violations:
                lines.append(f"  {file_path}:{v['line']}")
                lines.append(f"    {v['message']}")
                lines.append(f"    -> Use compute_market_structure_df() instead")
                lines.append("")

        # Report Microstructure violations
        if micro_violations:
            lines.append("MICROSTRUCTURE VIOLATIONS:")
            lines.append("")
            for file_path, v in micro_violations:
                lines.append(f"  {file_path}:{v['line']}")
                lines.append(f"    {v['message']}")
                lines.append(f"    -> Use compute_microstructure_df() instead")
                lines.append("")

        lines.append("")
        lines.append("REMEDIATION")
        lines.append("-" * 80)
        lines.append("To fix these violations:")
        lines.append("1. Remove direct imports of internal functions")
        lines.append("2. Replace direct calls with entry-point functions:")
        lines.append("   - compute_market_structure_df(df, cfg)")
        lines.append("   - compute_microstructure_df(df, cfg, trades_df, book_df)")
        lines.append("3. Extract needed columns from returned DataFrames")

    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Check for direct calls to internal MS/Micro functions"
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("."),
        help="Root path to check (default: current directory)"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error code if violations found"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Check target modules
    all_violations = {}

    for module in TARGET_MODULES:
        module_path = args.path / module
        if module_path.exists():
            logger.info(f"Checking {module_path}...")
            violations = check_directory(module_path)
            all_violations.update(violations)

    # Generate report
    report = generate_report(all_violations)
    print(report)

    # Exit with error if strict mode and violations found
    if args.strict and all_violations:
        logger.error("Entry-point violations detected in strict mode")
        exit(1)


if __name__ == "__main__":
    main()
