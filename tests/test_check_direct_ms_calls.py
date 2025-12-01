"""
Tests for entry-point enforcement checker.

Task CRITICAL-5: Validate entry-point enforcement detection.
"""
import ast
import tempfile
from pathlib import Path

import pytest

from scripts.check_direct_ms_calls import (
    DirectCallChecker,
    check_file,
    should_check_file,
    generate_report,
)


class TestDirectCallChecker:
    """Test AST-based direct call detection."""

    def test_detects_direct_import(self):
        """Test detection of direct imports from internal modules."""
        code = """
from finantradealgo.market_structure.swing import detect_swings

def my_function():
    pass
"""
        tree = ast.parse(code)
        checker = DirectCallChecker("test.py")
        checker.visit(tree)

        assert len(checker.violations) == 1
        assert checker.violations[0]["type"] == "import"
        assert checker.violations[0]["function"] == "detect_swings"
        assert checker.violations[0]["module"] == "market_structure"

    def test_detects_direct_call(self):
        """Test detection of direct calls to internal functions."""
        code = """
from finantradealgo.market_structure.swing import detect_swings

def my_function(df):
    swings = detect_swings(df)
    return swings
"""
        tree = ast.parse(code)
        checker = DirectCallChecker("test.py")
        checker.visit(tree)

        # Should detect both import and call
        assert len(checker.violations) >= 2
        assert any(v["type"] == "import" for v in checker.violations)
        assert any(v["type"] == "call" for v in checker.violations)

    def test_detects_microstructure_import(self):
        """Test detection of microstructure internal imports."""
        code = """
from finantradealgo.microstructure.imbalance import compute_imbalance

def my_function():
    pass
"""
        tree = ast.parse(code)
        checker = DirectCallChecker("test.py")
        checker.visit(tree)

        assert len(checker.violations) == 1
        assert checker.violations[0]["module"] == "microstructure"
        assert checker.violations[0]["function"] == "compute_imbalance"

    def test_allows_entry_point_import(self):
        """Test that entry-point imports are allowed."""
        code = """
from finantradealgo.market_structure.engine import compute_market_structure_df
from finantradealgo.microstructure.microstructure_engine import compute_microstructure_df

def my_function(df):
    ms_df = compute_market_structure_df(df)
    micro_df = compute_microstructure_df(df)
    return ms_df, micro_df
"""
        tree = ast.parse(code)
        checker = DirectCallChecker("test.py")
        checker.visit(tree)

        # Should not detect any violations
        assert len(checker.violations) == 0

    def test_detects_aliased_import(self):
        """Test detection of imports with aliases."""
        code = """
from finantradealgo.market_structure.swing import detect_swings as ds

def my_function(df):
    swings = ds(df)
    return swings
"""
        tree = ast.parse(code)
        checker = DirectCallChecker("test.py")
        checker.visit(tree)

        # Should detect both import and call
        assert len(checker.violations) >= 2

    def test_allows_other_imports(self):
        """Test that non-internal imports are allowed."""
        code = """
import pandas as pd
from finantradealgo.features.basic_features import compute_returns

def my_function(df):
    returns = compute_returns(df)
    return returns
"""
        tree = ast.parse(code)
        checker = DirectCallChecker("test.py")
        checker.visit(tree)

        # Should not detect any violations
        assert len(checker.violations) == 0


class TestShouldCheckFile:
    """Test file filtering logic."""

    def test_excludes_test_files(self):
        """Test that test files are excluded."""
        assert not should_check_file(Path("tests/test_something.py"))
        assert not should_check_file(Path("test_module.py"))

    def test_excludes_internal_modules(self):
        """Test that internal implementation modules are excluded."""
        assert not should_check_file(Path("finantradealgo/market_structure/swing.py"))
        assert not should_check_file(Path("finantradealgo/microstructure/imbalance.py"))

    def test_includes_feature_modules(self):
        """Test that feature modules are included."""
        assert should_check_file(Path("finantradealgo/features/feature_pipeline.py"))
        assert should_check_file(Path("scripts/generate_features.py"))

    def test_excludes_pycache(self):
        """Test that __pycache__ is excluded."""
        assert not should_check_file(Path("finantradealgo/__pycache__/module.pyc"))


class TestCheckFile:
    """Test file checking functionality."""

    def test_checks_file_with_violation(self):
        """Test checking a file with violations."""
        code = """
from finantradealgo.market_structure.swing import detect_swings

def my_function(df):
    return detect_swings(df)
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            temp_path = Path(f.name)

        try:
            violations = check_file(temp_path)
            assert len(violations) >= 1
            assert any(v["function"] == "detect_swings" for v in violations)
        finally:
            temp_path.unlink()

    def test_checks_file_without_violation(self):
        """Test checking a file without violations."""
        code = """
from finantradealgo.market_structure.engine import compute_market_structure_df

def my_function(df):
    return compute_market_structure_df(df)
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            temp_path = Path(f.name)

        try:
            violations = check_file(temp_path)
            assert len(violations) == 0
        finally:
            temp_path.unlink()

    def test_handles_syntax_error(self):
        """Test handling of files with syntax errors."""
        code = "def invalid syntax here"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            temp_path = Path(f.name)

        try:
            violations = check_file(temp_path)
            # Should return empty list, not crash
            assert violations == []
        finally:
            temp_path.unlink()


class TestGenerateReport:
    """Test report generation."""

    def test_generates_report_no_violations(self):
        """Test report generation with no violations."""
        violations_by_file = {}
        report = generate_report(violations_by_file)

        assert "Total violations: 0" in report
        assert "OK - No entry-point violations detected!" in report

    def test_generates_report_with_violations(self):
        """Test report generation with violations."""
        violations_by_file = {
            "test.py": [
                {
                    "type": "import",
                    "function": "detect_swings",
                    "module": "market_structure",
                    "line": 1,
                    "message": "Direct import of internal function 'detect_swings'",
                }
            ]
        }
        report = generate_report(violations_by_file)

        assert "Total violations: 1" in report
        assert "Files with violations: 1" in report
        assert "VIOLATIONS DETECTED" in report
        assert "detect_swings" in report
        assert "compute_market_structure_df()" in report

    def test_report_separates_modules(self):
        """Test that report separates MS and Micro violations."""
        violations_by_file = {
            "test1.py": [
                {
                    "type": "import",
                    "function": "detect_swings",
                    "module": "market_structure",
                    "line": 1,
                    "message": "Direct import of internal function 'detect_swings'",
                }
            ],
            "test2.py": [
                {
                    "type": "import",
                    "function": "compute_imbalance",
                    "module": "microstructure",
                    "line": 1,
                    "message": "Direct import of internal function 'compute_imbalance'",
                }
            ]
        }
        report = generate_report(violations_by_file)

        assert "MARKET STRUCTURE VIOLATIONS:" in report
        assert "MICROSTRUCTURE VIOLATIONS:" in report
        assert "detect_swings" in report
        assert "compute_imbalance" in report

    def test_report_includes_remediation(self):
        """Test that report includes remediation guidance."""
        violations_by_file = {
            "test.py": [
                {
                    "type": "import",
                    "function": "detect_swings",
                    "module": "market_structure",
                    "line": 1,
                    "message": "Direct import",
                }
            ]
        }
        report = generate_report(violations_by_file)

        assert "REMEDIATION" in report
        assert "Remove direct imports" in report
        assert "Replace direct calls with entry-point functions" in report
