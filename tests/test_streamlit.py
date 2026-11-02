"""
Smoke tests for the Streamlit frontend.
Validates that the app can be imported and core data-loading functions work.
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestStreamlitApp:
    def test_import(self):
        """Streamlit app module should be importable without crashing."""
        # We only check that the file is valid Python
        import py_compile
        result = py_compile.compile(
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "streamlit_app.py"),
            doraise=True,
        )
        assert result is not None

    def test_data_files_exist(self):
        """Required data files for the dashboard must exist."""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        required = [
            "ferry_tickets_cleaned.csv",
            "all_model_results.csv",
            "uncertainty_results.csv",
            "uncertainty_summary.json",
            os.path.join("eda_plots", "summary_statistics.json"),
        ]
        for f in required:
            path = os.path.join(root, f)
            assert os.path.exists(path), f"Missing required file: {f}"

    def test_csv_loadable(self):
        """Core CSV should be parseable by pandas."""
        import pandas as pd
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        df = pd.read_csv(os.path.join(root, "ferry_tickets_cleaned.csv"), nrows=5)
        assert "Timestamp" in df.columns or "timestamp" in df.columns.str.lower()
