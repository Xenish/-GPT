import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure the project root is in the Python path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from finantradealgo.features.feature_pipeline import (
    FeaturePipelineConfig,
    build_feature_pipeline,
)

# Define paths for the fixture and the golden file
FIXTURE_INPUT_PATH = ROOT / "tests" / "data" / "microstructure_fixture.csv"
GOLDEN_OUTPUT_PATH = ROOT / "tests" / "data" / "microstructure_fixture_out.parquet"


@pytest.mark.skipif(
    not GOLDEN_OUTPUT_PATH.exists(),
    reason="Golden fixture file not found. Please run 'scripts/generate_golden_fixture.py' to create it.",
)
def test_microstructure_regression():
    """
    Compares the current pipeline output against a pre-generated "golden"
    file to detect any unintended changes (regressions) in the logic.
    """
    # 1. Configure and run the pipeline on the fixture input
    pipeline_cfg = FeaturePipelineConfig(
        use_microstructure=True,
        use_base=True,
        use_ta=True,
        use_candles=True,
        use_osc=True,
        use_htf=False,
        use_market_structure=False,
        use_external=False,
        use_rule_signals=False,
        drop_na=False,
    )
    df_current_output, _ = build_feature_pipeline(
        csv_ohlcv_path=str(FIXTURE_INPUT_PATH),
        pipeline_cfg=pipeline_cfg,
    )

    # 2. Load the golden file
    df_golden = pd.read_parquet(GOLDEN_OUTPUT_PATH)

    # 3. Compare the two DataFrames
    # Use pandas' testing utility, which gives detailed diffs on failure
    pd.testing.assert_frame_equal(df_current_output, df_golden)
