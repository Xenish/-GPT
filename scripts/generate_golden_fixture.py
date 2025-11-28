import sys
from pathlib import Path

# Ensure the project root is in the Python path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from finantradealgo.features.feature_pipeline import (
    FeaturePipelineConfig,
    build_feature_pipeline,
)

# Define paths for the fixture and the golden output
FIXTURE_INPUT_PATH = ROOT / "tests" / "data" / "microstructure_fixture.csv"
GOLDEN_OUTPUT_PATH = ROOT / "tests" / "data" / "microstructure_fixture_out.parquet"


def generate_golden_file():
    """
    Runs the feature pipeline on a fixed input file and saves the output
    as a "golden file" for regression testing.
    """
    print(f"Loading fixture from: {FIXTURE_INPUT_PATH}")
    if not FIXTURE_INPUT_PATH.exists():
        raise FileNotFoundError(f"Fixture file not found: {FIXTURE_INPUT_PATH}")

    # Configure the pipeline to run all the features we've built
    pipeline_cfg = FeaturePipelineConfig(
        use_microstructure=True,
        # Enable other relevant features for a realistic test
        use_base=True,
        use_ta=True,
        use_candles=True,
        use_osc=True,
        # Disable features that require extra data or are out of scope
        use_htf=False,
        use_market_structure=False,
        use_external=False,
        use_rule_signals=False,
        drop_na=False,  # Important for regression tests to keep all rows
    )

    print("Running feature pipeline to generate golden file...")
    df_out, _ = build_feature_pipeline(
        csv_ohlcv_path=str(FIXTURE_INPUT_PATH),
        pipeline_cfg=pipeline_cfg,
    )

    # Save the output to a Parquet file
    print(f"Saving golden file to: {GOLDEN_OUTPUT_PATH}")
    df_out.to_parquet(GOLDEN_OUTPUT_PATH, index=False)
    print("Golden file generated successfully.")


if __name__ == "__main__":
    generate_golden_file()
