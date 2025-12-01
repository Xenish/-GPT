"""
Generates a golden fixture file for the Market Structure engine.
"""
import sys
from pathlib import Path
import pandas as pd

# Ensure the project root is in the Python path
ROOT = Path(__file__).resolve().parents[1]  # Go up one level to project root
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from finantradealgo.features.market_structure_features import add_market_structure_features
from finantradealgo.market_structure.config import MarketStructureConfig

# --- Configuration ---
SOURCE_DATA_PATH = ROOT / "data" / "ohlcv" / "BTCUSDT_15m.csv"
FIXTURE_INPUT_PATH = ROOT / "tests" / "data" / "market_structure_fixture.csv"
GOLDEN_OUTPUT_PATH = ROOT / "tests" / "golden" / "market_structure_fixture_out.parquet"
NUM_BARS = 500  # Use a fixed number of bars for the fixture


def generate_ms_golden_file():
    """
    Takes a slice of a source data file, saves it as a fixture, runs the
    Market Structure engine on it, and saves the output as a "golden file"
    for regression testing.
    """
    print(f"Loading source data from: {SOURCE_DATA_PATH}")
    if not SOURCE_DATA_PATH.exists():
        raise FileNotFoundError(f"Source data not found: {SOURCE_DATA_PATH}")

    df_source = pd.read_csv(SOURCE_DATA_PATH, parse_dates=["timestamp"])
    
    # 1. Create and save the deterministic input fixture
    df_fixture = df_source.tail(NUM_BARS).copy()
    print(f"Saving {NUM_BARS} bars to fixture input: {FIXTURE_INPUT_PATH}")
    FIXTURE_INPUT_PATH.parent.mkdir(exist_ok=True)
    df_fixture.to_csv(FIXTURE_INPUT_PATH, index=False)

    # 2. Run the Market Structure engine on the fixture
    print("Running Market Structure engine to generate golden file...")
    cfg = MarketStructureConfig()  # Use default config for consistency
    df_with_features = add_market_structure_features(df_fixture, cfg)

    # 3. Select only the market structure columns for the golden file
    ms_cols = [col for col in df_with_features.columns if col.startswith("ms_")]
    df_golden = df_with_features[ms_cols]

    # 4. Save the golden output file
    print(f"Saving golden file to: {GOLDEN_OUTPUT_PATH}")
    GOLDEN_OUTPUT_PATH.parent.mkdir(exist_ok=True)
    df_golden.to_parquet(GOLDEN_OUTPUT_PATH, index=False)

    print("Market Structure golden fixture generated successfully.")


if __name__ == "__main__":
    generate_ms_golden_file()
