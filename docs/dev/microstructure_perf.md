# Microstructure Feature Performance

This document records the performance benchmarks for the microstructure feature computation engine.

The benchmarks are run using the `scripts/bench_microstructure.py` script.

## Methodology

The script loads a given OHLCV dataset (e.g., 1 year of 15m data), runs the `compute_microstructure_df` function, and measures the total execution time. This provides an estimate of the performance cost of adding these features to the pipeline.

The key metric is **milliseconds per 1k bars**, which shows how efficiently the vectorized functions are running.

## Benchmark Results

The results below should be updated whenever significant changes are made to the microstructure feature calculations.

To generate new results, run:
```bash
python scripts/bench_microstructure.py
```

---

### Machine Specs

*   **CPU:** [Please fill in your CPU model]
*   **RAM:** [Please fill in your RAM amount]
*   **Python Version:** [e.g., 3.10.4]

---

### Run: `BTCUSDT_15m.csv`

*   **Date of Run:** `YYYY-MM-DD`
*   **Total Bars Processed:** `[Fill in from script output]`
*   **Total Computation Time:** `[Fill in from script output]` seconds
*   **Time per 1k Bars:** `[Fill in from script output]` ms

---

*(You can add more sections for different datasets like AIAUSDT if needed.)*
