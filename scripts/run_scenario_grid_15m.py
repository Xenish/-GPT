from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd

from finantradealgo.backtester.scenario_engine import ScenarioConfig, ScenarioEngine
from finantradealgo.features.feature_pipeline_15m import build_feature_pipeline_from_system_config
from finantradealgo.system.config_loader import load_system_config


def load_scenarios(cfg: dict, preset_name: str) -> list[ScenarioConfig]:
    scenario_section = cfg.get("scenario", {}) or {}
    presets = scenario_section.get("presets", {}) or {}
    entries = presets.get(preset_name)
    if not entries:
        raise ValueError(f"Scenario preset '{preset_name}' not found in config.")
    scenarios: list[ScenarioConfig] = []
    for entry in entries:
        scenarios.append(
            ScenarioConfig(
                name=entry["name"],
                strategy_name=entry["strategy_name"],
                strategy_params=entry.get("strategy_params"),
                risk_params=entry.get("risk_params"),
                feature_preset=entry.get("feature_preset"),
                train_mode=entry.get("train_mode"),
            )
        )
    return scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strategy scenario grid on 15m data.")
    parser.add_argument("--preset", default="core_15m", help="Scenario preset name from system config.")
    args = parser.parse_args()

    cfg = load_system_config()
    df_features, meta = build_feature_pipeline_from_system_config(cfg)
    scenarios = load_scenarios(cfg, args.preset)

    engine = ScenarioEngine(cfg)
    result_df = engine.run_scenarios(scenarios, df_features)

    output_dir = Path("outputs") / "backtests"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"scenario_grid_{args.preset}.csv"
    result_df.to_csv(out_path, index=False)

    print(f"[INFO] Scenario grid completed. Saved to {out_path}")
    if not result_df.empty:
        print("\nTop scenarios by Sharpe:")
        print(result_df.sort_values("sharpe", ascending=False).head(3).to_string(index=False))


if __name__ == "__main__":
    main()
