"use client";

import type { BacktestRunInfo } from "@/store/useChartStore";

type Props = {
  runs: BacktestRunInfo[];
  selectedRunId: string | null;
  onChange: (runId: string) => void;
};

export default function BacktestSelector({
  runs,
  selectedRunId,
  onChange,
}: Props) {
  return (
    <label className="flex flex-col text-sm gap-1">
      <span>Backtest run:</span>
      <select
        value={selectedRunId ?? ""}
        onChange={(e) => onChange(e.target.value)}
        className="border px-2 py-1 rounded"
      >
        <option value="" disabled>
          Select run
        </option>
        {runs.map((run) => {
          const metrics = run.metrics || {};
          const cum = metrics["cum_return"];
          const label = `${run.strategy} | ${
            cum !== undefined && cum !== null ? (cum * 100).toFixed(1) + "%" : "—"
          } | ${run.start ? run.start.slice(0, 10) : "?"}`;
          return (
            <option key={run.run_id} value={run.run_id}>
              {label}
            </option>
          );
        })}
      </select>
    </label>
  );
}
