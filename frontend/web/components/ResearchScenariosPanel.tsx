/**
 * Research Scenarios Panel
 *
 * Run and view scenario analyses from the Research Service.
 */

"use client";

import { useState } from "react";
import { useResearchStore } from "@/store/useResearchStore";
import type { ScenarioRequest } from "@/store/useResearchStore";

export default function ResearchScenariosPanel() {
  const {
    scenarioResults,
    isRunningScenarios,
    scenarioError,
    runScenarios,
    clearErrors,
  } = useResearchStore();

  const [baseSymbol, setBaseSymbol] = useState("AIAUSDT");
  const [baseTimeframe, setBaseTimeframe] = useState("15m");
  const [scenarios, setScenarios] = useState<ScenarioRequest[]>([
    {
      label: "Rule baseline",
      strategy: "rule",
      params: {},
    },
  ]);
  const [selectedRowIndex, setSelectedRowIndex] = useState<number | null>(null);

  const handleAddScenario = () => {
    setScenarios([
      ...scenarios,
      {
        label: `Scenario ${scenarios.length + 1}`,
        strategy: "rule",
        params: {},
      },
    ]);
  };

  const handleRemoveScenario = (index: number) => {
    setScenarios(scenarios.filter((_, i) => i !== index));
  };

  const handleUpdateScenario = (
    index: number,
    field: keyof ScenarioRequest,
    value: any
  ) => {
    const updated = [...scenarios];
    updated[index] = { ...updated[index], [field]: value };
    setScenarios(updated);
  };

  const handleRunScenarios = async () => {
    clearErrors();
    await runScenarios({
      base_symbol: baseSymbol,
      base_timeframe: baseTimeframe,
      scenarios,
    });
  };

  const formatMetricValue = (value?: number | null, precision = 2) => {
    if (value === null || value === undefined || Number.isNaN(value)) {
      return "-";
    }
    return Number(value).toFixed(precision);
  };

  const selectedRow = selectedRowIndex !== null ? scenarioResults[selectedRowIndex] : null;

  return (
    <div className="space-y-4">
      {/* Header */}
      <div>
        <h2 className="text-lg font-semibold">Scenario Analysis</h2>
        <p className="text-sm text-gray-600">
          Compare multiple strategy configurations side-by-side.
        </p>
      </div>

      {/* Error Display */}
      {scenarioError && (
        <div className="rounded bg-red-50 border border-red-200 p-3 text-sm text-red-700">
          {scenarioError}
        </div>
      )}

      {/* Configuration */}
      <div className="rounded bg-white shadow p-4 space-y-3">
        <h3 className="font-semibold text-sm">Configuration</h3>

        <div className="flex gap-3 text-sm">
          <label className="flex items-center gap-2">
            <span className="text-gray-600">Base Symbol:</span>
            <input
              type="text"
              className="border px-2 py-1 rounded w-32"
              value={baseSymbol}
              onChange={(e) => setBaseSymbol(e.target.value)}
            />
          </label>
          <label className="flex items-center gap-2">
            <span className="text-gray-600">Base Timeframe:</span>
            <select
              className="border px-2 py-1 rounded"
              value={baseTimeframe}
              onChange={(e) => setBaseTimeframe(e.target.value)}
            >
              <option value="5m">5m</option>
              <option value="15m">15m</option>
              <option value="1h">1h</option>
              <option value="4h">4h</option>
            </select>
          </label>
        </div>

        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="font-medium text-sm">Scenarios</label>
            <button
              onClick={handleAddScenario}
              className="px-2 py-1 border rounded text-xs bg-slate-100 hover:bg-slate-200"
            >
              + Add Scenario
            </button>
          </div>

          <div className="space-y-2">
            {scenarios.map((scenario, index) => (
              <div
                key={index}
                className="flex gap-2 items-start p-2 border rounded bg-slate-50 text-sm"
              >
                <div className="flex-1 grid grid-cols-3 gap-2">
                  <input
                    type="text"
                    className="border px-2 py-1 rounded"
                    placeholder="Label"
                    value={scenario.label || ""}
                    onChange={(e) =>
                      handleUpdateScenario(index, "label", e.target.value)
                    }
                  />
                  <select
                    className="border px-2 py-1 rounded"
                    value={scenario.strategy}
                    onChange={(e) =>
                      handleUpdateScenario(index, "strategy", e.target.value)
                    }
                  >
                    <option value="rule">rule</option>
                    <option value="trend_continuation">trend_continuation</option>
                    <option value="sweep_reversal">sweep_reversal</option>
                    <option value="volatility_breakout">volatility_breakout</option>
                  </select>
                  <input
                    type="text"
                    className="border px-2 py-1 rounded font-mono text-xs"
                    placeholder='Params (JSON, e.g. {"key": "value"})'
                    value={JSON.stringify(scenario.params || {})}
                    onChange={(e) => {
                      try {
                        const parsed = JSON.parse(e.target.value || "{}");
                        handleUpdateScenario(index, "params", parsed);
                      } catch {
                        // Invalid JSON, ignore
                      }
                    }}
                  />
                </div>
                <button
                  onClick={() => handleRemoveScenario(index)}
                  className="px-2 py-1 text-red-600 hover:bg-red-50 rounded text-xs"
                >
                  Remove
                </button>
              </div>
            ))}
          </div>
        </div>

        <div className="flex justify-end">
          <button
            onClick={handleRunScenarios}
            disabled={isRunningScenarios || scenarios.length === 0}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-gray-400"
          >
            {isRunningScenarios ? "Running..." : "Run Scenarios"}
          </button>
        </div>
      </div>

      {/* Results */}
      {scenarioResults.length > 0 && (
        <div className="space-y-3">
          <h3 className="font-semibold">Results ({scenarioResults.length})</h3>

          <div className="rounded bg-white shadow overflow-auto">
            <table className="min-w-full text-xs">
              <thead>
                <tr className="text-left border-b bg-slate-100">
                  <th className="px-3 py-2">Label</th>
                  <th className="px-3 py-2">Strategy</th>
                  <th className="px-3 py-2">Symbol</th>
                  <th className="px-3 py-2">TF</th>
                  <th className="px-3 py-2 text-right">Cum Return</th>
                  <th className="px-3 py-2 text-right">Sharpe</th>
                  <th className="px-3 py-2 text-right">Max DD</th>
                  <th className="px-3 py-2 text-right">Trades</th>
                  <th className="px-3 py-2 text-right">Win Rate</th>
                </tr>
              </thead>
              <tbody>
                {scenarioResults.map((row, index) => {
                  const isSelected = index === selectedRowIndex;
                  return (
                    <tr
                      key={index}
                      className={`border-b last:border-0 cursor-pointer ${
                        isSelected ? "bg-blue-50" : "hover:bg-slate-50"
                      }`}
                      onClick={() => setSelectedRowIndex(index)}
                    >
                      <td className="px-3 py-2">{row.label}</td>
                      <td className="px-3 py-2">{row.strategy}</td>
                      <td className="px-3 py-2">{row.symbol}</td>
                      <td className="px-3 py-2">{row.timeframe}</td>
                      <td className="px-3 py-2 text-right">
                        {formatMetricValue(row.cum_return)}
                      </td>
                      <td className="px-3 py-2 text-right">
                        {formatMetricValue(row.sharpe)}
                      </td>
                      <td className="px-3 py-2 text-right">
                        {formatMetricValue(row.max_dd)}
                      </td>
                      <td className="px-3 py-2 text-right">
                        {row.trade_count ?? "-"}
                      </td>
                      <td className="px-3 py-2 text-right">
                        {formatMetricValue(row.win_rate ? row.win_rate * 100 : null, 1)}
                        {row.win_rate !== null && row.win_rate !== undefined ? "%" : ""}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Selected Row Details */}
          {selectedRow && (
            <div className="rounded bg-white shadow p-4">
              <h4 className="font-semibold text-sm mb-3">
                Details: {selectedRow.label}
              </h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                <div>
                  <div className="text-xs text-gray-600">Strategy</div>
                  <div className="font-medium">{selectedRow.strategy}</div>
                </div>
                <div>
                  <div className="text-xs text-gray-600">Symbol/TF</div>
                  <div className="font-medium">
                    {selectedRow.symbol}/{selectedRow.timeframe}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-gray-600">Cum Return</div>
                  <div className="font-medium">
                    {formatMetricValue(selectedRow.cum_return)}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-gray-600">Sharpe</div>
                  <div className="font-medium">
                    {formatMetricValue(selectedRow.sharpe)}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-gray-600">Max DD</div>
                  <div className="font-medium">
                    {formatMetricValue(selectedRow.max_dd)}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-gray-600">Trades</div>
                  <div className="font-medium">{selectedRow.trade_count ?? "-"}</div>
                </div>
                <div>
                  <div className="text-xs text-gray-600">Win Rate</div>
                  <div className="font-medium">
                    {formatMetricValue(
                      selectedRow.win_rate ? selectedRow.win_rate * 100 : null,
                      1
                    )}
                    {selectedRow.win_rate !== null && selectedRow.win_rate !== undefined
                      ? "%"
                      : ""}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
