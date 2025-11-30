"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";
import { fetchBacktests, fetchChart, fetchSummary, fetchTrades } from "@/lib/api";
import { useChartStore } from "@/store/useChartStore";
import StrategySelector from "@/components/StrategySelector";
import BacktestSelector from "@/components/BacktestSelector";
import TradeTable from "@/components/TradeTable";

const Chart = dynamic(() => import("@/components/Chart"), { ssr: false });

export default function HomePage() {
  const {
    symbol,
    timeframe,
    strategies,
    availableSymbols,
    availableTimeframes,
    availableStrategies,
    mlTargets,
    selectedStrategy,
    backtests,
    selectedRunId,
    trades,
    selectedTradeId,
    bars,
    meta,
    summary,
    overlays,
    loading,
    error,
    setSymbol,
    setTimeframe,
    setSelectedStrategy,
    setBacktests,
    setSelectedRunId,
    setTrades,
    setSelectedTradeId,
    setBars,
    setMeta,
    setSummary,
    setOverlays,
    setLoading,
    setError,
    runBacktest,
    isRunningBacktest,
    lastError,
    initMeta,
    ruleParams,
    setRuleParams,
    lastRunMetrics,
    scenarioResults,
    selectedScenarioId,
    isLoadingScenarios,
    scenarioError,
    fetchScenarioResults,
    setSelectedScenarioId,
    mlModels,
    selectedModelId,
    featureImportance,
    isLoadingMlModels,
    isLoadingFeatureImportance,
    mlError,
    fetchMlModels,
    selectModel,
    portfolioRuns,
    selectedPortfolioRunId,
    portfolioEquity,
    isLoadingPortfolioEquity,
    portfolioError,
    fetchPortfolioRuns,
    fetchPortfolioEquity,
    liveStatus,
    liveError,
    isSendingLiveCommand,
    fetchLiveStatus,
    sendLiveCommand,
  } = useChartStore();

  const [activeTab, setActiveTab] = useState<"single" | "portfolio" | "live" | "lab" | "ml">("single");

  // ML Lab: separate symbol/timeframe selection filtered by ml_targets
  const [mlSymbol, setMlSymbol] = useState<string>("");
  const [mlTimeframe, setMlTimeframe] = useState<string>("");

  const filteredRuns = useMemo(
    () => backtests.filter((run) => run.strategy === selectedStrategy),
    [backtests, selectedStrategy]
  );
  const selectedScenario = useMemo(
    () =>
      scenarioResults.find((row) => row.scenario_id === selectedScenarioId) ?? null,
    [scenarioResults, selectedScenarioId]
  );
  const selectedMlModel = useMemo(
    () => mlModels.find((m) => m.model_id === selectedModelId) ?? null,
    [mlModels, selectedModelId]
  );

  const formatKeyParams = (params: Record<string, any>, limit = 3) => {
    const entries = Object.entries(params ?? {});
    if (!entries.length) {
      return "-";
    }
    return entries
      .slice(0, limit)
      .map(([key, value]) => `${key}=${value}`)
      .join(", ");
  };

  const formatMetricValue = (value?: number | null, precision = 2) => {
    if (value === null || value === undefined || Number.isNaN(value)) {
      return "-";
    }
    return Number(value).toFixed(precision);
  };

  useEffect(() => {
    initMeta();
  }, [initMeta]);

  useEffect(() => {
    if (!symbol || !timeframe) return;
    async function loadBacktests() {
      try {
        const runs = await fetchBacktests(symbol, timeframe);
        setBacktests(runs);
      } catch (err) {
        console.error(err);
        setBacktests([]);
      }
    }
    loadBacktests();
  }, [symbol, timeframe, setBacktests]);

  useEffect(() => {
    if (!filteredRuns.length) {
      setSelectedRunId(null);
      setTrades([]);
      return;
    }
    if (!selectedRunId || !filteredRuns.some((run) => run.run_id === selectedRunId)) {
      setSelectedRunId(filteredRuns[0].run_id);
    }
  }, [filteredRuns, selectedRunId, setSelectedRunId, setTrades]);

  useEffect(() => {
    async function loadChart() {
      try {
        setLoading(true);
        setError(null);
        const { bars: nextBars, meta } = await fetchChart(
          symbol,
          timeframe,
          selectedRunId
        );
        setBars(nextBars);
        setMeta(meta);
        try {
          const summaryStats = await fetchSummary(symbol, timeframe);
          setSummary(summaryStats);
        } catch {
          setSummary(null);
        }
      } catch (err: any) {
        console.error(err);
        setError(err?.message ?? "Failed to fetch chart");
      } finally {
        setLoading(false);
      }
    }
    loadChart();
  }, [
    symbol,
    timeframe,
    selectedRunId,
    setLoading,
    setError,
    setBars,
    setMeta,
    setSummary,
  ]);

  useEffect(() => {
    async function loadTrades() {
      if (!selectedRunId) {
        setTrades([]);
        return;
      }
      try {
        const rows = await fetchTrades(selectedRunId);
        setTrades(rows);
      } catch (err) {
        console.error(err);
        setTrades([]);
      }
    }
    loadTrades();
  }, [selectedRunId, setTrades]);

  useEffect(() => {
    if (activeTab !== "live") return;
    fetchLiveStatus();
    const id = setInterval(() => fetchLiveStatus(), 5000);
    return () => clearInterval(id);
  }, [activeTab, fetchLiveStatus]);

  // Initialize ML Lab selection when mlTargets are loaded
  useEffect(() => {
    if (mlTargets.length > 0 && !mlSymbol) {
      setMlSymbol(mlTargets[0].symbol);
      setMlTimeframe(mlTargets[0].timeframe);
    }
  }, [mlTargets, mlSymbol]);

  useEffect(() => {
    if (activeTab === "ml" && !isLoadingMlModels && mlModels.length === 0 && mlSymbol && mlTimeframe) {
      fetchMlModels(mlSymbol, mlTimeframe);
    }
  }, [activeTab, fetchMlModels, isLoadingMlModels, mlModels.length, mlSymbol, mlTimeframe]);

  return (
    <main className="min-h-screen px-6 py-4 flex flex-col gap-4 bg-slate-50">
      <div className="flex gap-3 border-b pb-2 text-sm">
        <button
          className={`px-2 ${activeTab === "single" ? "font-semibold border-b-2 border-blue-500" : ""}`}
          onClick={() => setActiveTab("single")}
        >
          Single Backtests
        </button>
        <button
          className={`px-2 ${activeTab === "portfolio" ? "font-semibold border-b-2 border-blue-500" : ""}`}
          onClick={() => {
            setActiveTab("portfolio");
            fetchPortfolioRuns();
          }}
        >
          Portfolio
        </button>
        <button
          className={`px-2 ${activeTab === "lab" ? "font-semibold border-b-2 border-blue-500" : ""}`}
          onClick={() => setActiveTab("lab")}
        >
          Strategy Lab
        </button>
        <button
          className={`px-2 ${activeTab === "ml" ? "font-semibold border-b-2 border-blue-500" : ""}`}
          onClick={() => setActiveTab("ml")}
        >
          ML Lab
        </button>
        <button
          className={`px-2 ${activeTab === "live" ? "font-semibold border-b-2 border-blue-500" : ""}`}
          onClick={() => setActiveTab("live")}
        >
          Live
        </button>
      </div>

      {activeTab === "single" && (
        <>
          <header className="flex items-center gap-4 flex-wrap">
            <h1 className="text-xl font-bold">FinanTrade Chart</h1>

        <label className="flex items-center gap-2 text-sm">
          <span>Symbol:</span>
          <select
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
            className="border px-2 py-1 rounded"
          >
            {(availableSymbols.length ? availableSymbols : [symbol]).map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>

        <label className="flex items-center gap-2 text-sm">
          <span>TF:</span>
          <select
            value={timeframe}
            onChange={(e) => setTimeframe(e.target.value)}
            className="border px-2 py-1 rounded"
          >
            {(availableTimeframes.length ? availableTimeframes : [timeframe]).map((tf) => (
              <option key={tf} value={tf}>
                {tf}
              </option>
            ))}
          </select>
        </label>

        <StrategySelector
          strategies={availableStrategies.length ? availableStrategies : strategies}
          selected={selectedStrategy}
          onChange={(value) => setSelectedStrategy(value)}
        />

        <BacktestSelector
          runs={filteredRuns}
          selectedRunId={selectedRunId}
          onChange={(runId) => setSelectedRunId(runId)}
        />

        <button
          onClick={runBacktest}
          disabled={isRunningBacktest}
          className="px-3 py-1 border rounded text-sm"
        >
          {isRunningBacktest ? "Running..." : "Run backtest"}
        </button>

        {lastError && (
          <div className="text-sm text-red-600">
            Backtest calistirilirken hata: {lastError}
          </div>
        )}

        {lastRunMetrics && (
          <div className="text-sm text-green-700">
            Backtest tamamlandı · trades: {lastRunMetrics.trade_count ?? "-"} - cum_return:{" "}
            {lastRunMetrics.cum_return !== undefined && lastRunMetrics.cum_return !== null
              ? (lastRunMetrics.cum_return * 100).toFixed(2) + "%"
              : "-"}
          </div>
        )}

        <div className="flex items-center gap-3 text-sm">
          <label className="flex items-center gap-1">
            <input
              type="checkbox"
              checked={overlays.showRuleSignals}
              onChange={(e) =>
                setOverlays({ showRuleSignals: e.target.checked })
              }
            />
            Rule signals
          </label>
          <label className="flex items-center gap-1">
            <input
              type="checkbox"
              checked={overlays.showMicrostructure}
              onChange={(e) =>
                setOverlays({ showMicrostructure: e.target.checked })
              }
            />
            Microstructure
          </label>
          <label className="flex items-center gap-1">
            <input
              type="checkbox"
              checked={overlays.showMarketStructure}
              onChange={(e) =>
                setOverlays({ showMarketStructure: e.target.checked })
              }
            />
            Market structure
          </label>
          <label className="flex items-center gap-1">
            <input
              type="checkbox"
              checked={overlays.showMLProba}
              onChange={(e) => setOverlays({ showMLProba: e.target.checked })}
            />
            ML proba
          </label>
        </div>

        {selectedStrategy === "rule" && (
          <details className="w-full">
            <summary className="cursor-pointer text-sm font-medium">
              Advanced rule params
            </summary>
            <div className="mt-2 flex flex-wrap gap-4 text-sm">
              <label className="flex items-center gap-2">
                <span>ms_trend_min</span>
                <input
                  type="number"
                  className="border px-2 py-1 rounded w-28"
                  value={ruleParams.ms_trend_min ?? ""}
                  onChange={(e) =>
                    setRuleParams({
                      ms_trend_min: e.target.value === "" ? undefined : Number(e.target.value),
                    })
                  }
                />
              </label>
              <label className="flex items-center gap-2">
                <span>ms_trend_max</span>
                <input
                  type="number"
                  className="border px-2 py-1 rounded w-28"
                  value={ruleParams.ms_trend_max ?? ""}
                  onChange={(e) =>
                    setRuleParams({
                      ms_trend_max: e.target.value === "" ? undefined : Number(e.target.value),
                    })
                  }
                />
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={ruleParams.use_ms_chop_filter ?? false}
                  onChange={(e) => setRuleParams({ use_ms_chop_filter: e.target.checked })}
                />
                use_ms_chop_filter
              </label>
            </div>
          </details>
        )}

        {loading && <span className="text-sm text-gray-600">Loading...</span>}
        {error && <span className="text-sm text-red-600">{error}</span>}
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-[3fr_1fr] gap-4 flex-1">
        <div className="flex flex-col gap-4">
          <section className="rounded bg-white shadow p-4">
            <Chart
              bars={bars}
              overlays={overlays}
              trades={trades}
              selectedTradeId={selectedTradeId}
            />
          </section>
          <section className="rounded bg-white shadow p-4">
            <h2 className="font-semibold mb-2 text-sm uppercase tracking-wider text-gray-500">
              Trades
            </h2>
            <TradeTable
              trades={trades}
              selectedTradeId={selectedTradeId}
              onSelect={(trade) => setSelectedTradeId(trade.trade_id)}
            />
          </section>

        </div>

        <div className="flex flex-col gap-4">
          <section className="rounded bg-white shadow p-4">
            <h2 className="font-semibold mb-2 text-sm uppercase tracking-wider text-gray-500">
              Strategy Stats
            </h2>
            <ul className="text-sm space-y-1">
              <li>
                Final equity:{" "}
                <strong>
                  {summary?.final_equity
                    ? summary.final_equity.toFixed(2)
                    : "-"}
                </strong>
              </li>
              <li>
                Cum return:{" "}
                <strong>
                  {summary?.cum_return !== undefined &&
                  summary?.cum_return !== null
                    ? (summary.cum_return * 100).toFixed(2) + "%"
                    : "-"}
                </strong>
              </li>
              <li>
                Trades: <strong>{summary?.trade_count ?? "-"}</strong>
              </li>
              <li>
                Win rate:{" "}
                <strong>
                  {summary?.win_rate !== undefined && summary?.win_rate !== null
                    ? (summary.win_rate * 100).toFixed(1) + "%"
                    : "-"}
                </strong>
              </li>
              <li>Selected run: {selectedRunId ?? "None"}</li>
            </ul>
          </section>

          <section className="rounded bg-white shadow p-4">
            <h2 className="font-semibold mb-2 text-sm uppercase tracking-wider text-gray-500">
              Metadata
            </h2>
            {meta ? (
              <ul className="text-sm space-y-1">
                <li>Pipeline: {meta.pipeline_version ?? "-"}</li>
                <li>Preset: {meta.features_preset ?? "-"}</li>
                <li>Run ID: {meta.run_id ?? "-"}</li>
                <li>
                  Updated:{" "}
                  {meta.last_updated
                    ? new Date(meta.last_updated).toLocaleString()
                    : "-"}
                </li>
              </ul>
            ) : (
              <p className="text-sm text-gray-500">No metadata.</p>
            )}
          </section>

            </div>
          </div>
        </>
      )}

      {activeTab === "lab" && (
        <section className="space-y-4">
          <div className="flex flex-wrap items-center gap-4 text-sm">
            <label className="flex items-center gap-2">
              <span>Symbol</span>
              <select
                value={symbol}
                onChange={(e) => setSymbol(e.target.value)}
                className="border px-2 py-1 rounded"
              >
                {(availableSymbols.length ? availableSymbols : [symbol]).map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex items-center gap-2">
              <span>Timeframe</span>
              <select
                value={timeframe}
                onChange={(e) => setTimeframe(e.target.value)}
                className="border px-2 py-1 rounded"
              >
                {(availableTimeframes.length ? availableTimeframes : [timeframe]).map((tf) => (
                  <option key={tf} value={tf}>
                    {tf}
                  </option>
                ))}
              </select>
            </label>
            <button
              className="px-3 py-1 border rounded"
              onClick={() => fetchScenarioResults()}
              disabled={isLoadingScenarios}
            >
              {isLoadingScenarios ? "Loading..." : "Load Scenarios"}
            </button>
            {scenarioError && <span className="text-xs text-red-600">{scenarioError}</span>}
          </div>

          <div className="rounded bg-white shadow overflow-auto">
            <table className="min-w-full text-xs">
              <thead>
                <tr className="text-left border-b bg-slate-100">
                  <th className="px-2 py-1">Label</th>
                  <th className="px-2 py-1">Strategy</th>
                  <th className="px-2 py-1">Key Params</th>
                  <th className="px-2 py-1 text-right">CumRet</th>
                  <th className="px-2 py-1 text-right">Sharpe</th>
                  <th className="px-2 py-1 text-right">MaxDD</th>
                  <th className="px-2 py-1 text-right">Trades</th>
                </tr>
              </thead>
              <tbody>
                {scenarioResults.map((row) => {
                  const metrics = row.metrics ?? {};
                  const isSelected = row.scenario_id === selectedScenarioId;
                  return (
                    <tr
                      key={row.scenario_id}
                      className={`border-b last:border-0 cursor-pointer ${
                        isSelected ? "bg-blue-50" : "hover:bg-slate-50"
                      }`}
                      onClick={() => setSelectedScenarioId(row.scenario_id)}
                    >
                      <td className="px-2 py-1">{row.label ?? row.scenario_id}</td>
                      <td className="px-2 py-1">{row.strategy}</td>
                      <td className="px-2 py-1">{formatKeyParams(row.params)}</td>
                      <td className="px-2 py-1 text-right">
                        {formatMetricValue(metrics.cum_return)}
                      </td>
                      <td className="px-2 py-1 text-right">
                        {formatMetricValue(metrics.sharpe)}
                      </td>
                      <td className="px-2 py-1 text-right">
                        {formatMetricValue(metrics.max_drawdown)}
                      </td>
                      <td className="px-2 py-1 text-right">
                        {metrics.trade_count ?? "-"}
                      </td>
                    </tr>
                  );
                })}
                {scenarioResults.length === 0 && !isLoadingScenarios && (
                  <tr>
                    <td className="px-2 py-4 text-center text-slate-500" colSpan={7}>
                      No scenarios loaded yet.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          <div className="rounded bg-white shadow p-4">
            {selectedScenario ? (
              <div className="space-y-3 text-sm">
                <div className="flex flex-wrap gap-4">
                  <div>
                    <div className="text-xs uppercase text-gray-500">Label</div>
                    <div className="font-semibold">
                      {selectedScenario.label ?? selectedScenario.scenario_id}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs uppercase text-gray-500">Strategy</div>
                    <div className="font-semibold">{selectedScenario.strategy}</div>
                  </div>
                  <div>
                    <div className="text-xs uppercase text-gray-500">Symbol/TF</div>
                    <div className="font-semibold">
                      {selectedScenario.symbol}/{selectedScenario.timeframe}
                    </div>
                  </div>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <div>
                    <div className="text-xs uppercase text-gray-500">Cum Return</div>
                    <div className="font-semibold">
                      {formatMetricValue(selectedScenario.metrics?.cum_return)}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs uppercase text-gray-500">Sharpe</div>
                    <div className="font-semibold">
                      {formatMetricValue(selectedScenario.metrics?.sharpe)}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs uppercase text-gray-500">Max DD</div>
                    <div className="font-semibold">
                      {formatMetricValue(selectedScenario.metrics?.max_drawdown)}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs uppercase text-gray-500">Trades</div>
                    <div className="font-semibold">
                      {selectedScenario.metrics?.trade_count ?? "-"}
                    </div>
                  </div>
                </div>
                <div>
                  <div className="text-xs uppercase text-gray-500 mb-1">Params</div>
                  <pre className="bg-slate-50 rounded p-2 text-xs whitespace-pre-wrap">
                    {JSON.stringify(selectedScenario.params ?? {}, null, 2)}
                  </pre>
                </div>
              </div>
            ) : (
              <p className="text-sm text-gray-500">Select a scenario to view the details.</p>
            )}
          </div>
        </section>
      )}

      {activeTab === "ml" && (
        <section className="space-y-3">
          <div className="flex flex-wrap items-center gap-4 text-sm">
            <label className="flex items-center gap-2">
              <span>Symbol/TF</span>
              <select
                value={`${mlSymbol}|${mlTimeframe}`}
                onChange={(e) => {
                  const [sym, tf] = e.target.value.split("|");
                  setMlSymbol(sym);
                  setMlTimeframe(tf);
                  fetchMlModels(sym, tf);
                }}
                className="border px-2 py-1 rounded"
              >
                {mlTargets.length > 0 ? (
                  mlTargets.map((target) => (
                    <option key={`${target.symbol}-${target.timeframe}`} value={`${target.symbol}|${target.timeframe}`}>
                      {target.symbol}/{target.timeframe}
                    </option>
                  ))
                ) : (
                  <option value="">No ML targets configured</option>
                )}
              </select>
            </label>
            <button
              onClick={() => mlSymbol && mlTimeframe && fetchMlModels(mlSymbol, mlTimeframe)}
              disabled={isLoadingMlModels || !mlSymbol || !mlTimeframe}
              className="px-3 py-1 border rounded text-xs"
            >
              {isLoadingMlModels ? "Loading..." : "Refresh"}
            </button>
            {mlError && <span className="text-xs text-red-600">{mlError}</span>}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-[1.2fr,1.8fr] gap-4">
          <div className="border rounded bg-white shadow p-3 space-y-2 text-xs">
            <div className="flex items-center justify-between gap-2">
              <div className="font-semibold text-sm">
                Models ({mlSymbol}/{mlTimeframe})
              </div>
            </div>
            <div className="max-h-72 overflow-auto border rounded">
              {mlModels.length === 0 && !isLoadingMlModels ? (
                <div className="p-2 text-slate-500">No models found.</div>
              ) : (
                <ul>
                  {mlModels.map((m) => {
                    const isSelected = m.model_id === selectedModelId;
                    const acc = m.metrics?.accuracy ?? m.metrics?.acc;
                    const f1 = m.metrics?.f1;
                    return (
                      <li
                        key={m.model_id}
                        className={`px-2 py-1 cursor-pointer border-b last:border-b-0 hover:bg-slate-50 ${
                          isSelected ? "bg-blue-50" : ""
                        }`}
                        onClick={() => selectModel(m.model_id)}
                      >
                        <div className="flex justify-between">
                          <span className="font-medium text-[11px]">
                            {m.model_type} · {m.model_id.slice(0, 8)}
                          </span>
                          <span className="text-[10px] text-slate-500">
                            {new Date(m.created_at).toLocaleString()}
                          </span>
                        </div>
                        <div className="flex gap-3 text-[10px] text-slate-700">
                          <span>acc: {acc !== undefined ? acc.toFixed(3) : "-"}</span>
                          <span>f1: {f1 !== undefined ? f1.toFixed(3) : "-"}</span>
                        </div>
                      </li>
                    );
                  })}
                </ul>
              )}
            </div>
          </div>

          <div className="border rounded bg-white shadow p-3 space-y-3 text-xs">
            <div className="flex justify-between items-center">
              <div className="font-semibold text-sm">Model details</div>
              {isLoadingFeatureImportance && (
                <span className="text-[10px] text-slate-500">Loading importance...</span>
              )}
            </div>
            {selectedMlModel ? (
              <>
                <div className="flex flex-wrap gap-4">
                  <div>
                    <div className="text-xs uppercase text-gray-500">Model</div>
                    <div className="font-semibold text-[11px] break-all">{selectedMlModel.model_id}</div>
                  </div>
                  <div>
                    <div className="text-xs uppercase text-gray-500">Created</div>
                    <div className="font-semibold">
                      {new Date(selectedMlModel.created_at).toLocaleString()}
                    </div>
                  </div>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                  {[
                    { label: "Accuracy", value: selectedMlModel.metrics?.accuracy },
                    { label: "Precision", value: selectedMlModel.metrics?.precision },
                    { label: "Recall", value: selectedMlModel.metrics?.recall },
                    { label: "F1", value: selectedMlModel.metrics?.f1 },
                    { label: "ROC AUC", value: selectedMlModel.metrics?.roc_auc },
                    { label: "Sharpe", value: selectedMlModel.metrics?.sharpe },
                    { label: "Cum Return", value: selectedMlModel.metrics?.cum_return },
                  ].map((entry) => (
                    <div key={entry.label} className="rounded border p-2 text-center">
                      <div className="text-[10px] uppercase text-gray-500">{entry.label}</div>
                      <div className="font-semibold text-sm">{formatMetricValue(entry.value)}</div>
                    </div>
                  ))}
                </div>
                <div>
                  <div className="font-medium mb-1">Feature importance</div>
                  {featureImportance.length === 0 ? (
                    <div className="text-slate-500 text-[11px]">
                      No feature importance data for this model.
                    </div>
                  ) : (
                    <div className="max-h-64 overflow-auto space-y-1">
                      {featureImportance
                        .slice()
                        .sort((a, b) => b.value - a.value)
                        .map((fi) => (
                          <div key={fi.name} className="flex items-center gap-2">
                            <div className="w-32 truncate" title={fi.name}>
                              {fi.name}
                            </div>
                            <div className="flex-1 bg-slate-100 h-3 rounded">
                              <div
                                className="h-3 rounded bg-blue-500"
                                style={{ width: `${Math.min(fi.value * 100, 100)}%` }}
                              />
                            </div>
                            <div className="w-12 text-right text-[10px]">
                              {(fi.value * 100).toFixed(1)}%
                            </div>
                          </div>
                        ))}
                    </div>
                  )}
                </div>
              </>
            ) : (
              <div className="text-slate-500 text-[11px]">Select a model from the list.</div>
            )}
          </div>
          </div>
        </section>
      )}

      {activeTab === "portfolio" && (
        <section className="space-y-3">
          <div className="flex items-center gap-3 text-sm">
            <select
              className="border px-2 py-1 rounded"
              value={selectedPortfolioRunId ?? ""}
              onChange={(e) => fetchPortfolioEquity(e.target.value)}
            >
              {portfolioRuns.map((run) => (
                <option key={run.run_id} value={run.run_id}>
                  {run.run_id} ({run.symbols.join(", ")})
                </option>
              ))}
            </select>
            <button
              className="px-3 py-1 border rounded"
              onClick={() => selectedPortfolioRunId && fetchPortfolioEquity(selectedPortfolioRunId)}
              disabled={!selectedPortfolioRunId || isLoadingPortfolioEquity}
            >
              {isLoadingPortfolioEquity ? "Loading..." : "Load"}
            </button>
            {portfolioError && <span className="text-red-600">{portfolioError}</span>}
          </div>
          <div className="rounded bg-white shadow p-4">
            <h2 className="font-semibold mb-2 text-sm uppercase tracking-wider text-gray-500">Portfolio Equity</h2>
            <p className="text-sm text-gray-600">Points: {portfolioEquity.length}</p>
          </div>
          {selectedPortfolioRunId && (
            <div className="rounded bg-white shadow p-4">
              <h3 className="font-semibold text-sm mb-2">Metrics</h3>
              {(() => {
                const run = portfolioRuns.find((r) => r.run_id === selectedPortfolioRunId);
                if (!run) return <p className="text-sm text-gray-500">No metrics.</p>;
                const m = run.metrics || {};
                return (
                  <ul className="text-sm space-y-1">
                    <li>Final equity: {m.final_equity ?? "-"}</li>
                    <li>
                      Cum return:{" "}
                      {m.cum_return !== null && m.cum_return !== undefined
                        ? (m.cum_return * 100).toFixed(2) + "%"
                        : "-"}
                    </li>
                    <li>
                      Max DD:{" "}
                      {m.max_drawdown !== null && m.max_drawdown !== undefined
                        ? (m.max_drawdown * 100).toFixed(2) + "%"
                        : "-"}
                    </li>
                    <li>Sharpe: {m.sharpe ?? "-"}</li>
                  </ul>
                );
              })()}
            </div>
          )}
        </section>
      )}

      {activeTab === "live" && (
        <section className="space-y-3">
          <div className="flex gap-2 text-sm">
            <button
              className="px-3 py-1 border rounded"
              onClick={() => sendLiveCommand("pause")}
              disabled={isSendingLiveCommand}
            >
              Pause
            </button>
            <button
              className="px-3 py-1 border rounded"
              onClick={() => sendLiveCommand("resume")}
              disabled={isSendingLiveCommand}
            >
              Resume
            </button>
            <button
              className="px-3 py-1 border rounded"
              onClick={() => sendLiveCommand("flatten")}
              disabled={isSendingLiveCommand}
            >
              Flatten
            </button>
            <button
              className="px-3 py-1 border rounded"
              onClick={() => sendLiveCommand("stop")}
              disabled={isSendingLiveCommand}
            >
              Stop
            </button>
            {liveError && <span className="text-red-600">{liveError}</span>}
          </div>

          {liveStatus && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div className="rounded bg-white shadow p-3">
                <div className="text-gray-500 text-xs">Equity</div>
                <div className="font-semibold">{liveStatus.equity.toFixed(2)}</div>
              </div>
              <div className="rounded bg-white shadow p-3">
                <div className="text-gray-500 text-xs">Realized PnL</div>
                <div className="font-semibold">{(liveStatus.realized_pnl ?? 0).toFixed(2)}</div>
              </div>
              <div className="rounded bg-white shadow p-3">
                <div className="text-gray-500 text-xs">Unrealized PnL</div>
                <div className="font-semibold">{(liveStatus.unrealized_pnl ?? 0).toFixed(2)}</div>
              </div>
              <div className="rounded bg-white shadow p-3">
                <div className="text-gray-500 text-xs">Daily Realized</div>
                <div className="font-semibold">{(liveStatus.daily_realized_pnl ?? 0).toFixed(2)}</div>
              </div>
            </div>
          )}

          {liveStatus && (
            <div className="rounded bg-white shadow p-4">
              <h3 className="font-semibold text-sm mb-2">Open Positions</h3>
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-left border-b">
                    <th className="pr-2">Symbol</th>
                    <th className="pr-2">Side</th>
                    <th className="pr-2">Qty</th>
                    <th className="pr-2">Entry</th>
                    <th className="pr-2">PnL</th>
                  </tr>
                </thead>
                <tbody>
                  {liveStatus.open_positions.map((p) => (
                    <tr key={`${p.symbol}-${p.side}-${p.entry_price}`} className="border-b last:border-0">
                      <td className="pr-2 py-1">{p.symbol}</td>
                      <td className="pr-2">{p.side}</td>
                      <td className="pr-2">{p.qty}</td>
                      <td className="pr-2">{p.entry_price}</td>
                      <td className="pr-2">{p.pnl?.toFixed?.(2) ?? "-"}</td>
                    </tr>
                  ))}
                  {liveStatus.open_positions.length === 0 && (
                    <tr>
                      <td className="py-2 text-gray-500" colSpan={5}>
                        No open positions.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          )}
        </section>
      )}
    </main>
  );
}

