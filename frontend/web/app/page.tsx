"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";
import { fetchBacktests, fetchChart, fetchSummary, fetchTrades } from "@/lib/api";
import { useChartStore } from "@/store/useChartStore";
import StrategySelector from "@/components/StrategySelector";
import BacktestSelector from "@/components/BacktestSelector";
import TradeTable from "@/components/TradeTable";
import LiveMonitor from "@/components/LiveMonitor";

const Chart = dynamic(() => import("@/components/Chart"), { ssr: false });

export default function HomePage() {
  const {
    symbol,
    timeframe,
    strategies,
    availableSymbols,
    availableTimeframes,
    availableStrategies,
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
    scenarioPresets,
    scenarioPreset,
    scenarioResults,
    runScenarioPreset,
    isRunningScenario,
    scenarioError,
    portfolioRuns,
    selectedPortfolioRunId,
    portfolioEquity,
    isLoadingPortfolioEquity,
    portfolioError,
    fetchPortfolioRuns,
    fetchPortfolioEquity,
  } = useChartStore();

  const [activeTab, setActiveTab] = useState<"single" | "portfolio">("single");
  const filteredRuns = useMemo(
    () => backtests.filter((run) => run.strategy === selectedStrategy),
    [backtests, selectedStrategy]
  );

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

  return (
    <main className="min-h-screen px-6 py-4 flex flex-col gap-4 bg-slate-50">
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
            Backtest Ã§alÄ±ÅŸtÄ±rÄ±lÄ±rken hata: {lastError}
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

          <section className="rounded bg-white shadow p-4">
            <h2 className="font-semibold mb-2 text-sm uppercase tracking-wider text-gray-500">
              Scenarios
            </h2>
            <div className="flex flex-wrap items-center gap-3 text-sm mb-3">
              <label className="flex items-center gap-2">
                <span>Preset</span>
                <select
                  className="border px-2 py-1 rounded"
                  value={scenarioPreset}
                  onChange={(e) => runScenarioPreset(e.target.value)}
                  disabled={isRunningScenario}
                >
                  <option value="core_15m">core_15m</option>
                </select>
              </label>
              <button
                className="px-3 py-1 border rounded"
                onClick={() => runScenarioPreset()}
                disabled={isRunningScenario}
              >
                {isRunningScenario ? "Running..." : "Run scenarios"}
              </button>
              {scenarioError && <span className="text-red-600">{scenarioError}</span>}
            </div>
            <div className="overflow-auto">
              <table className="min-w-full text-sm">
                <thead>
                  <tr className="text-left border-b">
                    <th className="pr-4">Label</th>
                    <th className="pr-4">Strategy</th>
                    <th className="pr-4">Cum Return</th>
                    <th className="pr-4">Sharpe</th>
                    <th className="pr-4">Trades</th>
                  </tr>
                </thead>
                <tbody>
                  {scenarioResults.map((row) => (
                    <tr key={row.label + row.strategy} className="border-b last:border-0">
                      <td className="pr-4 py-1">{row.label}</td>
                      <td className="pr-4">{row.strategy}</td>
                      <td className="pr-4">
                        {row.cum_return !== null && row.cum_return !== undefined
                          ? (row.cum_return * 100).toFixed(2) + "%"
                          : "-"}
                      </td>
                      <td className="pr-4">
                        {row.sharpe !== null && row.sharpe !== undefined ? row.sharpe.toFixed(2) : "-"}
                      </td>
                      <td className="pr-4">{row.trade_count ?? "-"}</td>
                    </tr>
                  ))}
                  {scenarioResults.length === 0 && (
                    <tr>
                      <td className="py-2 text-gray-500" colSpan={5}>
                        No scenario results yet.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
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

          <LiveMonitor />
        </div>
      </div>
    </main>
  );
}

