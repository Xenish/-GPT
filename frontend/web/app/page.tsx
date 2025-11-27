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
    availableScenarioPresets,
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
    liveStatus,
    liveError,
    isSendingLiveCommand,
    fetchLiveStatus,
    sendLiveCommand,
  } = useChartStore();

  const [activeTab, setActiveTab] = useState<"single" | "portfolio" | "live">("single");
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

  useEffect(() => {
    if (activeTab !== "live") return;
    fetchLiveStatus();
    const id = setInterval(() => fetchLiveStatus(), 5000);
    return () => clearInterval(id);
  }, [activeTab, fetchLiveStatus]);

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
        </>
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

