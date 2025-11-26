"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo } from "react";
import {
  fetchBacktests,
  fetchChart,
  fetchSummary,
  fetchTrades,
  runBacktest,
} from "@/lib/api";
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
    isRunning,
    setIsRunning,
    setLastRunId,
  } = useChartStore();

  const filteredRuns = useMemo(
    () => backtests.filter((run) => run.strategy === selectedStrategy),
    [backtests, selectedStrategy]
  );

  useEffect(() => {
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

  async function handleRunBacktest() {
    try {
      setIsRunning(true);
      const res = await runBacktest(symbol, timeframe, selectedStrategy);
      setLastRunId(res.run_id);
      const runs = await fetchBacktests(symbol, timeframe);
      setBacktests(runs);
      setSelectedRunId(res.run_id);
    } catch (err) {
      console.error(err);
      window.alert("Backtest çalıştırılırken hata oluştu.");
    } finally {
      setIsRunning(false);
    }
  }

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
            <option value="AIAUSDT">AIAUSDT</option>
          </select>
        </label>

        <label className="flex items-center gap-2 text-sm">
          <span>TF:</span>
          <select
            value={timeframe}
            onChange={(e) => setTimeframe(e.target.value)}
            className="border px-2 py-1 rounded"
          >
            <option value="15m">15m</option>
          </select>
        </label>

        <StrategySelector
          strategies={strategies}
          selected={selectedStrategy}
          onChange={(value) => setSelectedStrategy(value)}
        />

        <BacktestSelector
          runs={filteredRuns}
          selectedRunId={selectedRunId}
          onChange={(runId) => setSelectedRunId(runId)}
        />

        <button
          onClick={handleRunBacktest}
          disabled={isRunning}
          className="px-3 py-1 border rounded text-sm"
        >
          {isRunning ? "Running..." : "Run backtest"}
        </button>

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
                    : "—"}
                </strong>
              </li>
              <li>
                Cum return:{" "}
                <strong>
                  {summary?.cum_return !== undefined &&
                  summary?.cum_return !== null
                    ? (summary.cum_return * 100).toFixed(2) + "%"
                    : "—"}
                </strong>
              </li>
              <li>
                Trades: <strong>{summary?.trade_count ?? "—"}</strong>
              </li>
              <li>
                Win rate:{" "}
                <strong>
                  {summary?.win_rate !== undefined && summary?.win_rate !== null
                    ? (summary.win_rate * 100).toFixed(1) + "%"
                    : "—"}
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
                <li>Pipeline: {meta.pipeline_version ?? "—"}</li>
                <li>Preset: {meta.features_preset ?? "—"}</li>
                <li>Run ID: {meta.run_id ?? "—"}</li>
                <li>
                  Updated:{" "}
                  {meta.last_updated
                    ? new Date(meta.last_updated).toLocaleString()
                    : "—"}
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
