"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { fetchBacktests } from "@/lib/api";

type BacktestRow = {
  run_id: string;
  strategy: string;
  symbol: string;
  timeframe: string;
  metrics?: Record<string, number | null>;
};

export default function BacktestsPage() {
  const [symbol, setSymbol] = useState("AIAUSDT");
  const [timeframe, setTimeframe] = useState("15m");
  const [strategy, setStrategy] = useState<string | undefined>(undefined);
  const [rows, setRows] = useState<BacktestRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // hydrate filters from localStorage
  useEffect(() => {
    const stored = typeof window !== "undefined" ? window.localStorage.getItem("bt_filters") : null;
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        setSymbol(parsed.symbol ?? symbol);
        setTimeframe(parsed.timeframe ?? timeframe);
        setStrategy(parsed.strategy ?? strategy);
      } catch {
        /* ignore */
      }
    }
  }, []);

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await fetchBacktests(symbol, timeframe, strategy);
        setRows(data);
      } catch (err: any) {
        setRows([]);
        setError(err?.message ?? "Failed to load backtests");
      } finally {
        setLoading(false);
      }
    };
    load();

    if (typeof window !== "undefined") {
      window.localStorage.setItem(
        "bt_filters",
        JSON.stringify({ symbol, timeframe, strategy })
      );
    }
  }, [symbol, timeframe, strategy]);

  return (
    <main className="p-6 space-y-4">
      <header className="flex flex-wrap gap-3 items-center">
        <h1 className="text-xl font-semibold">Backtests</h1>
        <label className="text-sm flex items-center gap-1">
          Symbol
          <input
            className="border rounded px-2 py-1"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
          />
        </label>
        <label className="text-sm flex items-center gap-1">
          Timeframe
          <input
            className="border rounded px-2 py-1"
            value={timeframe}
            onChange={(e) => setTimeframe(e.target.value)}
          />
        </label>
        <label className="text-sm flex items-center gap-1">
          Strategy
          <input
            className="border rounded px-2 py-1"
            value={strategy ?? ""}
            placeholder="optional"
            onChange={(e) => setStrategy(e.target.value || undefined)}
          />
        </label>
        {loading && <span className="text-sm text-gray-500">Loading...</span>}
        {error && <span className="text-sm text-red-600">{error}</span>}
      </header>

      <div className="rounded bg-white shadow">
        {loading ? (
          <div className="p-4 space-y-2">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="h-10 bg-slate-100 animate-pulse rounded" />
            ))}
          </div>
        ) : (
          <table className="w-full text-sm">
          <thead>
            <tr className="text-left border-b bg-slate-100">
              <th className="px-3 py-2">Job ID</th>
              <th className="px-3 py-2">Strategy</th>
              <th className="px-3 py-2">Symbol</th>
              <th className="px-3 py-2">TF</th>
              <th className="px-3 py-2">Sharpe</th>
              <th className="px-3 py-2">Total Return</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const m = r.metrics || {};
              return (
                <tr key={r.run_id} className="border-b last:border-0 hover:bg-slate-50">
                  <td className="px-3 py-2">
                    <Link className="text-blue-600 underline" href={`/backtests/${r.run_id}`}>
                      {r.run_id}
                    </Link>
                  </td>
                  <td className="px-3 py-2">{r.strategy}</td>
                  <td className="px-3 py-2">{r.symbol}</td>
                  <td className="px-3 py-2">{r.timeframe}</td>
                  <td className="px-3 py-2">{m.sharpe ?? "-"}</td>
                  <td className="px-3 py-2">{m.cum_return !== null && m.cum_return !== undefined ? (m.cum_return * 100).toFixed(2) + "%" : "-"}</td>
                </tr>
              );
            })}
            {rows.length === 0 && !loading && (
              <tr>
                <td className="px-3 py-6 text-center text-gray-500" colSpan={6}>
                  No backtests found for filters.
                </td>
              </tr>
            )}
          </tbody>
          </table>
        )}
        {error && (
          <div className="p-3 text-sm text-red-700 flex items-center gap-2">
            <span>{error}</span>
            <button
              className="px-2 py-1 border rounded text-xs"
              onClick={() => {
                // trigger reload
                setSymbol((s) => s);
              }}
            >
              Retry
            </button>
          </div>
        )}
      </div>
    </main>
  );
}
