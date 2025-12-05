"use client";

import { useEffect, useState } from "react";
import { fetchLiveReport, fetchLiveStatus } from "@/lib/api";

type Position = {
  symbol: string;
  side: string;
  qty: number;
  entry_price: number;
  current_price?: number | null;
  pnl?: number | null;
};

type Trade = {
  timestamp?: string;
  side?: string;
  qty?: number;
  price?: number;
  pnl?: number;
};

export default function LiveDashboardPage() {
  const [status, setStatus] = useState<any | null>(null);
  const [report, setReport] = useState<any | null>(null);
  const [positions, setPositions] = useState<Position[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const stale = (status?.heartbeat_age_sec ?? 0) > 30;
  const killSwitch = Boolean(status?.kill_switch_triggered);

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      try {
        const liveStatus = await fetchLiveStatus();
        setStatus(liveStatus);
        const rep = await fetchLiveReport("json");
        setReport(rep);
        const tradesDf = rep?.sections?.find((s: any) => s.title === "Recent Trades")?.data?.trades;
        setTrades(tradesDf ?? []);
        setPositions(liveStatus?.open_positions ?? []);
        setError(null);
      } catch (err: any) {
        setError(err?.message ?? "Failed to load live dashboard");
      } finally {
        setLoading(false);
      }
    };
    load();
    const id = setInterval(load, 5000);
    return () => clearInterval(id);
  }, []);

  return (
    <main className="p-6 space-y-4">
      <header className="flex items-center gap-3">
        <h1 className="text-2xl font-semibold">Live Dashboard</h1>
        {loading && <span className="text-sm text-gray-500">Loading...</span>}
        {error && <span className="text-sm text-red-600">{error}</span>}
      </header>

      {stale && (
        <div className="rounded border border-red-300 bg-red-50 text-red-700 px-3 py-2 text-sm">
          Heartbeat stale ({status?.heartbeat_age_sec ?? "?"}s). Check data/WS connection.
        </div>
      )}
      {killSwitch && (
        <div className="rounded border border-red-500 bg-red-100 text-red-900 px-4 py-3 text-sm font-semibold">
          Kill switch triggered: {status?.kill_switch_reason ?? "n/a"}
        </div>
      )}

      <section className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <InfoCard label="Run ID" value={status?.run_id ?? "-"} />
        <InfoCard label="Mode" value={status?.mode ?? "-"} />
        <InfoCard label="Symbol/TF" value={`${status?.symbol ?? "-"} / ${status?.timeframe ?? "-"}`} />
        <InfoCard label="Heartbeat Age (s)" value={status?.heartbeat_age_sec ?? "-"} />
        <InfoCard label="Equity Now" value={status?.equity_now ?? status?.equity ?? "-"} />
        <InfoCard label="Daily PnL" value={status?.daily_pnl ?? "-"} />
        <InfoCard label="Max Intraday DD" value={status?.max_intraday_dd ?? "-"} />
        <InfoCard label="Validation Issues" value={(status?.validation_issues ?? []).join(", ") || "-"} />
      </section>

      <section className="rounded bg-white shadow p-4">
        <div className="flex justify-between items-center mb-2">
          <h2 className="font-semibold text-sm uppercase text-gray-600">Mini Equity</h2>
          {report?.artifacts?.snapshot_path && (
            <span className="text-xs text-gray-500">Snapshot: {report.artifacts.snapshot_path}</span>
          )}
        </div>
        <div className="text-sm text-gray-500">Render equity sparkline from live report (placeholder).</div>
      </section>

      <section className="rounded bg-white shadow p-4">
        <h2 className="font-semibold text-sm uppercase text-gray-600 mb-2">Open Positions</h2>
        <div className="overflow-auto">
          <table className="min-w-full text-xs">
            <thead>
              <tr className="border-b">
                <th className="px-2 py-1 text-left">Symbol</th>
                <th className="px-2 py-1 text-left">Side</th>
                <th className="px-2 py-1 text-right">Qty</th>
                <th className="px-2 py-1 text-right">Entry</th>
                <th className="px-2 py-1 text-right">Current</th>
                <th className="px-2 py-1 text-right">Unrealized PnL</th>
              </tr>
            </thead>
            <tbody>
              {positions.map((p, idx) => (
                <tr key={idx} className="border-b last:border-0">
                  <td className="px-2 py-1">{p.symbol}</td>
                  <td className="px-2 py-1">{p.side}</td>
                  <td className="px-2 py-1 text-right">{p.qty}</td>
                  <td className="px-2 py-1 text-right">{p.entry_price}</td>
                  <td className="px-2 py-1 text-right">{p.current_price ?? "-"}</td>
                  <td className="px-2 py-1 text-right">{p.pnl ?? "-"}</td>
                </tr>
              ))}
              {positions.length === 0 && (
                <tr>
                  <td className="px-2 py-3 text-center text-gray-500" colSpan={6}>
                    No open positions.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>

      <section className="rounded bg-white shadow p-4">
        <h2 className="font-semibold text-sm uppercase text-gray-600 mb-2">Last Trades</h2>
        <div className="overflow-auto">
          <table className="min-w-full text-xs">
            <thead>
              <tr className="border-b">
                <th className="px-2 py-1 text-left">Time</th>
                <th className="px-2 py-1 text-left">Side</th>
                <th className="px-2 py-1 text-right">Size</th>
                <th className="px-2 py-1 text-right">Price</th>
                <th className="px-2 py-1 text-right">Realized PnL</th>
              </tr>
            </thead>
            <tbody>
              {trades.map((t, idx) => (
                <tr key={idx} className="border-b last:border-0">
                  <td className="px-2 py-1">{t.timestamp ?? "-"}</td>
                  <td className="px-2 py-1">{t.side ?? "-"}</td>
                  <td className="px-2 py-1 text-right">{t.qty ?? "-"}</td>
                  <td className="px-2 py-1 text-right">{t.price ?? "-"}</td>
                  <td className="px-2 py-1 text-right">{t.pnl ?? "-"}</td>
                </tr>
              ))}
              {trades.length === 0 && (
                <tr>
                  <td className="px-2 py-3 text-center text-gray-500" colSpan={5}>
                    No recent trades.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>

      <section className="rounded bg-white shadow p-4">
        <h2 className="font-semibold text-sm uppercase text-gray-600 mb-2">Risk Summary</h2>
        <ul className="text-sm space-y-1">
          <li>Max position notional: {status?.risk_stats?.max_position_notional ?? "-"}</li>
          <li>Max open trades: {status?.risk_stats?.max_open_trades ?? "-"}</li>
          <li>
            Daily loss limit: {status?.risk_stats?.daily_loss_limit ?? "-"} (usage:{" "}
            {status?.risk_stats?.daily_loss_used ?? "-"})
          </li>
        </ul>
      </section>
    </main>
  );
}

function InfoCard({ label, value }: { label: string; value: any }) {
  return (
    <div className="rounded border bg-white p-3">
      <div className="text-xs uppercase text-gray-500">{label}</div>
      <div className="font-semibold text-lg">{value ?? "-"}</div>
    </div>
  );
}
