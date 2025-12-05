"use client";

import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import { ReportSection, fetchBacktestReport, fetchTrades } from "@/lib/api";
import dynamic from "next/dynamic";

const Chart = dynamic(() => import("@/components/Chart"), { ssr: false });

type Props = { params: { jobId: string } };

export default function BacktestDetailPage({ params }: Props) {
  const searchParams = useSearchParams();
  const runId = searchParams.get("run_id") ?? params.jobId;
  const [report, setReport] = useState<any | null>(null);
  const [trades, setTrades] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const rep = await fetchBacktestReport(runId);
      setReport(rep);
      const tradePath = rep?.artifacts?.trades_csv;
      if (tradePath) {
        const res = await fetch(tradePath);
        if (res.ok) {
          const text = await res.text();
          // naive CSV parse
          const [header, ...rows] = text.trim().split("\n");
          const cols = header.split(",");
          const parsed = rows.map((line) => {
            const vals = line.split(",");
            const obj: Record<string, any> = {};
            cols.forEach((c, i) => (obj[c] = vals[i]));
            return obj;
          });
          setTrades(parsed);
        }
      }
    } catch (err: any) {
      setError(err?.message ?? "Failed to load backtest");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, [runId]);

  const overviewSection: ReportSection | undefined = useMemo(
    () => report?.sections?.find((s: ReportSection) => s.title === "Overview" || s.title === "Job Overview"),
    [report]
  );

  const equityCsv = report?.artifacts?.equity_curve_csv;

  return (
    <main className="p-6 space-y-4">
      <header className="flex flex-col gap-1">
        <Link href="/backtests" className="text-sm text-blue-600 underline">
          ← Backtests
        </Link>
        <h1 className="text-2xl font-semibold">Backtest {runId}</h1>
        {report?.strategy_id && (
          <p className="text-sm text-gray-600">
            {report.strategy_id} · {report.symbol}/{report.timeframe}
          </p>
        )}
      </header>

      {loading && (
        <div className="space-y-2">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="h-10 bg-slate-100 animate-pulse rounded" />
          ))}
        </div>
      )}
      {error && (
        <div className="text-red-700 bg-red-50 border border-red-200 rounded px-3 py-2 text-sm flex items-center gap-2">
          <span>{error}</span>
          <button className="px-2 py-1 border rounded text-xs" onClick={load}>
            Retry
          </button>
        </div>
      )}

      {report && (
        <>
          <div className="flex gap-2">
            <button
              className="px-3 py-1 border rounded text-sm"
              onClick={async () => {
                const repHtml = await fetchBacktestReport(runId, "html");
                const blob = new Blob([repHtml as any], { type: "text/html" });
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = `${runId}.html`;
                a.click();
                URL.revokeObjectURL(url);
              }}
            >
              Download report.html
            </button>
            <button
              className="px-3 py-1 border rounded text-sm"
              onClick={async () => {
                const repMd = await fetchBacktestReport(runId, "markdown");
                const blob = new Blob([repMd as any], { type: "text/markdown" });
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = `${runId}.md`;
                a.click();
                URL.revokeObjectURL(url);
              }}
            >
              Download report.md
            </button>
          </div>

          <section className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {Object.entries(report.metrics || {}).map(([k, v]) => (
              <div key={k} className="rounded border bg-white p-3">
                <div className="text-xs uppercase text-gray-500">{k}</div>
                <div className="font-semibold text-lg">{typeof v === "number" ? v.toFixed(3) : String(v)}</div>
              </div>
            ))}
          </section>

          {equityCsv && (
            <div className="rounded bg-white shadow p-4">
              <div className="flex justify-between items-center mb-2">
                <h2 className="font-semibold text-sm uppercase text-gray-600">Equity Curve</h2>
                <a href={equityCsv} className="text-blue-600 underline text-sm">
                  Download CSV
                </a>
              </div>
              <div className="text-sm text-gray-500">Render CSV using lightweight-charts (placeholder)</div>
            </div>
          )}

          <section className="rounded bg-white shadow p-4">
            <div className="flex justify-between items-center mb-2">
              <h2 className="font-semibold text-sm uppercase text-gray-600">Trades</h2>
              {report?.artifacts?.trades_csv && (
                <a href={report.artifacts.trades_csv} className="text-blue-600 underline text-sm">
                  Download CSV
                </a>
              )}
            </div>
            <div className="overflow-auto">
              <table className="min-w-full text-xs">
                <thead>
                  <tr className="border-b">
                    <th className="px-2 py-1 text-left">Entry</th>
                    <th className="px-2 py-1 text-left">Exit</th>
                    <th className="px-2 py-1 text-left">Side</th>
                    <th className="px-2 py-1 text-right">PnL</th>
                  </tr>
                </thead>
                <tbody>
                  {trades.map((t, idx) => (
                    <tr key={idx} className="border-b last:border-0">
                      <td className="px-2 py-1">{t.timestamp || t.entry_time || "-"}</td>
                      <td className="px-2 py-1">{t.timestamp_exit || t.exit_time || "-"}</td>
                      <td className="px-2 py-1">{t.side || "-"}</td>
                      <td className="px-2 py-1 text-right">{t.pnl ?? "-"}</td>
                    </tr>
                  ))}
                  {trades.length === 0 && (
                    <tr>
                      <td className="px-2 py-3 text-center text-gray-500" colSpan={4}>
                        No trades parsed.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </section>

          {overviewSection && (
            <section className="rounded bg-white shadow p-4">
              <h2 className="font-semibold text-sm uppercase text-gray-600">{overviewSection.title}</h2>
              <pre className="text-xs bg-slate-50 p-2 rounded whitespace-pre-wrap">{overviewSection.content}</pre>
            </section>
          )}
        </>
      )}
    </main>
  );
}
