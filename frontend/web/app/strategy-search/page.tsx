"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { fetchStrategySearchReport } from "@/lib/api";

type JobSummary = {
  job_id: string;
  strategy_id?: string | null;
  symbol?: string | null;
  timeframe?: string | null;
  metrics?: Record<string, any>;
};

export default function StrategySearchPage() {
  const [jobId, setJobId] = useState("");
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const loadJob = async () => {
    if (!jobId) return;
    try {
      setLoading(true);
      const rep = await fetchStrategySearchReport(jobId);
      setJobs((prev) => {
        const next = prev.filter((j) => j.job_id !== jobId);
        next.unshift({
          job_id: jobId,
          strategy_id: rep.strategy_id,
          symbol: rep.symbol,
          timeframe: rep.timeframe,
          metrics: rep.metrics,
        });
        return next;
      });
      setError(null);
    } catch (err: any) {
      setError(err?.message ?? "Failed to load job");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="p-6 space-y-4">
      <header className="flex flex-wrap items-center gap-3">
        <h1 className="text-xl font-semibold">Strategy Search Jobs</h1>
        <input
          className="border rounded px-2 py-1 text-sm"
          placeholder="job_id"
          value={jobId}
          onChange={(e) => setJobId(e.target.value)}
        />
        <button className="px-3 py-1 border rounded text-sm" onClick={loadJob}>
          Load
        </button>
        {loading && <span className="text-sm text-gray-500">Loading...</span>}
        {error && (
          <span className="text-sm text-red-600 flex items-center gap-2">
            {error}
            <button className="px-2 py-1 border rounded text-xs" onClick={loadJob}>
              Retry
            </button>
          </span>
        )}
      </header>

      <div className="rounded bg-white shadow">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left border-b bg-slate-100">
              <th className="px-3 py-2">Job ID</th>
              <th className="px-3 py-2">Strategy</th>
              <th className="px-3 py-2">Symbol</th>
              <th className="px-3 py-2">TF</th>
              <th className="px-3 py-2">Best Sharpe</th>
            </tr>
          </thead>
          <tbody>
            {jobs.map((j) => (
              <tr key={j.job_id} className="border-b last:border-0 hover:bg-slate-50">
                <td className="px-3 py-2">
                  <Link href={`/strategy-search/${j.job_id}`} className="text-blue-600 underline">
                    {j.job_id}
                  </Link>
                </td>
                <td className="px-3 py-2">{j.strategy_id ?? "-"}</td>
                <td className="px-3 py-2">{j.symbol ?? "-"}</td>
                <td className="px-3 py-2">{j.timeframe ?? "-"}</td>
                <td className="px-3 py-2">{j.metrics?.best_sharpe ?? "-"}</td>
              </tr>
            ))}
            {jobs.length === 0 && (
              <tr>
                <td className="px-3 py-6 text-center text-gray-500" colSpan={5}>
                  Enter a job_id to load strategy search report summary.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </main>
  );
}
