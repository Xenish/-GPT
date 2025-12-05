"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { ReportSection, fetchStrategySearchReport } from "@/lib/api";

type Props = { params: { jobId: string } };

export default function StrategySearchDetailPage({ params }: Props) {
  const [report, setReport] = useState<any | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const rep = await fetchStrategySearchReport(params.jobId);
      setReport(rep);
    } catch (err: any) {
      setError(err?.message ?? "Failed to load report");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, [params.jobId]);

  const recommendations = useMemo(
    () => report?.sections?.find((s: ReportSection) => s.title?.toLowerCase().includes("recommendations")),
    [report]
  );
  const topSection = useMemo(
    () => report?.sections?.find((s: ReportSection) => s.title?.toLowerCase().includes("top performers")),
    [report]
  );

  return (
    <main className="p-6 space-y-4">
      <header className="flex flex-col gap-1">
        <Link href="/strategy-search" className="text-sm text-blue-600 underline">
          ← Strategy Search
        </Link>
        <h1 className="text-2xl font-semibold">Strategy Search {params.jobId}</h1>
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
                const repHtml = await fetchStrategySearchReport(params.jobId, "html");
                const blob = new Blob([repHtml as any], { type: "text/html" });
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = `${params.jobId}.html`;
                a.click();
                URL.revokeObjectURL(url);
              }}
            >
              Download report.html
            </button>
            <button
              className="px-3 py-1 border rounded text-sm"
              onClick={async () => {
                const repMd = await fetchStrategySearchReport(params.jobId, "markdown");
                const blob = new Blob([repMd as any], { type: "text/markdown" });
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = `${params.jobId}.md`;
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

          <section className="rounded bg-white shadow p-4">
            <div className="flex justify-between items-center mb-2">
              <h2 className="font-semibold text-sm uppercase text-gray-600">Top Parameter Sets</h2>
              {report?.artifacts?.heatmap_html && (
                <a href={report.artifacts.heatmap_html} className="text-blue-600 underline text-sm">
                  Heatmap
                </a>
              )}
            </div>
            {topSection?.data ? (
              <pre className="text-xs bg-slate-50 p-2 rounded whitespace-pre-wrap">
                {JSON.stringify(topSection.data, null, 2)}
              </pre>
            ) : (
              <p className="text-sm text-gray-500">No top performers section found.</p>
            )}
          </section>

          {recommendations && (
            <section className="rounded bg-white shadow p-4">
              <h2 className="font-semibold text-sm uppercase text-gray-600">Recommendations</h2>
              <pre className="text-xs bg-slate-50 p-2 rounded whitespace-pre-wrap">
                {recommendations.content ?? "No recommendations text."}
              </pre>
            </section>
          )}
        </>
      )}
    </main>
  );
}
