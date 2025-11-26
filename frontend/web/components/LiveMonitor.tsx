"use client";

import { useEffect } from "react";
import { fetchLiveStatus } from "@/lib/api";
import { useChartStore } from "@/store/useChartStore";

export default function LiveMonitor() {
  const {
    liveStatus,
    isLiveLoading,
    liveError,
    setLiveStatus,
    setIsLiveLoading,
    setLiveError,
  } = useChartStore((state) => ({
    liveStatus: state.liveStatus,
    isLiveLoading: state.isLiveLoading,
    liveError: state.liveError,
    setLiveStatus: state.setLiveStatus,
    setIsLiveLoading: state.setIsLiveLoading,
    setLiveError: state.setLiveError,
  }));

  useEffect(() => {
    let active = true;
    async function poll() {
      try {
        setIsLiveLoading(true);
        const status = await fetchLiveStatus();
        if (!active) return;
        setLiveStatus(status);
        setLiveError(null);
      } catch (err: any) {
        console.error(err);
        if (!active) return;
        if (err?.response?.status === 404) {
          setLiveStatus(null);
          setLiveError(null);
        } else {
          setLiveError(err?.message ?? "Failed to load live status");
        }
      } finally {
        if (active) {
          setIsLiveLoading(false);
        }
      }
    }
    poll();
    const id = window.setInterval(poll, 5000);
    return () => {
      active = false;
      window.clearInterval(id);
    };
  }, [setIsLiveLoading, setLiveError, setLiveStatus]);

  if (isLiveLoading && !liveStatus) {
    return (
      <div className="border rounded p-4 bg-white">
        <h2 className="font-semibold mb-2 text-sm uppercase tracking-wider text-gray-500">
          Live Monitor
        </h2>
        <p className="text-sm text-gray-500">Loading live status...</p>
      </div>
    );
  }

  return (
    <div className="border rounded p-4 bg-white">
      <h2 className="font-semibold mb-2 text-sm uppercase tracking-wider text-gray-500">
        Live Monitor
      </h2>
      {liveError && (
        <p className="text-sm text-red-600 mb-2">{liveError}</p>
      )}
      {liveStatus ? (
        <>
          <div className="text-sm space-y-1">
            <p>
              Run: <strong>{liveStatus.run_id}</strong>
            </p>
            <p>
              {liveStatus.strategy} | {liveStatus.symbol}/{liveStatus.timeframe}
            </p>
            <p>
              Equity:{" "}
              <strong>{liveStatus.equity?.toFixed(2) ?? "Unknown"}</strong>
            </p>
            <p>
              Realized PnL:{" "}
              <strong>
                {liveStatus.realized_pnl !== undefined &&
                liveStatus.realized_pnl !== null
                  ? liveStatus.realized_pnl.toFixed(2)
                  : "—"}
              </strong>
            </p>
            <p>
              Daily PnL:{" "}
              <strong>
                {liveStatus.daily_realized_pnl !== undefined &&
                liveStatus.daily_realized_pnl !== null
                  ? liveStatus.daily_realized_pnl.toFixed(2)
                  : "—"}
              </strong>
            </p>
            <p>
              Open positions: <strong>{liveStatus.open_positions.length}</strong>
            </p>
            <p className="text-xs text-gray-500">
              Last bar:{" "}
              {liveStatus.last_bar_time
                ? new Date(liveStatus.last_bar_time).toLocaleString()
                : "—"}
            </p>
          </div>

          {liveStatus.open_positions.length > 0 && (
            <div className="mt-3">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-left text-gray-500">
                    <th className="pr-2">Symbol</th>
                    <th className="pr-2">Side</th>
                    <th className="pr-2">Qty</th>
                    <th className="pr-2">PnL</th>
                  </tr>
                </thead>
                <tbody>
                  {liveStatus.open_positions.map((pos) => (
                    <tr key={pos.id}>
                      <td className="pr-2">{pos.symbol}</td>
                      <td className="pr-2">{pos.side}</td>
                      <td className="pr-2">{pos.qty}</td>
                      <td className="pr-2">
                        {pos.pnl !== undefined && pos.pnl !== null
                          ? pos.pnl.toFixed(2)
                          : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      ) : (
        <p className="text-sm text-gray-500">
          No active live session detected.
        </p>
      )}
    </div>
  );
}
