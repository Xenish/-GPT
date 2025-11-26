"use client";

import type { TradeRow } from "@/store/useChartStore";

type Props = {
  trades: TradeRow[];
  selectedTradeId: string | null;
  onSelect: (trade: TradeRow) => void;
};

export default function TradeTable({ trades, selectedTradeId, onSelect }: Props) {
  if (!trades.length) {
    return <p className="text-sm text-gray-500">No trades for this run.</p>;
  }

  return (
    <div className="overflow-auto border rounded">
      <table className="min-w-full text-sm">
        <thead className="bg-slate-100 text-gray-700">
          <tr>
            <th className="px-3 py-2 text-left">Entry</th>
            <th className="px-3 py-2 text-left">Exit</th>
            <th className="px-3 py-2 text-left">Side</th>
            <th className="px-3 py-2 text-right">PnL</th>
            <th className="px-3 py-2 text-right">Hold (min)</th>
          </tr>
        </thead>
        <tbody>
          {trades.map((trade) => {
            const entry = trade.entry_time
              ? new Date(trade.entry_time).toLocaleString()
              : "-";
            const exit = trade.exit_time
              ? new Date(trade.exit_time).toLocaleString()
              : "-";
            const holdMinutes =
              trade.entry_time && trade.exit_time
                ? Math.round(
                    (new Date(trade.exit_time).getTime() -
                      new Date(trade.entry_time).getTime()) /
                      60000
                  )
                : null;
            const isSelected = trade.trade_id === selectedTradeId;
            return (
              <tr
                key={trade.trade_id}
                className={`cursor-pointer ${
                  isSelected ? "bg-blue-50" : "hover:bg-slate-50"
                }`}
                onClick={() => onSelect(trade)}
              >
                <td className="px-3 py-2">{entry}</td>
                <td className="px-3 py-2">{exit}</td>
                <td className="px-3 py-2">{trade.side}</td>
                <td className="px-3 py-2 text-right">
                  {trade.pnl !== undefined && trade.pnl !== null
                    ? trade.pnl.toFixed(2)
                    : "-"}
                </td>
                <td className="px-3 py-2 text-right">
                  {holdMinutes !== null ? holdMinutes : "-"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
