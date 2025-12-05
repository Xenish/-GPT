import { render, screen, waitFor } from "@testing-library/react";
import React from "react";
import BacktestDetailPage from "@/app/backtests/[jobId]/page";

jest.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(),
}));

jest.mock("@/lib/api", () => ({
  fetchBacktestReport: jest.fn().mockResolvedValue({
    title: "Backtest report",
    run_id: "run1",
    strategy_id: "rule",
    symbol: "BTCUSDT",
    timeframe: "15m",
    metrics: { net_profit: 100, max_drawdown: -5 },
    artifacts: { trades_csv: "/trades.csv" },
    sections: [{ title: "Overview", metrics: { net_profit: 100, max_drawdown: -5 } }],
  }),
  fetchTrades: jest.fn().mockResolvedValue([{ id: 1, pnl: 10 }]),
}));

// Mock fetch for CSV download in component
const mockFetch = jest.fn().mockResolvedValue({
  ok: true,
  text: () => Promise.resolve("id,pnl\n1,10"),
});
(global as any).fetch = mockFetch;

describe("Backtest detail page", () => {
  it("renders metrics and trades", async () => {
    render(<BacktestDetailPage params={{ jobId: "run1" }} />);

    await waitFor(() => {
      expect(screen.getByText(/Backtest run1/i)).toBeInTheDocument();
    });
    expect(screen.getByText(/net_profit/i)).toBeInTheDocument();
    expect(screen.getByText(/max_drawdown/i)).toBeInTheDocument();
    expect(screen.getByText(/run1/i)).toBeInTheDocument();
  });
});
