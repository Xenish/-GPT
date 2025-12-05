import { render, screen, waitFor } from "@testing-library/react";
import React from "react";
import LiveDashboardPage from "@/app/live/page";

jest.mock("@/lib/api", () => ({
  fetchLiveStatus: jest.fn().mockResolvedValue({
    run_id: "live1",
    mode: "live",
    symbol: "BTCUSDT",
    timeframe: "15m",
    equity_now: 11000,
    daily_pnl: 100,
    max_intraday_dd: -50,
    heartbeat_age_sec: 5,
    open_positions: [
      { symbol: "BTCUSDT", side: "LONG", qty: 0.1, entry_price: 100, current_price: 105, pnl: 50 },
    ],
  }),
  fetchLiveReport: jest.fn().mockResolvedValue({
    artifacts: {},
    sections: [
      {
        title: "Recent Trades",
        data: { trades: [{ symbol: "BTCUSDT", pnl: 10 }] },
      },
    ],
  }),
}));

describe("Live dashboard page", () => {
  it("renders positions, trades, and risk metrics", async () => {
    render(<LiveDashboardPage />);

    await waitFor(() => {
      expect(screen.getByText(/Live Dashboard/i)).toBeInTheDocument();
    });

    expect(screen.getByText(/Open Positions/i)).toBeInTheDocument();
    expect(screen.getByText(/BTCUSDT/)).toBeInTheDocument();
    expect(screen.getByText(/Trades/i)).toBeInTheDocument();
    expect(screen.getByText(/Risk/i)).toBeInTheDocument();
    expect(screen.getByText(/daily pnl/i)).toBeInTheDocument();
  });
});
