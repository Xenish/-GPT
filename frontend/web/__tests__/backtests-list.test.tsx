import { render, screen, waitFor } from "@testing-library/react";
import BacktestsPage from "@/app/backtests/page";
import React from "react";

jest.mock("@/lib/api", () => ({
  fetchBacktests: jest.fn().mockResolvedValue([
    { run_id: "run1", strategy: "rule", symbol: "BTCUSDT", timeframe: "15m", metrics: { sharpe: 1.2, cum_return: 0.1 } },
    { run_id: "run2", strategy: "ml", symbol: "AIAUSDT", timeframe: "15m", metrics: { sharpe: 0.8, cum_return: 0.05 } },
  ]),
}));

describe("Backtests list page", () => {
  it("renders backtest rows after loading", async () => {
    render(<BacktestsPage />);

    expect(screen.getByText(/Loading.../i)).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText("run1")).toBeInTheDocument();
      expect(screen.getByText("run2")).toBeInTheDocument();
    });
    expect(screen.getAllByRole("link", { name: /run/i }).length).toBeGreaterThan(0);
  });
});
