import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import StrategySearchPage from "@/app/strategy-search/page";
import React from "react";

jest.mock("@/lib/api", () => ({
  fetchStrategySearchReport: jest.fn().mockResolvedValue({
    job_id: "job1",
    strategy_id: "rule",
    symbol: "BTCUSDT",
    timeframe: "15m",
    metrics: { best_sharpe: 1.2 },
  }),
}));

describe("Strategy search list page", () => {
  it("renders jobs after loading", async () => {
    render(<StrategySearchPage />);

    const input = screen.getByPlaceholderText(/job_id/i);
    fireEvent.change(input, { target: { value: "job1" } });
    fireEvent.click(screen.getAllByText(/Load/i)[0]);

    await waitFor(() => {
      expect(screen.getByText("job1")).toBeInTheDocument();
      expect(screen.getByText("rule")).toBeInTheDocument();
    });
  });
});
