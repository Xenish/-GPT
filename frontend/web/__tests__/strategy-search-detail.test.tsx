import { render, screen, waitFor } from "@testing-library/react";
import React from "react";
import StrategySearchDetailPage from "@/app/strategy-search/[jobId]/page";

jest.mock("@/lib/api", () => ({
  fetchStrategySearchReport: jest.fn().mockResolvedValue({
    job_id: "job1",
    strategy_id: "rule",
    symbol: "BTCUSDT",
    timeframe: "15m",
    metrics: { best_sharpe: 1.2, best_cum_return: 0.1 },
    sections: [{ title: "Top Performers", metrics: { best_sharpe: 1.2 } }],
    artifacts: { heatmap_html: "/heatmap.html" },
  }),
}));

describe("Strategy search detail page", () => {
  it("renders metrics and heatmap link", async () => {
    render(<StrategySearchDetailPage params={{ jobId: "job1" }} />);

    await waitFor(() => {
      expect(screen.getByText(/Strategy Search job1/i)).toBeInTheDocument();
    });
    expect(screen.getByText(/best_sharpe/i)).toBeInTheDocument();
    expect(screen.getByText(/1.2/)).toBeInTheDocument();
    expect(screen.getByText(/Heatmap/i)).toBeInTheDocument();
  });
});
