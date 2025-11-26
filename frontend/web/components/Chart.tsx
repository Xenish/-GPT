"use client";

import { useEffect, useRef } from "react";
import {
  CandlestickData,
  IChartApi,
  ISeriesApi,
  SeriesMarker,
  Time,
  createChart,
} from "lightweight-charts";
import type { BarPoint, TradeRow } from "@/store/useChartStore";

type Props = {
  bars: BarPoint[];
  overlays: {
    showRuleSignals: boolean;
    showMicrostructure: boolean;
    showMarketStructure: boolean;
    showMLProba: boolean;
  };
  trades: TradeRow[];
  selectedTradeId: string | null;
};

export default function Chart({
  bars,
  overlays,
  trades,
  selectedTradeId,
}: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);

  useEffect(() => {
    if (!containerRef.current || chartRef.current) {
      return;
    }

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height: 500,
      layout: {
        background: { color: "white" },
        textColor: "black",
      },
      rightPriceScale: { borderVisible: false },
      timeScale: { borderVisible: false },
      grid: {
        horzLines: { color: "#e2e8f0" },
        vertLines: { color: "#e2e8f0" },
      },
    });

    const candleSeries = chart.addCandlestickSeries();
    chartRef.current = chart;
    candleSeriesRef.current = candleSeries;

    const handleResize = () => {
      if (containerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: containerRef.current.clientWidth,
        });
      }
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
      chartRef.current = null;
      candleSeriesRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!candleSeriesRef.current) {
      return;
    }

    const data: CandlestickData[] = bars.map((bar) => ({
      time: bar.time as any,
      open: bar.open,
      high: bar.high,
      low: bar.low,
      close: bar.close,
    }));
    candleSeriesRef.current.setData(data);

    const markers: SeriesMarker<Time>[] = [];

    if (overlays.showRuleSignals) {
      bars.forEach((bar) => {
        if (bar.rule_entry) {
          markers.push({
            time: bar.time as any,
            position: "belowBar",
            color: "#16a34a",
            shape: "arrowUp",
            text: "ENTRY",
          });
        }
        if (bar.rule_exit) {
          markers.push({
            time: bar.time as any,
            position: "aboveBar",
            color: "#dc2626",
            shape: "arrowDown",
            text: "EXIT",
          });
        }
      });
    }

    if (overlays.showMicrostructure) {
      bars.forEach((bar) => {
        if (bar.ms_trend === 1) {
          markers.push({
            time: bar.time as any,
            position: "belowBar",
            color: "#1d4ed8",
            shape: "circle",
            text: "MS+",
          });
        } else if (bar.ms_trend === -1) {
          markers.push({
            time: bar.time as any,
            position: "belowBar",
            color: "#f97316",
            shape: "circle",
            text: "MS-",
          });
        }
      });
    }

    if (overlays.showMarketStructure) {
      bars.forEach((bar) => {
        if (bar.fvg_up) {
          markers.push({
            time: bar.time as any,
            position: "belowBar",
            color: "#0ea5e9",
            shape: "square",
            text: "FVG↑",
          });
        }
        if (bar.fvg_down) {
          markers.push({
            time: bar.time as any,
            position: "aboveBar",
            color: "#be123c",
            shape: "square",
            text: "FVG↓",
          });
        }
      });
    }

    if (overlays.showMLProba) {
      bars.forEach((bar) => {
        if (typeof bar.ml_long_proba === "number") {
          markers.push({
            time: bar.time as any,
            position: "belowBar",
            color: "#10b981",
            shape: "square",
            text: (bar.ml_long_proba * 100).toFixed(0) + "%",
          });
        }
      });
    }

    if (trades.length) {
      bars.forEach((bar) => {
        const entryMarkers = bar.trade_entries || [];
        entryMarkers.forEach((tradeId) => {
          const selected = tradeId === selectedTradeId;
          markers.push({
            time: bar.time as any,
            position: "belowBar",
            color: selected ? "#9333ea" : "#6d28d9",
            shape: selected ? "arrowUp" : "circle",
            text: selected ? "TRADE IN" : "IN",
          });
        });
        const exitMarkers = bar.trade_exits || [];
        exitMarkers.forEach((tradeId) => {
          const selected = tradeId === selectedTradeId;
          markers.push({
            time: bar.time as any,
            position: "aboveBar",
            color: selected ? "#db2777" : "#be185d",
            shape: selected ? "arrowDown" : "circle",
            text: selected ? "TRADE OUT" : "OUT",
          });
        });
      });
    }

    const cleaned = markers.filter(
      (marker) => marker.time !== undefined && marker.time !== null
    );

    const getTs = (time: Time) => {
      if (typeof time === "number") {
        return time;
      }
      if (typeof time === "string") {
        return new Date(time).getTime() / 1000;
      }
      if (typeof time === "object" && time !== null && "year" in time) {
        const obj = time as { year: number; month: number; day: number };
        return new Date(obj.year, obj.month - 1, obj.day).getTime() / 1000;
      }
      return 0;
    };

    cleaned.sort((a, b) => getTs(a.time) - getTs(b.time));

    candleSeriesRef.current.setMarkers(cleaned);
  }, [bars, overlays, trades, selectedTradeId]);

  useEffect(() => {
    if (!chartRef.current || !selectedTradeId || !trades.length) {
      return;
    }
    const trade = trades.find((t) => t.trade_id === selectedTradeId);
    if (!trade || !trade.entry_time) {
      return;
    }
    const entrySecs = new Date(trade.entry_time).getTime() / 1000;
    const exitSecs = trade.exit_time
      ? new Date(trade.exit_time).getTime() / 1000
      : entrySecs + 3600;
    const buffer = 3 * 15 * 60; // 3 bars buffer
    chartRef.current.timeScale().setVisibleRange({
      from: entrySecs - buffer,
      to: exitSecs + buffer,
    });
  }, [selectedTradeId, trades]);

  return <div ref={containerRef} className="w-full h-[500px]" />;
}
