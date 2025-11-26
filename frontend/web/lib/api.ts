import axios, { AxiosError } from "axios";
import type {
  BacktestRunInfo,
  BarPoint,
  ChartMeta,
  SummaryStats,
  TradeRow,
  LiveStatus,
} from "@/store/useChartStore";

const API_BASE = "http://localhost:8000";

export async function fetchChart(
  symbol: string,
  timeframe: string,
  runId?: string | null
): Promise<{
  bars: BarPoint[];
  meta: ChartMeta;
}> {
  const params = runId ? { run_id: runId } : undefined;
  const res = await axios.get(`${API_BASE}/api/chart/${symbol}/${timeframe}`, {
    params,
  });
  return {
    bars: res.data.bars as BarPoint[],
    meta: res.data.meta as ChartMeta,
  };
}

export async function fetchSummary(
  symbol: string,
  timeframe: string
): Promise<SummaryStats> {
  const url = `${API_BASE}/api/summary/${symbol}/${timeframe}`;
  const res = await axios.get(url);
  return res.data as SummaryStats;
}

export async function fetchBacktests(
  symbol: string,
  timeframe: string,
  strategy?: string
): Promise<BacktestRunInfo[]> {
  const params = strategy ? { strategy } : undefined;
  const res = await axios.get(
    `${API_BASE}/api/backtests/${symbol}/${timeframe}`,
    { params }
  );
  return res.data as BacktestRunInfo[];
}

export async function fetchTrades(runId: string): Promise<TradeRow[]> {
  const res = await axios.get(`${API_BASE}/api/trades/${runId}`);
  return res.data as TradeRow[];
}

export async function fetchLiveStatus(
  runId?: string | null
): Promise<LiveStatus | null> {
  try {
    const params = runId ? { run_id: runId } : undefined;
    const res = await axios.get(`${API_BASE}/api/live/status`, { params });
    return res.data as LiveStatus;
  } catch (err) {
    if (axios.isAxiosError(err) && err.response?.status === 404) {
      return null;
    }
    throw err;
  }
}

export async function runBacktest(
  symbol: string,
  timeframe: string,
  strategy: string
): Promise<{
  run_id: string;
  symbol: string;
  timeframe: string;
  strategy: string;
  metrics: Record<string, number>;
  trade_count: number;
}> {
  const res = await axios.post(`${API_BASE}/api/backtests/run`, {
    symbol,
    timeframe,
    strategy,
  });
  return res.data;
}
