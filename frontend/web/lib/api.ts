import axios, { AxiosError } from "axios";

function getApiBaseUrl() {
  const env = process.env.NEXT_PUBLIC_API_BASE_URL;
  if (env && env.trim().length > 0) {
    return env.replace(/\/+$/, "");
  }
  return "http://localhost:8000";
}

const apiBase = getApiBaseUrl();
const api = axios.create({
  baseURL: `${apiBase}/api`,
});

export async function fetchChart(
  symbol: string,
  timeframe: string,
  runId?: string | null
): Promise<{
  bars: any[];
  meta: Record<string, any>;
}> {
  const params = runId ? { run_id: runId } : undefined;
  const res = await api.get(`/chart/${symbol}/${timeframe}`, {
    params,
  });
  return {
    bars: res.data.bars as any[],
    meta: res.data.meta as Record<string, any>,
  };
}

export async function fetchSummary(
  symbol: string,
  timeframe: string
): Promise<Record<string, number>> {
  const res = await api.get(`/summary/${symbol}/${timeframe}`);
  return res.data as Record<string, number>;
}

export async function fetchBacktests(
  symbol: string,
  timeframe: string,
  strategy?: string
): Promise<any[]> {
  const params = strategy ? { strategy } : undefined;
  const res = await api.get(`/backtests/${symbol}/${timeframe}`, { params });
  return res.data as any[];
}

export async function fetchTrades(runId: string): Promise<any[]> {
  const res = await api.get(`/trades/${runId}`);
  return res.data as any[];
}

export async function fetchLiveStatus(
  runId?: string | null
): Promise<any | null> {
  try {
    const params = runId ? { run_id: runId } : undefined;
    const res = await api.get(`/live/status`, { params });
    return res.data as any;
  } catch (err) {
    if (axios.isAxiosError(err) && err.response?.status === 404) {
      return null;
    }
    throw err;
  }
}

export async function runBacktest(
  body: {
    symbol: string;
    timeframe: string;
    strategy: string;
    strategy_params?: Record<string, any>;
  }
): Promise<{
  run_id: string;
  symbol: string;
  timeframe: string;
  strategy: string;
  metrics: Record<string, number>;
  trade_count: number;
}> {
  const res = await api.post(`/backtests/run`, body);
  return res.data as {
    run_id: string;
    symbol: string;
    timeframe: string;
    strategy: string;
    metrics: Record<string, number>;
    trade_count: number;
  };
}

export type MLTarget = {
  symbol: string;
  timeframe: string;
};

export async function getMeta(): Promise<{
  symbols: string[];
  timeframes: string[];
  strategies: string[];
  scenario_presets: string[];
  lookback_days?: Record<string, number>;
  default_lookback_days?: number;
  ml_targets?: MLTarget[];
}> {
  const res = await api.get(`/meta`);
  return res.data;
}

export async function runScenarios(body: {
  symbol: string;
  timeframe: string;
  preset_name: string;
}): Promise<{
  preset_name: string;
  rows: {
    label: string;
    strategy: string;
    cum_return: number | null;
    sharpe: number | null;
    trade_count: number | null;
  }[];
}> {
  const res = await api.post(`/scenarios/run`, body);
  return res.data;
}

export async function getPortfolioBacktests(): Promise<
  {
    run_id: string;
    symbols: string[];
    timeframe: string;
    start?: string;
    end?: string;
    metrics: Record<string, number | null>;
  }[]
> {
  const res = await api.get(`/portfolio/backtests`);
  return res.data;
}

export async function getPortfolioEquity(
  runId: string
): Promise<{ time: number; portfolio_equity: number }[]> {
  const res = await api.get(`/portfolio/equity/${runId}`);
  return res.data;
}

export async function getLiveStatus(runId?: string) {
  const params = runId ? { run_id: runId } : undefined;
  const res = await api.get(`/live/status`, { params });
  return res.data;
}

export async function sendLiveControl(body: { command: string; run_id?: string }) {
  const res = await api.post(`/live/control`, body);
  return res.data;
}

export type ScenarioResult = {
  scenario_id: string;
  label?: string | null;
  symbol: string;
  timeframe: string;
  strategy: string;
  params: Record<string, any>;
  metrics: Record<string, number>;
};

export async function getScenarioResults(symbol: string, timeframe: string) {
  const res = await api.get(`/scenarios/${symbol}/${timeframe}`);
  return res.data as ScenarioResult[];
}

export type ModelInfo = {
  model_id: string;
  symbol: string;
  timeframe: string;
  model_type: string;
  created_at: string;
  metrics: Record<string, number>;
};

export type FeatureImportanceItem = {
  name: string;
  value: number;
};

export async function getMlModels(symbol: string, timeframe: string) {
  const res = await api.get(`/ml/models/${symbol}/${timeframe}`);
  return res.data as ModelInfo[];
}

export async function getFeatureImportance(modelId: string) {
  const res = await api.get(`/ml/models/${modelId}/importance`);
  const raw = (res.data ?? {}) as Record<string, number>;
  return Object.entries(raw).map(([name, value]) => ({
    name,
    value,
  }));
}
