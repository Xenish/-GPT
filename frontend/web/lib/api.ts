import axios, { AxiosError } from "axios";

const API_BASE = "http://localhost:8000";

export async function fetchChart(
  symbol: string,
  timeframe: string,
  runId?: string | null
): Promise<{
  bars: any[];
  meta: Record<string, any>;
}> {
  const params = runId ? { run_id: runId } : undefined;
  const res = await axios.get(`${API_BASE}/api/chart/${symbol}/${timeframe}`, {
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
  const url = `${API_BASE}/api/summary/${symbol}/${timeframe}`;
  const res = await axios.get(url);
  return res.data as Record<string, number>;
}

export async function fetchBacktests(
  symbol: string,
  timeframe: string,
  strategy?: string
): Promise<any[]> {
  const params = strategy ? { strategy } : undefined;
  const res = await axios.get(
    `${API_BASE}/api/backtests/${symbol}/${timeframe}`,
    { params }
  );
  return res.data as any[];
}

export async function fetchTrades(runId: string): Promise<any[]> {
  const res = await axios.get(`${API_BASE}/api/trades/${runId}`);
  return res.data as any[];
}

export async function fetchLiveStatus(
  runId?: string | null
): Promise<any | null> {
  try {
    const params = runId ? { run_id: runId } : undefined;
    const res = await axios.get(`${API_BASE}/api/live/status`, { params });
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
  const res = await axios.post(`${API_BASE}/api/backtests/run`, body);
  return res.data as {
    run_id: string;
    symbol: string;
    timeframe: string;
    strategy: string;
    metrics: Record<string, number>;
    trade_count: number;
  };
}

export async function getMeta(): Promise<{
  symbols: string[];
  timeframes: string[];
  strategies: string[];
  scenario_presets: string[];
}> {
  const res = await axios.get(`${API_BASE}/api/meta`);
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
  const res = await axios.post(`${API_BASE}/api/scenarios/run`, body);
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
  const res = await axios.get(`${API_BASE}/api/portfolio/backtests`);
  return res.data;
}

export async function getPortfolioEquity(
  runId: string
): Promise<{ time: number; portfolio_equity: number }[]> {
  const res = await axios.get(`${API_BASE}/api/portfolio/equity/${runId}`);
  return res.data;
}

export async function getLiveStatus(runId?: string) {
  const params = runId ? { run_id: runId } : undefined;
  const res = await axios.get(`${API_BASE}/api/live/status`, { params });
  return res.data;
}

export async function sendLiveControl(body: { command: string; run_id?: string }) {
  const res = await axios.post(`${API_BASE}/api/live/control`, body);
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
  const res = await axios.get(`${API_BASE}/api/scenarios/${symbol}/${timeframe}`);
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
  const res = await axios.get(`${API_BASE}/api/ml/models/${symbol}/${timeframe}`);
  return res.data as ModelInfo[];
}

export async function getFeatureImportance(modelId: string) {
  const res = await axios.get(`${API_BASE}/api/ml/models/${modelId}/importance`);
  const raw = (res.data ?? {}) as Record<string, number>;
  return Object.entries(raw).map(([name, value]) => ({
    name,
    value,
  }));
}
