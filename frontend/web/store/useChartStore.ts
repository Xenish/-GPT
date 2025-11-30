import { create } from "zustand";

import {
  fetchBacktests,
  fetchChart,
  fetchTrades,
  runBacktest as runBacktestApi,
  getMeta,
  getPortfolioBacktests,
  getPortfolioEquity,
  getLiveStatus,
  sendLiveControl,
  getScenarioResults,
  getMlModels,
  getFeatureImportance,
} from "@/lib/api";
import type { ScenarioResult, ModelInfo, FeatureImportanceItem, MLTarget } from "@/lib/api";

export type BarPoint = {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
  rule_entry?: number;
  rule_exit?: number;
  ml_long_proba?: number;
  ms_trend?: number;
  ms_chop?: number;
  ms_hh_ll_trend?: number;
  fvg_up?: number;
  fvg_down?: number;
  trade_entries?: string[];
  trade_exits?: string[];
};

export type ChartMeta = {
  pipeline_version?: string;
  features_preset?: string;
  last_updated?: string;
  run_id?: string | null;
  strategy?: string | null;
};

export type SummaryStats = {
  final_equity?: number;
  cum_return?: number;
  trade_count?: number;
  win_rate?: number;
};

export type BacktestRunInfo = {
  run_id: string;
  strategy: string;
  symbol: string;
  timeframe: string;
  start?: string | null;
  end?: string | null;
  metrics: Record<string, number | null>;
};

export type TradeRow = {
  trade_id: string;
  entry_time: string;
  exit_time?: string | null;
  side: string;
  qty: number;
  entry_price: number;
  exit_price?: number | null;
  pnl?: number | null;
  reason?: string | null;
};

export type LivePosition = {
  id: string;
  symbol: string;
  side: string;
  qty: number;
  entry_price: number;
  current_price?: number | null;
  pnl?: number | null;
  entry_time?: string | null;
};

export type LiveStatus = {
  run_id: string;
  symbol: string;
  timeframe: string;
  strategy: string;
  start_time?: string | null;
  last_bar_time?: string | null;
  last_bar_time_ts?: number | null;
  mode?: string | null;
  equity: number;
  realized_pnl?: number | null;
  unrealized_pnl?: number | null;
  daily_realized_pnl?: number | null;
  daily_unrealized_pnl?: number | null;
  open_positions: LivePosition[];
  risk_stats: Record<string, any>;
  data_source?: string | null;
  stale_data_seconds?: number | null;
  ws_reconnect_count?: number | null;
  last_orders?: Record<string, any>[];
  timestamp?: number | null;
};

type ChartState = {
  symbol: string;
  timeframe: string;
  strategies: string[];
  selectedStrategy: string;
  backtests: BacktestRunInfo[];
  selectedRunId: string | null;
  trades: TradeRow[];
  selectedTradeId: string | null;
  bars: BarPoint[];
  meta: ChartMeta | null;
  summary: SummaryStats | null;
  liveStatus: LiveStatus | null;
  isLiveLoading: boolean;
  liveError: string | null;
  isSendingLiveCommand: boolean;
  isRunningBacktest: boolean;
  lastError: string | null;
  lastRunId?: string;
  lastRunMetrics?: { trade_count?: number; cum_return?: number; sharpe?: number };
  availableSymbols: string[];
  availableTimeframes: string[];
  availableStrategies: string[];
  lookbackDaysByTf: Record<string, number> | null;
  mlTargets: MLTarget[];
  ruleParams: {
    ms_trend_min?: number;
    ms_trend_max?: number;
    use_ms_chop_filter?: boolean;
  };
  scenarioResults: ScenarioResult[];
  selectedScenarioId: string | null;
  isLoadingScenarios: boolean;
  scenarioError: string | null;
  mlModels: ModelInfo[];
  selectedModelId: string | null;
  featureImportance: FeatureImportanceItem[];
  isLoadingMlModels: boolean;
  isLoadingFeatureImportance: boolean;
  mlError: string | null;
  portfolioRuns: {
    run_id: string;
    symbols: string[];
    timeframe: string;
    start?: string | null;
    end?: string | null;
    metrics: Record<string, number | null>;
  }[];
  selectedPortfolioRunId: string | null;
  portfolioEquity: { time: number; portfolio_equity: number }[];
  isLoadingPortfolioEquity: boolean;
  portfolioError: string | null;
  overlays: {
    showRuleSignals: boolean;
    showMicrostructure: boolean;
    showMarketStructure: boolean;
    showMLProba: boolean;
  };
  loading: boolean;
  error: string | null;
  setSymbol: (symbol: string) => void;
  setTimeframe: (tf: string) => void;
  setSelectedStrategy: (strategy: string) => void;
  setBacktests: (runs: BacktestRunInfo[]) => void;
  setSelectedRunId: (runId: string | null) => void;
  setTrades: (trades: TradeRow[]) => void;
  setSelectedTradeId: (tradeId: string | null) => void;
  setBars: (bars: BarPoint[]) => void;
  setMeta: (meta: ChartMeta | null) => void;
  setSummary: (summary: SummaryStats | null) => void;
  setOverlays: (partial: Partial<ChartState["overlays"]>) => void;
  setLoading: (loading: boolean) => void;
  setError: (msg: string | null) => void;
  setLiveStatus: (status: LiveStatus | null) => void;
  setIsLiveLoading: (flag: boolean) => void;
  setLiveError: (msg: string | null) => void;
  setLastRunId: (id?: string) => void;
  initMeta: () => Promise<void>;
  setRuleParams: (partial: Partial<ChartState["ruleParams"]>) => void;
  fetchPortfolioRuns: () => Promise<void>;
  fetchPortfolioEquity: (runId: string) => Promise<void>;
  fetchLiveStatus: () => Promise<void>;
  sendLiveCommand: (command: "pause" | "resume" | "stop" | "flatten") => Promise<void>;
  fetchScenarioResults: (symbol?: string, timeframe?: string) => Promise<void>;
  setSelectedScenarioId: (id: string | null) => void;
  fetchMlModels: (symbol?: string, timeframe?: string) => Promise<void>;
  selectModel: (modelId: string) => Promise<void>;
  runBacktest: () => Promise<void>;
};

export const useChartStore = create<ChartState>((set, get) => ({
  symbol: "AIAUSDT",
  timeframe: "15m",
  strategies: ["rule", "ml"],
  selectedStrategy: "rule",
  backtests: [],
  selectedRunId: null,
  trades: [],
  selectedTradeId: null,
  bars: [],
  meta: null,
  summary: null,
  liveStatus: null,
  isLiveLoading: false,
  liveError: null,
  isSendingLiveCommand: false,
  isRunningBacktest: false,
  lastError: null,
  lastRunId: undefined,
  lastRunMetrics: undefined,
  availableSymbols: [],
  availableTimeframes: [],
  availableStrategies: [],
  lookbackDaysByTf: null,
  mlTargets: [],
  ruleParams: {},
  scenarioResults: [],
  selectedScenarioId: null,
  isLoadingScenarios: false,
  scenarioError: null,
  mlModels: [],
  selectedModelId: null,
  featureImportance: [],
  isLoadingMlModels: false,
  isLoadingFeatureImportance: false,
  mlError: null,
  portfolioRuns: [],
  selectedPortfolioRunId: null,
  portfolioEquity: [],
  isLoadingPortfolioEquity: false,
  portfolioError: null,
  overlays: {
    showRuleSignals: true,
    showMicrostructure: false,
    showMarketStructure: false,
    showMLProba: false,
  },
  loading: false,
  error: null,
  setSymbol: (symbol) => set({ symbol }),
  setTimeframe: (timeframe) => set({ timeframe }),
  setSelectedStrategy: (strategy) => set({ selectedStrategy: strategy }),
  setBacktests: (runs) => set({ backtests: runs }),
  setSelectedRunId: (runId) => set({ selectedRunId: runId }),
  setTrades: (trades) =>
    set({
      trades: [...trades].sort((a, b) => {
        const parse = (t: string | undefined | null) => (t ? new Date(t).getTime() : 0);
        const ta = parse(a.entry_time ?? (a as any).timestamp ?? null);
        const tb = parse(b.entry_time ?? (b as any).timestamp ?? null);
        return ta - tb;
      }),
    }),
  setSelectedTradeId: (tradeId) => set({ selectedTradeId: tradeId }),
  setBars: (bars) => set({ bars }),
  setMeta: (meta) => set({ meta }),
  setSummary: (summary) => set({ summary }),
  setOverlays: (partial) => set((state) => ({ overlays: { ...state.overlays, ...partial } })),
  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error }),
  setLiveStatus: (liveStatus) => set({ liveStatus }),
  setIsLiveLoading: (flag) => set({ isLiveLoading: flag }),
  setLiveError: (msg) => set({ liveError: msg }),
  setLastRunId: (id) => set({ lastRunId: id }),
  initMeta: async () => {
    const data = await getMeta();
    set((state) => {
      const symbol = data.symbols.includes(state.symbol) ? state.symbol : data.symbols[0];
      const timeframe = data.timeframes.includes(state.timeframe) ? state.timeframe : data.timeframes[0];
      const strategy = data.strategies.includes(state.selectedStrategy)
        ? state.selectedStrategy
        : data.strategies[0];
      return {
        availableSymbols: data.symbols,
        availableTimeframes: data.timeframes,
        availableStrategies: data.strategies,
        lookbackDaysByTf: data.lookback_days || null,
        mlTargets: data.ml_targets || [],
        symbol,
        timeframe,
        strategies: data.strategies,
        selectedStrategy: strategy,
      };
    });
  },
  setRuleParams: (partial) =>
    set((state) => ({
      ruleParams: { ...state.ruleParams, ...partial },
    })),
  fetchScenarioResults: async (symbolOverride, timeframeOverride) => {
    const state = get();
    const symbol = symbolOverride ?? state.symbol;
    const timeframe = timeframeOverride ?? state.timeframe;
    set({ isLoadingScenarios: true, scenarioError: null });
    try {
      const results = await getScenarioResults(symbol, timeframe);
      set({
        scenarioResults: results,
        selectedScenarioId: results.length > 0 ? results[0].scenario_id : null,
      });
    } catch (err: any) {
      const msg = err?.response?.data?.detail ?? err?.message ?? "Failed to load scenario results";
      set({ scenarioError: msg, scenarioResults: [], selectedScenarioId: null });
    } finally {
      set({ isLoadingScenarios: false });
    }
  },
  setSelectedScenarioId: (id) => set({ selectedScenarioId: id }),
  fetchMlModels: async (symbolOverride, timeframeOverride) => {
    const state = get();
    const symbol = symbolOverride ?? state.symbol;
    const timeframe = timeframeOverride ?? state.timeframe;
    set({ isLoadingMlModels: true, mlError: null });
    try {
      const models = await getMlModels(symbol, timeframe);
      set({
        mlModels: models,
        selectedModelId: models.length > 0 ? models[0].model_id : null,
        featureImportance: [],
      });
      if (models.length > 0) {
        await get().selectModel(models[0].model_id);
      }
    } catch (err: any) {
      const msg = err?.response?.data?.detail ?? err?.message ?? "Failed to load ML models";
      set({
        mlError: msg,
        mlModels: [],
        selectedModelId: null,
        featureImportance: [],
      });
    } finally {
      set({ isLoadingMlModels: false });
    }
  },
  selectModel: async (modelId: string) => {
    if (!modelId) {
      set({ selectedModelId: null, featureImportance: [] });
      return;
    }
    set({
      selectedModelId: modelId,
      isLoadingFeatureImportance: true,
      mlError: null,
      featureImportance: [],
    });
    try {
      const fi = await getFeatureImportance(modelId);
      const total = fi.reduce((acc, item) => acc + item.value, 0) || 1;
      const normalized = fi.map((item) => ({
        name: item.name,
        value: item.value / total,
      }));
      set({ featureImportance: normalized });
    } catch (err: any) {
      const msg =
        err?.response?.data?.detail ?? err?.message ?? "Failed to load feature importance";
      set({ mlError: msg, featureImportance: [] });
    } finally {
      set({ isLoadingFeatureImportance: false });
    }
  },
  fetchPortfolioRuns: async () => {
    const runs = await getPortfolioBacktests();
    set((state) => ({
      portfolioRuns: runs,
      selectedPortfolioRunId: state.selectedPortfolioRunId ?? (runs[0]?.run_id ?? null),
    }));
  },
  fetchPortfolioEquity: async (runId: string) => {
    set({ isLoadingPortfolioEquity: true, portfolioError: null });
    try {
      const points = await getPortfolioEquity(runId);
      set({ portfolioEquity: points, selectedPortfolioRunId: runId });
    } catch (err: any) {
      const msg = err?.response?.data?.detail ?? err?.message ?? "Failed to load portfolio equity";
      set({ portfolioError: msg });
    } finally {
      set({ isLoadingPortfolioEquity: false });
    }
  },
  fetchLiveStatus: async () => {
    try {
      const status = await getLiveStatus();
      set({ liveStatus: status, liveError: null });
    } catch (err: any) {
      const msg = err?.response?.data?.detail ?? err?.message ?? "Failed to load live status";
      set({ liveError: msg });
    }
  },
  sendLiveCommand: async (command) => {
    set({ isSendingLiveCommand: true });
    try {
      await sendLiveControl({ command });
      const status = await getLiveStatus();
      set({ liveStatus: status, liveError: null });
    } catch (err: any) {
      const msg = err?.response?.data?.detail ?? err?.message ?? "Failed to send live command";
      set({ liveError: msg });
    } finally {
      set({ isSendingLiveCommand: false });
    }
  },
  runBacktest: async () => {
    const { symbol, timeframe, selectedStrategy, ruleParams } = get();
    set({ isRunningBacktest: true, lastError: null });
    try {
      const res = await runBacktestApi({
        symbol,
        timeframe,
        strategy: selectedStrategy,
        strategy_params: selectedStrategy === "rule" ? ruleParams : undefined,
      });
      set({
        selectedRunId: res.run_id,
        lastRunMetrics: {
          trade_count: res.trade_count,
          cum_return: res.metrics?.cum_return,
          sharpe: res.metrics?.sharpe,
        },
      });

      const runs = await fetchBacktests(symbol, timeframe, selectedStrategy);
      set({ backtests: runs });

      const { bars, meta } = await fetchChart(symbol, timeframe, res.run_id);
      set({ bars, meta });

      const trades = await fetchTrades(res.run_id);
      set({ trades });
    } catch (err: any) {
      const detailRaw = err?.response?.data?.detail ?? err?.message ?? "Backtest failed";
      const mapError = (msg: string) => {
        if (!msg) return "Backtest failed";
        if (msg.includes("No valid ML model found")) {
          return "ML modeli bulunamadı, önce train scriptini çalıştırın.";
        }
        if (msg.includes("Feature mismatch")) {
          return "Modelin feature set'i değişmiş; modeli yeniden eğitin.";
        }
        if (msg.includes("artifacts missing")) {
          return "Model dosyaları eksik; registry'yi temizleyip yeniden eğitin.";
        }
        return msg;
      };
      const detail = mapError(detailRaw);
      set({ lastError: detail });
    } finally {
      set({ isRunningBacktest: false });
    }
  },
}));
