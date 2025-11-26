import { create } from "zustand";

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
  equity: number;
  realized_pnl?: number | null;
  unrealized_pnl?: number | null;
  daily_realized_pnl?: number | null;
  open_positions: LivePosition[];
  risk_stats: Record<string, any>;
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
  isRunning: boolean;
  lastRunId?: string;
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
  setIsRunning: (flag: boolean) => void;
  setLastRunId: (id?: string) => void;
};

export const useChartStore = create<ChartState>((set) => ({
  symbol: "AIAUSDT",
  timeframe: "15m",
  strategies: ["rule", "ml", "trend_continuation", "sweep_reversal", "volatility_breakout"],
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
  isRunning: false,
  lastRunId: undefined,
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
        const parse = (t: string | undefined | null) =>
          t ? new Date(t).getTime() : 0;
        const ta = parse(a.entry_time ?? (a as any).timestamp ?? null);
        const tb = parse(b.entry_time ?? (b as any).timestamp ?? null);
        return ta - tb;
      }),
    }),
  setSelectedTradeId: (tradeId) => set({ selectedTradeId: tradeId }),
  setBars: (bars) => set({ bars }),
  setMeta: (meta) => set({ meta }),
  setSummary: (summary) => set({ summary }),
  setOverlays: (partial) =>
    set((state) => ({ overlays: { ...state.overlays, ...partial } })),
  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error }),
  setLiveStatus: (liveStatus) => set({ liveStatus }),
  setIsLiveLoading: (flag) => set({ isLiveLoading: flag }),
  setLiveError: (msg) => set({ liveError: msg }),
  setIsRunning: (flag) => set({ isRunning: flag }),
  setLastRunId: (id) => set({ lastRunId: id }),
}));
