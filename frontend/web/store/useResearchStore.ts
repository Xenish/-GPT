import { create } from "zustand";

// ============================================================================
// Types
// ============================================================================

export type StrategySearchJobRequest = {
  job_name: string;
  strategy_name: string;
  symbol: string;
  timeframe: string;
  mode?: "random" | "grid";
  n_samples?: number;
  fixed_params?: Record<string, any>;
  search_space_override?: Record<string, any>;
  random_seed?: number | null;
};

export type StrategySearchJobResponse = {
  job_id: string;
  status: string;
  message: string;
  results_path?: string | null;
};

export type JobStatusResponse = {
  job_id: string;
  status: string;
  created_at?: string | null;
  completed_at?: string | null;
  n_results: number;
  results_available: boolean;
};

export type ScenarioRequest = {
  label?: string;
  symbol?: string | null;
  timeframe?: string | null;
  strategy: string;
  params?: Record<string, any>;
};

export type RunScenariosRequest = {
  base_symbol: string;
  base_timeframe: string;
  scenarios: ScenarioRequest[];
};

export type ScenarioResultRow = {
  label: string;
  strategy: string;
  symbol: string;
  timeframe: string;
  cum_return: number | null;
  sharpe: number | null;
  max_dd: number | null;
  trade_count: number | null;
  win_rate: number | null;
};

export type RunScenariosResponse = {
  n_scenarios: number;
  results: ScenarioResultRow[];
};

export type ResearchHealthStatus = {
  service: string;
  status: string;
  job_limiter?: {
    max_jobs: number;
    active_jobs: number;
    available_slots: number;
  };
};

// ============================================================================
// Store State
// ============================================================================

type ResearchState = {
  // Strategy Search Jobs
  jobs: JobStatusResponse[];
  selectedJobId: string | null;
  isLoadingJobs: boolean;
  isCreatingJob: boolean;
  jobError: string | null;

  // Scenario Analysis
  scenarioResults: ScenarioResultRow[];
  isRunningScenarios: boolean;
  scenarioError: string | null;

  // Research Service Health
  healthStatus: ResearchHealthStatus | null;
  isLoadingHealth: boolean;
  healthError: string | null;

  // Actions
  fetchJobs: () => Promise<void>;
  fetchJobStatus: (jobId: string) => Promise<JobStatusResponse | null>;
  createSearchJob: (request: StrategySearchJobRequest) => Promise<StrategySearchJobResponse | null>;
  setSelectedJobId: (jobId: string | null) => void;
  runScenarios: (request: RunScenariosRequest) => Promise<void>;
  fetchHealthStatus: () => Promise<void>;
  clearErrors: () => void;
};

// ============================================================================
// Store Implementation
// ============================================================================

export const useResearchStore = create<ResearchState>((set, get) => ({
  // Initial state
  jobs: [],
  selectedJobId: null,
  isLoadingJobs: false,
  isCreatingJob: false,
  jobError: null,

  scenarioResults: [],
  isRunningScenarios: false,
  scenarioError: null,

  healthStatus: null,
  isLoadingHealth: false,
  healthError: null,

  // Actions
  fetchJobs: async () => {
    set({ isLoadingJobs: true, jobError: null });
    try {
      // Import dynamically to avoid circular dependencies
      const { fetchStrategySearchJobs } = await import("@/lib/researchApi");
      const jobs = await fetchStrategySearchJobs();
      set({ jobs });
    } catch (err: any) {
      const msg = err?.response?.data?.detail ?? err?.message ?? "Failed to fetch jobs";
      set({ jobError: msg, jobs: [] });
    } finally {
      set({ isLoadingJobs: false });
    }
  },

  fetchJobStatus: async (jobId: string) => {
    try {
      const { fetchJobStatus } = await import("@/lib/researchApi");
      const status = await fetchJobStatus(jobId);

      // Update job in list if it exists
      set((state) => ({
        jobs: state.jobs.map((job) =>
          job.job_id === jobId ? status : job
        ),
      }));

      return status;
    } catch (err: any) {
      const msg = err?.response?.data?.detail ?? err?.message ?? "Failed to fetch job status";
      set({ jobError: msg });
      return null;
    }
  },

  createSearchJob: async (request: StrategySearchJobRequest) => {
    set({ isCreatingJob: true, jobError: null });
    try {
      const { createStrategySearchJob } = await import("@/lib/researchApi");
      const response = await createStrategySearchJob(request);

      // Refresh job list after creation
      await get().fetchJobs();

      // Auto-select the new job
      set({ selectedJobId: response.job_id });

      return response;
    } catch (err: any) {
      const msg = err?.response?.data?.detail ?? err?.message ?? "Failed to create search job";
      set({ jobError: msg });
      return null;
    } finally {
      set({ isCreatingJob: false });
    }
  },

  setSelectedJobId: (jobId: string | null) => {
    set({ selectedJobId: jobId });
  },

  runScenarios: async (request: RunScenariosRequest) => {
    set({ isRunningScenarios: true, scenarioError: null });
    try {
      const { runScenarios } = await import("@/lib/researchApi");
      const response = await runScenarios(request);
      set({ scenarioResults: response.results });
    } catch (err: any) {
      const msg = err?.response?.data?.detail ?? err?.message ?? "Failed to run scenarios";
      set({ scenarioError: msg, scenarioResults: [] });
    } finally {
      set({ isRunningScenarios: false });
    }
  },

  fetchHealthStatus: async () => {
    set({ isLoadingHealth: true, healthError: null });
    try {
      const { fetchResearchHealth } = await import("@/lib/researchApi");
      const status = await fetchResearchHealth();
      set({ healthStatus: status });
    } catch (err: any) {
      const msg = err?.response?.data?.detail ?? err?.message ?? "Failed to fetch health status";
      set({ healthError: msg, healthStatus: null });
    } finally {
      set({ isLoadingHealth: false });
    }
  },

  clearErrors: () => {
    set({ jobError: null, scenarioError: null, healthError: null });
  },
}));
