/**
 * Research Service API Client
 *
 * API wrappers for the Research Service endpoints (FastAPI).
 * Base URL: http://localhost:8001/api/research
 */

import axios from "axios";
import type {
  StrategySearchJobRequest,
  StrategySearchJobResponse,
  JobStatusResponse,
  RunScenariosRequest,
  RunScenariosResponse,
  ResearchHealthStatus,
} from "@/store/useResearchStore";

// ============================================================================
// API Client Setup
// ============================================================================

function getResearchApiBaseUrl() {
  const env = process.env.NEXT_PUBLIC_RESEARCH_API_BASE_URL;
  if (env && env.trim().length > 0) {
    return env.replace(/\/+$/, "");
  }
  // Default to research service port
  return "http://localhost:8001";
}

const researchApiBase = getResearchApiBaseUrl();
const researchApi = axios.create({
  baseURL: `${researchApiBase}/api/research`,
});

// ============================================================================
// Health Check Endpoints
// ============================================================================

/**
 * Ping the research service
 */
export async function pingResearchService(): Promise<{ status: string }> {
  const res = await researchApi.get("/ping");
  return res.data;
}

/**
 * Get research service health status
 */
export async function fetchResearchHealth(): Promise<ResearchHealthStatus> {
  const res = await researchApi.get("/status");
  return res.data;
}

// ============================================================================
// Strategy Search Job Endpoints
// ============================================================================

/**
 * Create a new strategy search job
 *
 * POST /api/research/strategy-search/jobs/
 */
export async function createStrategySearchJob(
  request: StrategySearchJobRequest
): Promise<StrategySearchJobResponse> {
  const res = await researchApi.post("/strategy-search/jobs/", request);
  return res.data;
}

/**
 * Get status of a specific job
 *
 * GET /api/research/strategy-search/jobs/{job_id}
 */
export async function fetchJobStatus(jobId: string): Promise<JobStatusResponse> {
  const res = await researchApi.get(`/strategy-search/jobs/${jobId}`);
  return res.data;
}

/**
 * List recent strategy search jobs
 *
 * GET /api/research/strategy-search/jobs/
 */
export async function fetchStrategySearchJobs(
  limit: number = 10,
  status?: string
): Promise<JobStatusResponse[]> {
  const params: Record<string, any> = { limit };
  if (status) {
    params.status = status;
  }
  const res = await researchApi.get("/strategy-search/jobs/", { params });
  return res.data;
}

// ============================================================================
// Scenario Analysis Endpoints
// ============================================================================

/**
 * Run scenario analysis
 *
 * POST /api/research/scenarios/run
 */
export async function runScenarios(
  request: RunScenariosRequest
): Promise<RunScenariosResponse> {
  const res = await researchApi.post("/scenarios/run", request);
  return res.data;
}

/**
 * Get saved scenario results (if available)
 *
 * GET /api/research/scenarios/{symbol}/{timeframe}
 */
export async function fetchScenarioResults(
  symbol: string,
  timeframe: string
): Promise<any[]> {
  const res = await researchApi.get(`/scenarios/${symbol}/${timeframe}`);
  return res.data;
}

// ============================================================================
// Error Handling Helper
// ============================================================================

/**
 * Extract user-friendly error message from API error
 */
export function extractErrorMessage(err: any): string {
  if (axios.isAxiosError(err)) {
    // FastAPI validation errors
    if (err.response?.status === 422) {
      return "Invalid request parameters";
    }
    // Job limit reached
    if (err.response?.status === 429) {
      return err.response?.data?.detail ?? "Job limit reached, please wait";
    }
    // General API error
    return err.response?.data?.detail ?? err.message ?? "API request failed";
  }
  return err?.message ?? "Unknown error";
}
