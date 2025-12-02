/**
 * Research Jobs Panel
 *
 * Displays and manages strategy search jobs from the Research Service.
 */

"use client";

import { useEffect, useState } from "react";
import { useResearchStore } from "@/store/useResearchStore";
import type { StrategySearchJobRequest } from "@/store/useResearchStore";

export default function ResearchJobsPanel() {
  const {
    jobs,
    selectedJobId,
    isLoadingJobs,
    isCreatingJob,
    jobError,
    fetchJobs,
    fetchJobStatus,
    createSearchJob,
    setSelectedJobId,
    clearErrors,
  } = useResearchStore();

  const [showCreateForm, setShowCreateForm] = useState(false);
  const [formData, setFormData] = useState<StrategySearchJobRequest>({
    job_name: "search_job",
    strategy_name: "rule",
    symbol: "AIAUSDT",
    timeframe: "15m",
    mode: "random",
    n_samples: 50,
    random_seed: null,
  });

  useEffect(() => {
    fetchJobs();
  }, [fetchJobs]);

  const selectedJob = jobs.find((j) => j.job_id === selectedJobId) ?? null;

  const handleCreateJob = async () => {
    const result = await createSearchJob(formData);
    if (result) {
      setShowCreateForm(false);
      // Reset form
      setFormData({
        job_name: "search_job",
        strategy_name: "rule",
        symbol: "AIAUSDT",
        timeframe: "15m",
        mode: "random",
        n_samples: 50,
        random_seed: null,
      });
    }
  };

  const handleRefreshStatus = async (jobId: string) => {
    await fetchJobStatus(jobId);
  };

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Strategy Search Jobs</h2>
        <div className="flex gap-2">
          <button
            onClick={() => fetchJobs()}
            disabled={isLoadingJobs}
            className="px-3 py-1 border rounded text-sm bg-white hover:bg-slate-50"
          >
            {isLoadingJobs ? "Loading..." : "Refresh"}
          </button>
          <button
            onClick={() => {
              setShowCreateForm(!showCreateForm);
              clearErrors();
            }}
            className="px-3 py-1 border rounded text-sm bg-blue-500 text-white hover:bg-blue-600"
          >
            {showCreateForm ? "Cancel" : "New Job"}
          </button>
        </div>
      </div>

      {/* Error Display */}
      {jobError && (
        <div className="rounded bg-red-50 border border-red-200 p-3 text-sm text-red-700">
          {jobError}
        </div>
      )}

      {/* Create Job Form */}
      {showCreateForm && (
        <div className="rounded bg-white shadow p-4 space-y-3">
          <h3 className="font-semibold text-sm">Create Strategy Search Job</h3>
          <div className="grid grid-cols-2 gap-3 text-sm">
            <label className="flex flex-col gap-1">
              <span className="text-xs text-gray-600">Job Name</span>
              <input
                type="text"
                className="border px-2 py-1 rounded"
                value={formData.job_name}
                onChange={(e) =>
                  setFormData({ ...formData, job_name: e.target.value })
                }
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-gray-600">Strategy</span>
              <select
                className="border px-2 py-1 rounded"
                value={formData.strategy_name}
                onChange={(e) =>
                  setFormData({ ...formData, strategy_name: e.target.value })
                }
              >
                <option value="rule">rule</option>
                <option value="trend_continuation">trend_continuation</option>
                <option value="sweep_reversal">sweep_reversal</option>
                <option value="volatility_breakout">volatility_breakout</option>
              </select>
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-gray-600">Symbol</span>
              <input
                type="text"
                className="border px-2 py-1 rounded"
                value={formData.symbol}
                onChange={(e) =>
                  setFormData({ ...formData, symbol: e.target.value })
                }
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-gray-600">Timeframe</span>
              <select
                className="border px-2 py-1 rounded"
                value={formData.timeframe}
                onChange={(e) =>
                  setFormData({ ...formData, timeframe: e.target.value })
                }
              >
                <option value="5m">5m</option>
                <option value="15m">15m</option>
                <option value="1h">1h</option>
                <option value="4h">4h</option>
              </select>
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-gray-600">Mode</span>
              <select
                className="border px-2 py-1 rounded"
                value={formData.mode}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    mode: e.target.value as "random" | "grid",
                  })
                }
              >
                <option value="random">Random</option>
                <option value="grid">Grid</option>
              </select>
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-gray-600">Samples</span>
              <input
                type="number"
                className="border px-2 py-1 rounded"
                value={formData.n_samples}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    n_samples: parseInt(e.target.value) || 50,
                  })
                }
                min={1}
                max={1000}
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-gray-600">Random Seed (optional)</span>
              <input
                type="number"
                className="border px-2 py-1 rounded"
                value={formData.random_seed ?? ""}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    random_seed: e.target.value ? parseInt(e.target.value) : null,
                  })
                }
                placeholder="Auto"
              />
            </label>
          </div>
          <div className="flex justify-end gap-2">
            <button
              onClick={() => setShowCreateForm(false)}
              className="px-3 py-1 border rounded text-sm"
            >
              Cancel
            </button>
            <button
              onClick={handleCreateJob}
              disabled={isCreatingJob}
              className="px-3 py-1 bg-blue-500 text-white rounded text-sm hover:bg-blue-600"
            >
              {isCreatingJob ? "Creating..." : "Create Job"}
            </button>
          </div>
        </div>
      )}

      {/* Jobs List */}
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_1.5fr] gap-4">
        <div className="rounded bg-white shadow">
          <div className="border-b px-4 py-2 font-semibold text-sm">Jobs</div>
          <div className="max-h-96 overflow-auto">
            {jobs.length === 0 && !isLoadingJobs && (
              <div className="px-4 py-8 text-center text-sm text-gray-500">
                No jobs yet. Click &quot;New Job&quot; to create one.
              </div>
            )}
            {jobs.map((job) => {
              const isSelected = job.job_id === selectedJobId;
              return (
                <div
                  key={job.job_id}
                  className={`px-4 py-3 border-b last:border-0 cursor-pointer ${
                    isSelected ? "bg-blue-50" : "hover:bg-slate-50"
                  }`}
                  onClick={() => setSelectedJobId(job.job_id)}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex-1 min-w-0">
                      <div className="font-medium text-sm truncate">
                        {job.job_id}
                      </div>
                      <div className="text-xs text-gray-500 mt-1">
                        Status:{" "}
                        <span
                          className={`font-medium ${
                            job.status === "completed"
                              ? "text-green-600"
                              : job.status === "failed"
                              ? "text-red-600"
                              : "text-yellow-600"
                          }`}
                        >
                          {job.status}
                        </span>
                      </div>
                      {job.n_results > 0 && (
                        <div className="text-xs text-gray-500 mt-0.5">
                          Results: {job.n_results}
                        </div>
                      )}
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleRefreshStatus(job.job_id);
                      }}
                      className="text-xs text-blue-600 hover:text-blue-800"
                    >
                      Refresh
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Job Details */}
        <div className="rounded bg-white shadow p-4">
          {selectedJob ? (
            <div className="space-y-3 text-sm">
              <div>
                <h3 className="font-semibold mb-2">Job Details</h3>
                <dl className="space-y-2">
                  <div className="grid grid-cols-[120px_1fr] gap-2">
                    <dt className="text-gray-600">Job ID:</dt>
                    <dd className="font-mono text-xs break-all">{selectedJob.job_id}</dd>
                  </div>
                  <div className="grid grid-cols-[120px_1fr] gap-2">
                    <dt className="text-gray-600">Status:</dt>
                    <dd>
                      <span
                        className={`font-medium ${
                          selectedJob.status === "completed"
                            ? "text-green-600"
                            : selectedJob.status === "failed"
                            ? "text-red-600"
                            : "text-yellow-600"
                        }`}
                      >
                        {selectedJob.status}
                      </span>
                    </dd>
                  </div>
                  <div className="grid grid-cols-[120px_1fr] gap-2">
                    <dt className="text-gray-600">Created:</dt>
                    <dd>
                      {selectedJob.created_at
                        ? new Date(selectedJob.created_at).toLocaleString()
                        : "-"}
                    </dd>
                  </div>
                  {selectedJob.completed_at && (
                    <div className="grid grid-cols-[120px_1fr] gap-2">
                      <dt className="text-gray-600">Completed:</dt>
                      <dd>{new Date(selectedJob.completed_at).toLocaleString()}</dd>
                    </div>
                  )}
                  <div className="grid grid-cols-[120px_1fr] gap-2">
                    <dt className="text-gray-600">Results Count:</dt>
                    <dd>{selectedJob.n_results}</dd>
                  </div>
                  <div className="grid grid-cols-[120px_1fr] gap-2">
                    <dt className="text-gray-600">Results Available:</dt>
                    <dd>{selectedJob.results_available ? "Yes" : "No"}</dd>
                  </div>
                </dl>
              </div>

              {selectedJob.results_available && (
                <div className="pt-3 border-t">
                  <p className="text-xs text-gray-600">
                    View results at:{" "}
                    <code className="bg-slate-100 px-1 py-0.5 rounded text-[11px]">
                      outputs/strategy_search/{selectedJob.job_id}/results.csv
                    </code>
                  </p>
                </div>
              )}
            </div>
          ) : (
            <div className="text-sm text-gray-500">
              Select a job from the list to view details.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
