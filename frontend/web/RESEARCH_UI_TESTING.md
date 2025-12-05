# Research UI Testing Guide

This document provides manual testing procedures for the Research Service UI components.

## Prerequisites

1. **Backend Services Running**:
   - Main API: `http://localhost:8000`
   - Research Service: `http://localhost:8001`

2. **Frontend Running**:
   ```bash
   cd frontend/web
   npm run dev
   ```

3. **Environment Variables** (optional):
   Create `.env.local` if using custom API URLs:
   ```
   NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
   NEXT_PUBLIC_RESEARCH_API_BASE_URL=http://localhost:8001
   ```

## Test Suite 1: Core Tabs

### Test 1.1: Tab Switching (Research / Strategy Search / Live / Portfolio)
- [ ] Navigate to homepage
- [ ] Click "Research / Backtests" tab (default)
- [ ] Switch to "Strategy Search" tab
- [ ] Switch to "Live" tab
- [ ] Switch to "Portfolio / Monte Carlo" tab
- [ ] Return to "Research / Backtests"

**Expected**: Each tab highlights active state (blue underline, bold) and no console errors; state not lost when toggling.

---

## Test Suite 2: Scenario Analysis (Research)

### Test 2.1: Basic Scenario Configuration
- [ ] Navigate to Research > Scenarios
- [ ] Verify default configuration shows:
  - Base Symbol: AIAUSDT
  - Base Timeframe: 15m
  - 1 default scenario (Rule baseline)
- [ ] Change base symbol to "BTCUSDT"
- [ ] Change base timeframe to "1h"

**Expected**: Form fields update without errors.

### Test 2.2: Add/Remove Scenarios
- [ ] Click "+ Add Scenario" button
- [ ] Verify new scenario row appears
- [ ] Fill in scenario details:
  - Label: "Test Scenario"
  - Strategy: "trend_continuation"
  - Params: `{"trend_min": 1.0}`
- [ ] Add another scenario
- [ ] Click "Remove" on the second scenario
- [ ] Verify scenario is removed

**Expected**: Scenarios can be added and removed dynamically.

### Test 2.3: Run Scenarios (Happy Path)
- [ ] Configure 2-3 scenarios with different strategies
- [ ] Click "Run Scenarios" button
- [ ] Verify button shows "Running..." state
- [ ] Wait for results to load
- [ ] Verify results table populates with:
  - Label
  - Strategy
  - Symbol/TF
  - Metrics (Cum Return, Sharpe, Max DD, Trades, Win Rate)

**Expected**: Scenarios execute successfully and results display.

### Test 2.4: Scenario Results Interaction
- [ ] After running scenarios, click on a row in the results table
- [ ] Verify row highlights (blue background)
- [ ] Verify "Details" section below shows selected scenario
- [ ] Click different rows and verify details update

**Expected**: Row selection and detail display work correctly.

### Test 2.5: Error Handling
- [ ] Clear all scenarios
- [ ] Click "Run Scenarios"
- [ ] Verify button is disabled (no scenarios)
- [ ] Add a scenario with invalid params: `{invalid json`
- [ ] Verify form validates or ignores invalid JSON

**Expected**: Graceful error handling with user feedback.

---

## Test Suite 3: Strategy Search Tab

### Test 3.1: Job List Display
- [ ] Navigate to Research > Strategy Search
- [ ] Click "Refresh" button
- [ ] Verify job list loads
- [ ] If no jobs exist, verify message: "No jobs yet. Click 'New Job' to create one."

**Expected**: Job list displays or shows appropriate empty state.

### Test 3.2: Create New Job (Form Display)
- [ ] Click "New Job" button
- [ ] Verify create form appears with fields:
  - Job Name
  - Strategy (dropdown)
  - Symbol
  - Timeframe (dropdown)
  - Mode (Random/Grid)
  - Samples
  - Random Seed (optional)
- [ ] Click "Cancel"
- [ ] Verify form closes

**Expected**: Form displays and closes correctly.

### Test 3.3: Create Job (Happy Path)
- [ ] Click "New Job"
- [ ] Fill in form:
  - Job Name: "test_search_job"
  - Strategy: "rule"
  - Symbol: "AIAUSDT"
  - Timeframe: "15m"
  - Mode: "random"
  - Samples: 10 (small for quick test)
  - Seed: 42
- [ ] Click "Create Job"
- [ ] Verify button shows "Creating..."
- [ ] Wait for job to complete
- [ ] Verify:
  - Job appears in list
  - Job is auto-selected
  - Job details show on right panel

**Expected**: Job creates successfully and appears in list. *(Note: Job execution may take time depending on backend)*

### Test 3.4: Job Status Refresh
- [ ] Select a completed job
- [ ] Click "Refresh" button next to job in list
- [ ] Verify status updates

**Expected**: Status refresh works without errors.

### Test 3.5: Job Details Display
- [ ] Click on a job in the list
- [ ] Verify right panel shows:
  - Job ID
  - Status (completed/pending/failed)
  - Created timestamp
  - Completed timestamp (if done)
  - Results Count
  - Results Available (Yes/No)
- [ ] If results available, verify file path shown

**Expected**: Job details display accurately.

### Test 3.6: Multiple Jobs
- [ ] Create 2-3 jobs with different configurations
- [ ] Verify all jobs appear in list
- [ ] Verify job selection works
- [ ] Verify most recent job appears at top

**Expected**: Multiple jobs handled correctly.

### Test 3.7: Error Handling - Job Limit
- [ ] Create multiple jobs rapidly
- [ ] If job limit reached (429 error), verify error message displays:
  "Job limit reached. X/Y jobs running."

**Expected**: Job limit errors handled gracefully.

### Test 3.8: Error Handling - Invalid Parameters
- [ ] Try to create job with:
  - Samples: 0 (should fail validation)
  - Samples: 2000 (exceeds max)
- [ ] Verify form validation or API error message

**Expected**: Validation prevents invalid submissions.

---

## Test Suite 4: Integration Tests

### Test 4.1: API Connection - Research Service Down
- [ ] Stop research service (if running)
- [ ] Navigate to Research > Scenarios
- [ ] Try to run scenarios
- [ ] Verify error message displays (connection/network error)
- [ ] Restart research service
- [ ] Verify operations work again

**Expected**: Clear error messages when service unavailable.

### Test 4.2: API Connection - Research Service Running
- [ ] Ensure research service is running
- [ ] Navigate to Research tab
- [ ] Verify no errors in browser console
- [ ] Test basic operations (run scenario, refresh jobs)

**Expected**: All operations work smoothly.

### Test 4.3: State Persistence
- [ ] Create a scenario configuration (don't run)
- [ ] Switch to "Strategy Search" sub-tab
- [ ] Switch back to "Scenarios"
- [ ] Verify scenario configuration is still there (in same session)

**Expected**: UI state persists during navigation within session.

### Test 4.4: Multiple Tab Navigation
- [ ] Run a scenario
- [ ] While waiting, switch to "Single Backtests" tab
- [ ] Switch back to Research > Scenarios
- [ ] Verify results loaded correctly

**Expected**: Tab switching doesn't break async operations.

---

## Test Suite 5: Visual & UX Testing

### Test 5.1: Responsive Design
- [ ] Resize browser window to mobile width (< 768px)
- [ ] Verify tables scroll horizontally if needed
- [ ] Verify forms stack vertically
- [ ] Resize to tablet (768-1024px)
- [ ] Verify layouts adjust appropriately

**Expected**: UI remains usable at different screen sizes.

### Test 5.2: Loading States
- [ ] Observe "Running..." state on scenario run button
- [ ] Observe "Creating..." state on job creation
- [ ] Observe "Loading..." state on job refresh
- [ ] Verify loading states clear after operation completes

**Expected**: Clear visual feedback during async operations.

### Test 5.3: Error Display
- [ ] Trigger an error (e.g., invalid API call)
- [ ] Verify error message appears in red bordered box
- [ ] Verify error is readable and helpful

**Expected**: Errors are prominently displayed.

---

## Test Suite 6: Browser Console Tests

### Test 6.1: Console Errors
- [ ] Open browser DevTools console
- [ ] Navigate through all Research tab features
- [ ] Verify no React errors (red messages)
- [ ] Verify no API 500 errors (check Network tab)

**Expected**: Clean console with no errors during normal operation.

### Test 6.2: Network Requests
- [ ] Open DevTools Network tab
- [ ] Run a scenario
- [ ] Verify request to `/api/research/scenarios/run`
- [ ] Verify successful response (200 status)
- [ ] Refresh jobs
- [ ] Verify request to `/api/research/strategy-search/jobs/`

**Expected**: Correct API endpoints called with proper responses.

---

## Regression Tests

After any code changes to research components, run abbreviated smoke test:

- [ ] Can navigate to Research tab
- [ ] Can run a simple scenario (1 strategy)
- [ ] Can create a small strategy search job (10 samples)
- [ ] No errors in console
- [ ] Results display correctly

---

## Known Limitations / Future Enhancements

1. **No automated tests**: Current testing is manual. Future: Add Jest + React Testing Library.
2. **No job cancellation**: Jobs run to completion. Future: Add cancel/stop functionality.
3. **No result download**: Results shown in backend file system. Future: Add CSV download from UI.
4. **No job progress indicator**: Jobs show "completed" or "pending" only. Future: Add progress %.
5. **Limited error details**: API errors show summary only. Future: Add expandable error details.

---

## Test Report Template

Use this template when reporting test results:

```
## Research UI Test Report

**Date**: YYYY-MM-DD
**Tester**: [Name]
**Branch/Commit**: [Git SHA]
**Environment**:
  - Frontend: http://localhost:3000
  - Backend API: http://localhost:8000
  - Research API: http://localhost:8001

**Tests Run**: X/Y passed

**Failures**:
1. [Test ID] - [Description] - [Error/Screenshot]

**Notes**:
- [Any observations or comments]
```

---

## Quick Start Test (5 minutes)

For rapid verification after deployment:

1. [ ] Navigate to Research tab
2. [ ] Run 1 scenario (rule strategy, AIAUSDT, 15m)
3. [ ] Verify results appear
4. [ ] Create 1 search job (rule, 10 samples)
5. [ ] Verify job appears in list
6. [ ] Check console for errors

If all pass → ✅ Basic functionality working
