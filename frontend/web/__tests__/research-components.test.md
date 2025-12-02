# Research Components Test Checklist

This is a simplified test checklist for the Research UI components. For full test procedures, see `RESEARCH_UI_TESTING.md`.

## Component: ResearchScenariosPanel

### Rendering Tests
- [ ] Component renders without errors
- [ ] Default configuration displays (AIAUSDT, 15m)
- [ ] Default scenario ("Rule baseline") is present
- [ ] "Run Scenarios" button is visible and enabled

### Interaction Tests
- [ ] Can add a new scenario
- [ ] Can remove a scenario
- [ ] Can update scenario label
- [ ] Can change strategy dropdown
- [ ] Can edit params JSON
- [ ] Run button disables when no scenarios
- [ ] Run button shows "Running..." during execution

### Data Tests
- [ ] Results table renders after successful run
- [ ] Can click row to select
- [ ] Selected row details display correctly
- [ ] Metric values format correctly (decimals, percentages)

### Error Tests
- [ ] Error message displays on API failure
- [ ] Invalid JSON in params handled gracefully

---

## Component: ResearchJobsPanel

### Rendering Tests
- [ ] Component renders without errors
- [ ] Job list displays
- [ ] Empty state shows when no jobs
- [ ] "New Job" and "Refresh" buttons visible

### Create Job Form Tests
- [ ] Form appears when clicking "New Job"
- [ ] Form closes when clicking "Cancel"
- [ ] All form fields are editable
- [ ] Strategy dropdown has correct options
- [ ] Samples input validates (1-1000)

### Job Operations Tests
- [ ] Can create a new job
- [ ] Job appears in list after creation
- [ ] Can select job from list
- [ ] Job details display in right panel
- [ ] Refresh button updates job status

### Error Tests
- [ ] Error message displays on job creation failure
- [ ] 429 error (job limit) shows appropriate message
- [ ] Connection errors handled gracefully

---

## Store: useResearchStore

### State Management Tests
- [ ] Initial state is correct
- [ ] `fetchJobs()` loads jobs
- [ ] `createSearchJob()` creates job and updates list
- [ ] `runScenarios()` updates scenarioResults
- [ ] `fetchHealthStatus()` loads health data
- [ ] `clearErrors()` clears all error states

### Error Handling Tests
- [ ] API errors set appropriate error states
- [ ] Error messages are extracted correctly

---

## API Client: researchApi

### Endpoint Tests
- [ ] `pingResearchService()` calls /api/research/ping
- [ ] `fetchResearchHealth()` calls /api/research/status
- [ ] `createStrategySearchJob()` posts to /api/research/strategy-search/jobs/
- [ ] `fetchJobStatus()` gets /api/research/strategy-search/jobs/{id}
- [ ] `fetchStrategySearchJobs()` lists jobs
- [ ] `runScenarios()` posts to /api/research/scenarios/run

### Configuration Tests
- [ ] Uses correct base URL (localhost:8001)
- [ ] Respects NEXT_PUBLIC_RESEARCH_API_BASE_URL env var

---

## Integration: page.tsx

### Tab Navigation Tests
- [ ] "Research" tab button displays
- [ ] Clicking "Research" shows research section
- [ ] Sub-tabs (Scenarios/Strategy Search) work
- [ ] Switching tabs preserves component state

### Component Integration Tests
- [ ] ResearchScenariosPanel renders in Scenarios sub-tab
- [ ] ResearchJobsPanel renders in Strategy Search sub-tab
- [ ] Components can interact with store
- [ ] No console errors during navigation

---

## Automated Test Template (Future)

When implementing automated tests with Jest + React Testing Library, use this structure:

```typescript
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import ResearchScenariosPanel from '@/components/ResearchScenariosPanel';

describe('ResearchScenariosPanel', () => {
  it('renders without crashing', () => {
    render(<ResearchScenariosPanel />);
    expect(screen.getByText(/Scenario Analysis/i)).toBeInTheDocument();
  });

  it('allows adding a scenario', () => {
    render(<ResearchScenariosPanel />);
    const addButton = screen.getByText(/Add Scenario/i);
    fireEvent.click(addButton);
    // Assert new scenario row appears
  });

  it('displays results after running scenarios', async () => {
    // Mock API response
    // Trigger run
    // Wait for results
    // Assert results table populated
  });
});
```

---

## Test Coverage Goals

- **Unit Tests**: 80%+ coverage for store and API client
- **Component Tests**: All user interactions covered
- **Integration Tests**: Critical user flows end-to-end
- **E2E Tests**: At least 1 happy path scenario per feature

---

## CI/CD Integration (Future)

When setting up automated tests in CI:

1. **Pre-commit**: Run linter + type checks
2. **Pre-push**: Run unit tests
3. **PR**: Run all tests + visual regression
4. **Deploy**: Run smoke tests against staging

```yaml
# Example GitHub Actions workflow
name: Frontend Tests
on: [pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - run: npm ci
      - run: npm run test
      - run: npm run test:e2e
```
