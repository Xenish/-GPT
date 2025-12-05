# Frontend Testing

## Running tests
```bash
cd frontend/web
npm install   # first time
npm test      # runs Jest/RTL suite
```

## API mocks
- The frontend test suite mocks API calls via Jest. See `__mocks__/api.ts` for the exported mock fns.
- In tests, `jest.mock("@/lib/api", ...)` is used so no real backend requests are made.

## Structure
- Tests live under `frontend/web/__tests__/`.
- Jest config: `jest.config.ts`; setup: `jest.setup.ts` (loads `@testing-library/jest-dom`).
