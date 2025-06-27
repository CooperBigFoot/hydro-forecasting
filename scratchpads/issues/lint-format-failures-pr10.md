# Lint and Format Failures from PR #10

## Context
PR #10 was merged with failing CI checks at user's request. This scratchpad documents the failures that need to be addressed.

## Failed Checks
1. **Lint and Format Code** - FAILURE
   - CI Run: https://github.com/CooperBigFoot/hydro-forecasting/actions/runs/15928404364/job/44931238516
   
2. **Type Check Code** - FAILURE  
   - CI Run: https://github.com/CooperBigFoot/hydro-forecasting/actions/runs/15928404364/job/44931238513

## Merge Decision
- User explicitly requested merge despite failures
- Tests on Python 3.11 and 3.12 passed successfully
- Python 3.10 test was still running at merge time

## Follow-up Actions
- [ ] Create GitHub issue to track lint/format fixes
- [ ] Run local lint checks to identify specific issues
- [ ] Fix formatting issues with `uv run python scripts/dev.py format`
- [ ] Fix linting issues with `uv run python scripts/dev.py fix`
- [ ] Address type checking errors with `uv run ty`
- [ ] Ensure all CI checks pass before next PR

## Notes
- Merge strategy: squash merge (default for clean history)
- Branch will be deleted after merge
- Issue will be created to ensure these problems are addressed