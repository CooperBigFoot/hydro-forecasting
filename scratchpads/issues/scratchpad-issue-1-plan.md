# Issue #1: Fix Critical Import Issues Blocking Tests

## Objective
Fix ModuleNotFoundError in tests by updating import paths from `src.hydro_forecasting` to `hydro_forecasting`.

## Problem Analysis
- Tests are failing with `ModuleNotFoundError: No module named 'src'`
- The package is installed as `hydro_forecasting`, not `src.hydro_forecasting`
- This is blocking all test execution

## Plan
- [x] Analyze issue requirements
- [x] Create feature branch
- [x] Find all test files with incorrect imports
- [x] Update imports in identified files:
  - test_seed_integration.py
  - test_seed_manager.py
  - Any other test files with similar pattern
- [x] Verify tests can import correctly
- [ ] Create PR with fix

## Implementation Notes
- Pattern to find: `from src.hydro_forecasting.`
- Replace with: `from hydro_forecasting.`
- Must check all test files, not just the two mentioned

## Success Criteria
- No ModuleNotFoundError when running `uv run pytest`
- Tests start executing (even if some fail for other reasons)

## Implementation Summary
- Fixed imports in `tests/integration/test_seed_integration.py`: Changed line 26
- Fixed imports in `tests/unit/test_seed_manager.py`: Changed lines 34-39 and 3 occurrences on lines 214, 231, 240
- Verified no other test files have incorrect imports
- Tests now run successfully (245 tests collected, 243 passed, 2 failed for unrelated reasons)