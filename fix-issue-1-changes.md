# Fix for Issue #1: Critical Import Issues Blocking Tests

## Changes Made

### File: tests/integration/test_seed_integration.py
- Line 26: Changed `from src.hydro_forecasting.experiment_utils.seed_manager import SeedManager` 
  to `from hydro_forecasting.experiment_utils.seed_manager import SeedManager`

### File: tests/unit/test_seed_manager.py
- Lines 34-39: Changed import statement from `src.hydro_forecasting` to `hydro_forecasting`
- Line 214: Changed patch path from `src.hydro_forecasting` to `hydro_forecasting`
- Line 231: Changed patch path from `src.hydro_forecasting` to `hydro_forecasting`
- Line 240: Changed patch path from `src.hydro_forecasting` to `hydro_forecasting`

## Result
- All test imports now work correctly
- Tests execute successfully: 245 tests collected, 243 passed, 2 failed (unrelated to import issue)
- The critical ModuleNotFoundError has been resolved

## Branch
- fix/issue-1-import-errors