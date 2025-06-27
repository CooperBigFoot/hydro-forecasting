# Plan to Fix Linting, Formatting, Typing, and Test Issues

## Current Status Summary

The dev script shows multiple failures:

- **Format Check**: ❌ FAILED - 9 files need reformatting  
- **Linting**: ❌ FAILED - 235 errors found
- **Type Check**: ❌ FAILED - 261 diagnostics
- **Tests**: ❌ FAILED - Import errors preventing tests from running

## Phase 1: Fix Critical Import Issues (Blocking Tests)

### Problem

Tests fail with `ModuleNotFoundError: No module named 'src'` when importing:

- `from src.hydro_forecasting.experiment_utils.seed_manager import SeedManager`

### Root Cause

The tests are using absolute imports starting with `src.`, but the package is installed as `hydro_forecasting`.

### Solution

Update all test imports to use the correct package name:

- Change: `from src.hydro_forecasting.X import Y`
- To: `from hydro_forecasting.X import Y`

### Files to Update

- `tests/integration/test_seed_integration.py`
- `tests/unit/test_seed_manager.py`
- Check all other test files for similar import patterns

## Phase 2: Auto-Fix Simple Issues

### 2.1 Format Issues (9 files)

Run `uv run ruff format .` to automatically fix:

- Jupyter notebook cells in `experiments/HPT/run_experiment.ipynb`
- Jupyter notebook checkpoints (consider adding `.ipynb_checkpoints` to .gitignore)
- Line length and formatting issues

### 2.2 Auto-fixable Linting Issues (30 fixable)

Run `uv run ruff check --fix .` to fix:

- **W293**: blank-line-with-whitespace (22 instances)
- **W291**: trailing-whitespace (1 instance)  
- **F811**: redefined-while-unused (1 instance)
- **I001**: unsorted-imports (1 instance)

## Phase 3: Manual Linting Fixes (by priority)

### High Priority (Affects code quality)

1. **N806**: non-lowercase-variable-in-function (60 instances)
   - Review and rename variables to follow PEP8 conventions

2. **N803**: invalid-argument-name (40 instances)
   - Fix function argument names to be lowercase with underscores

3. **B904**: raise-without-from-inside-except (32 instances)
   - Add `from e` to exception chains for better debugging

### Medium Priority (Code style)

4. **F401**: unused-import (11 instances)
   - Remove unused imports

5. **F841**: unused-variable (10 instances)
   - Remove or use unused variables

6. **B007**: unused-loop-control-variable (17 instances)
   - Prefix with underscore if intentionally unused

### Low Priority (Simplifications)

7. **SIM108**: if-else-block-instead-of-if-exp (11 instances)
   - Convert to ternary operators where appropriate

8. **UP038**: non-pep604-isinstance (5 instances)
   - Use modern union syntax for isinstance checks

## Phase 4: Type Checking Issues

### Critical Type Errors

1. **unresolved-attribute**: `Type 'bool' has no attribute 'all'`
   - Location: `tests/unit/test_unified_pipeline.py:395`
   - Fix: The result of `pd.isna()` needs `.values` before `.all()`

2. **possibly-unresolved-reference**: Multiple warnings about `np` and `torch`
   - These are in conditional imports - may need type: ignore comments

### Type Checking Strategy

- Review all 261 diagnostics
- Focus on actual errors vs warnings
- Add type annotations where missing
- Use `# type: ignore` sparingly for false positives

## Phase 5: Test Suite Fixes

After fixing imports:

1. Run tests individually to identify actual test failures
2. Fix failing tests one by one
3. Ensure all tests pass with proper assertions

## Phase 6: Project Cleanup

### Add to .gitignore

```
.ipynb_checkpoints/
__pycache__/
*.pyc
.pytest_cache/
lightning_logs/
```

### Remove unnecessary files

- Jupyter notebook checkpoints
- Old experiment directories if not needed

## Phase 7: CI/CD Preparation

1. Ensure all checks pass locally with `uv run python scripts/dev.py full`
2. Update pre-commit hooks if needed
3. Document any intentional deviations from style rules

## Execution Order

1. **First**: Fix import issues (Phase 1) - Critical blocker
2. **Second**: Run auto-fixes (Phase 2) - Quick wins
3. **Third**: Fix type errors (Phase 4 critical only) - Actual bugs
4. **Fourth**: Fix tests (Phase 5) - Get tests passing
5. **Fifth**: Manual linting fixes (Phase 3) - Code quality
6. **Sixth**: Remaining type warnings (Phase 4) - Nice to have
7. **Last**: Cleanup (Phase 6 & 7) - Polish

## Expected Outcome

After completing all phases:

- ✅ All tests passing
- ✅ Code properly formatted
- ✅ Linting errors resolved
- ✅ Type checking passing
- ✅ Ready for CI/CD pipeline

## Time Estimate

- Phase 1: 15 minutes
- Phase 2: 5 minutes
- Phase 3: 1-2 hours
- Phase 4: 1 hour
- Phase 5: 30 minutes - 1 hour
- Phase 6-7: 30 minutes

Total: 3-5 hours of focused work
