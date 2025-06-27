# GitHub Issues Creation Process

## Overview
Created GitHub issues based on the linting/formatting/typing/test issues identified in `fix_linting_formatting_typing_tests.md`.

## Thought Process

### 1. Issue Categorization
I decided to create 5 separate issues instead of one large issue because:
- Each phase has different priorities and complexities
- Smaller issues are easier to tackle and review
- Multiple contributors can work on different issues in parallel
- Progress is more visible with granular issues

### 2. Priority Assignment
- **Issue #1 (Import Fixes)**: CRITICAL - Blocks all testing
- **Issue #2 (Auto-fixes)**: HIGH - Quick wins, improves baseline
- **Issue #3 (Manual Linting)**: MEDIUM - Important but time-consuming
- **Issue #4 (Type Checking)**: MEDIUM - Prevents bugs but not blocking
- **Issue #5 (Tests & Cleanup)**: LOW - Final polish

### 3. Issue Structure
Each issue includes:
- Clear summary of the problem
- Current status/metrics
- Specific tasks to complete
- Acceptance criteria checklist
- Time estimates
- Dependencies on other issues

### 4. Label Strategy
- Used existing GitHub labels:
  - `bug` for issues causing failures
  - `enhancement` for code quality improvements
- Avoided creating new labels to keep it simple

## Created Issues

1. **#1: Fix Critical Import Issues Blocking Tests**
   - Focus: ModuleNotFoundError preventing test execution
   - Scope: Update all test imports from `src.hydro_forecasting` to `hydro_forecasting`
   - Priority: CRITICAL

2. **#2: Auto-Fix Formatting and Simple Linting Issues**
   - Focus: Automated fixes with ruff
   - Scope: Format 9 files, fix 30 auto-fixable lint issues, update .gitignore
   - Priority: HIGH

3. **#3: Fix High-Priority Manual Linting Issues**
   - Focus: Naming conventions and exception handling
   - Scope: 100 naming issues, 32 exception chains, 21 unused code instances
   - Priority: MEDIUM

4. **#4: Fix Type Checking Issues**
   - Focus: Type errors and annotations
   - Scope: Fix critical attribute errors, add type hints, reduce 261 diagnostics
   - Priority: MEDIUM

5. **#5: Fix Remaining Test Failures and Project Cleanup**
   - Focus: Test fixes after imports resolved, final cleanup
   - Scope: Fix test logic, simplify code, clean repository
   - Priority: LOW

## Execution Strategy

Recommended order:
1. Start with #1 (imports) - unblocks everything else
2. Quick win with #2 (auto-fixes) - improves baseline quickly  
3. Tackle #3 and #4 in parallel if multiple contributors
4. Finish with #5 for final polish

## Success Metrics

After all issues are resolved:
- ✅ `uv run python scripts/dev.py full` passes completely
- ✅ CI/CD pipeline is green
- ✅ Codebase follows consistent style
- ✅ Tests provide good coverage
- ✅ Type checking helps catch bugs early

## Notes for Future Contributors

- Each issue is self-contained with clear instructions
- Time estimates are conservative - experienced devs may be faster
- Issues can be tackled by different team members
- The dev script (`scripts/dev.py`) is the source of truth for validation