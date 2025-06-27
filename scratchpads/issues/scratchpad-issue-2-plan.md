# Fix Issue #2: Auto-Fix Formatting and Simple Linting Issues

## Objective
Apply automatic fixes for formatting and simple linting issues to clean up the codebase.

## Current Status
- **Format Check**: 9 files need reformatting
- **Linting**: 30 auto-fixable issues out of 235 total
- **Additional**: .ipynb_checkpoints files need to be added to .gitignore

## Plan
- [x] Analyze issue requirements
- [ ] Create feature branch
- [ ] Update .gitignore first (prevent checkpoint files from being tracked)
- [ ] Run automatic formatting fixes
- [ ] Run automatic linting fixes
- [ ] Verify all checks pass with dev script
- [ ] Create PR

## Implementation Steps

### 1. Update .gitignore
Add entries for:
- .ipynb_checkpoints/
- __pycache__/
- *.pyc
- .pytest_cache/
- lightning_logs/

### 2. Run Auto-fixes
```bash
# Format all files
uv run ruff format .

# Fix auto-fixable linting issues
uv run ruff check --fix .
```

### 3. Verify Changes
```bash
# Run full dev check
uv run python scripts/dev.py full
```

## Expected Outcomes
- All formatting issues resolved
- 30 auto-fixable linting issues resolved
- No Jupyter checkpoint files in repository
- Clean git status with only intended changes

## Notes
- This is a straightforward cleanup task
- Should be done after issue #1 (import fixes) to ensure tests validate changes
- Focus on automated fixes only - don't manually fix complex linting issues