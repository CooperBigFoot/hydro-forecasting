# Project: streamflow-mapper

This file helps Claude Code understand our project setup and workflows.

## Tool Stack

- **UV**: Python package and project manager (replaces pip, poetry, pyenv)
- **RUFF**: Python linter and formatter (replaces Flake8, Black, isort)
- **TY**: Type checker from Astral (⚠️ Preview/alpha version - not used in CI/CD)

## Development Script (`scripts/dev.py`)

Our main development orchestration script that mirrors the CI/CD pipeline locally.

### Core Commands

```bash
# Full pipeline (recommended before commits)
uv run python scripts/dev.py full         # Run all checks like CI/CD

# Individual operations
uv run python scripts/dev.py format       # Format code with Ruff
uv run python scripts/dev.py check-format # Check if formatting needed
uv run python scripts/dev.py lint         # Lint code with Ruff
uv run python scripts/dev.py fix          # Auto-fix linting issues
uv run python scripts/dev.py type-check   # Run TY type checking (optional, preview)
uv run python scripts/dev.py test         # Run pytest test suite
uv run python scripts/dev.py test-cov     # Run tests with coverage
uv run python scripts/dev.py install      # Install/sync dependencies
```

### What `full` command does

1. **Dependencies** - `uv sync --dev`
2. **Format Check** - `ruff format --check --diff`
3. **Linting** - `ruff check`
4. **Tests** - `pytest tests/ -v`

Provides clear ✅/❌ summary at the end.

## UV Commands

```bash
# Setup
uv sync                    # Install dependencies
uv sync --dev             # Install with dev dependencies
uv sync --all-extras      # Install all optional dependencies
uv venv                   # Create virtual environment
uv python install         # Install Python from .python-version

# Dependencies
uv add <package>          # Add dependency
uv add --dev <package>    # Add dev dependency
uv add --optional <group> <package>  # Add to optional group
uv remove <package>       # Remove dependency
uv lock                   # Update lockfile
uv lock --upgrade-package <pkg>  # Update specific package

# Running
uv run <command>          # Run command in project environment
uv run python script.py   # Run Python script
uv run pytest            # Run tests
uv run ruff check        # Run linter

# Building
uv build                  # Build project
uv publish               # Publish to PyPI
```

## RUFF Commands

```bash
# Linting
uv run ruff check              # Check all files
uv run ruff check --fix        # Auto-fix issues
uv run ruff check --watch      # Watch mode
uv run ruff check <file>       # Check specific file

# Formatting
uv run ruff format             # Format all files
uv run ruff format --check     # Check formatting only
uv run ruff format <file>      # Format specific file

# Combined (recommended order)
uv run ruff check --fix && uv run ruff format
```

## TY Commands

```bash
# Type Checking
uv run ty                      # Check project
uv run ty <file>              # Check specific file
uv run ty --watch             # Watch mode

# Configuration
uv run ty --python-version 3.10  # Specify Python version
uv run ty --ignore <rule>        # Ignore specific rule
uv run ty --error <rule>         # Treat rule as error
```

## Development Workflow

### Starting New Work

```bash
git checkout -b feature/description
uv sync --dev
# Check for existing scratchpads
```

### During Development

```bash
# Quick checks during coding:
uv run python scripts/dev.py format      # Fix formatting
uv run python scripts/dev.py fix         # Fix auto-fixable lint issues

# In separate terminals for continuous feedback:
uv run ty --watch             # Continuous type checking
uv run ruff check --watch     # Continuous linting

# Run tests frequently:
uv run python scripts/dev.py test
uv run pytest tests/test_specific.py
```

### Before Committing

```bash
# Full pipeline check (mirrors CI/CD):
uv run python scripts/dev.py full

# Or manual steps:
uv run ruff check --fix && uv run ruff format && uv run ty && uv run pytest

# Or with pre-commit hooks:
git commit  # pre-commit runs automatically
```

### Troubleshooting

```bash
# If dev script fails:
uv run python scripts/dev.py install     # Reinstall dependencies
uv pip list | grep -E "(pytest|ruff|ty)" # Check tool availability

# Individual tool debugging:
uv run ruff --version
uv run pytest --version  
uv run ty --version
```

## CI/CD Pipeline

Our GitHub Actions pipeline (`.github/workflows/ci.yaml`) runs the same checks as `scripts/dev.py full`:

- **Lint & Format** - Ruff check and format validation
- **Test Suite** - pytest on multiple Python versions
- **Build Check** - Verify package builds correctly

Note: Type checking with TY is not included in CI/CD as it's still in preview/alpha

**Triggers**: Push to main, PRs to main, manual dispatch

## Scratchpad System

Organize complex work with markdown scratchpads:

```
scratchpads/
├── issues/      # Issue tracking (use /project:fix-issue command)
├── planning/    # Feature planning
└── research/    # Technical exploration
```

**Format**: `scratchpads/{type}/{description}.md`

**Template**:

```markdown
# [Task Name]

## Objective
[What needs to be done]

## Plan
- [ ] Step 1
- [ ] Step 2

## Notes
[Findings, decisions, code snippets]
```

## Project Configuration

- **Python**: >=3.10 (see .python-version)
- **Ruff**: line-length=88, rules: E,F,W,N,I,UP,B,C4,SIM
- **TY**: Basic type checking with gradual adoption
- **Layout**: Source code in `src/`
- **Virtual env**: `.venv/`

## Key Files

- `@README.md` - Project structure
- `@.github/copilot-instructions.md` - Coding standards
- `@pyproject.toml` - Project configuration
- `@scripts/dev.py` - Development orchestration script
- `@.claude/commands/fix-issue.md` - Issue workflow command

## Extended Thinking

For complex problems, use trigger words:

- "think" - Basic extended thinking
- "think hard/more/harder" - Deeper analysis

## Quick Reference

```bash
# Most common development workflow:
uv sync --dev                          # Setup
uv run python scripts/dev.py full      # Full check (like CI/CD)

# Fast iteration during coding:
uv run python scripts/dev.py format    # Format code
uv run python scripts/dev.py fix       # Fix lint issues
uv run python scripts/dev.py test      # Run tests

# Before committing:
uv run python scripts/dev.py full      # Ensure CI/CD will pass
```

## Error Handling

The `scripts/dev.py` includes graceful error handling:

- **Missing tools**: Warns and continues instead of failing
- **Detailed output**: Shows exact commands and exit codes
- **Summary report**: Clear ✅/❌ status for all checks
- **CI compatibility**: Same commands work locally and in GitHub Actions
