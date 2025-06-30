# Streamflow Mapper - Claude Code Guide

> Comprehensive development guide for the streamflow-mapper project using Claude Code with modern Python tooling.

## Quick Start

The most important commands you'll use daily:

```bash
# Setup new environment
uv sync --dev

# Full CI/CD pipeline check (run before commits)
uv run python scripts/dev.py full

# Fast iteration cycle
uv run python scripts/dev.py format && uv run python scripts/dev.py fix
uv run python scripts/dev.py test
```

## Project Architecture

### Core Technology Stack

**Primary Tools:**

- **UV**: Modern Python package and project manager (replaces pip, poetry, pyenv)
- **RUFF**: Fast Python linter and formatter (replaces Flake8, Black, isort)  
- **TY**: Type checker from Astral (⚠️ Alpha version - development use only)

**Why this stack:** UV provides faster dependency resolution and virtual environment management. RUFF offers significantly faster linting/formatting than traditional tools. TY provides type checking with better performance characteristics than mypy.

### Project Structure

```bash
streamflow-mapper/
├── src/                    # Source code
├── tests/                  # Test suite
├── scripts/
│   └── dev.py             # Development orchestration script
├── scratchpads/           # Claude Code working documents
│   ├── issues/           # Issue-specific planning
│   ├── planning/         # Feature development
│   └── research/         # Technical exploration
├── .claude/
│   └── commands/         # Custom slash commands
└── pyproject.toml        # Project configuration
```

## Development Orchestration Script

The `scripts/dev.py` script mirrors our CI/CD pipeline locally and should be your primary development tool.

### Core Commands

| Command | Purpose | When to Use |
|---------|---------|-------------|
| `uv run python scripts/dev.py full` | Run complete CI/CD pipeline | Before commits, PRs, releases |
| `uv run python scripts/dev.py format` | Format code with Ruff | During active development |
| `uv run python scripts/dev.py check-format` | Check formatting without changes | CI verification |
| `uv run python scripts/dev.py lint` | Lint code with Ruff | Code quality checks |
| `uv run python scripts/dev.py fix` | Auto-fix linting issues | Quick issue resolution |
| `uv run python scripts/dev.py test` | Run pytest suite | Feature verification |
| `uv run python scripts/dev.py test-cov` | Run tests with coverage | Coverage analysis |
| `uv run python scripts/dev.py install` | Install/sync dependencies | Environment setup |

### Full Pipeline Details

The `full` command executes these steps in order:

1. **Dependency Sync**: `uv sync --dev` - Ensures consistent environment
2. **Format Validation**: `ruff format --check --diff` - Enforces code style
3. **Linting**: `ruff check` - Catches code quality issues  
4. **Test Suite**: `pytest tests/ -v` - Verifies functionality

**Output**: Clear ✅/❌ summary showing pipeline status

## Package Management with UV

**Important**: Use UV exclusively for all Python package operations. Never use pip, pip-tools, poetry, or conda directly.

### Essential UV Commands

```bash
# Environment Management
uv sync                    # Install from lockfile
uv sync --dev             # Include development dependencies
uv sync --all-extras      # Include all optional dependencies
uv venv                   # Create virtual environment
uv python install         # Install Python from .python-version

# Dependency Management  
uv add <package>          # Add production dependency
uv add --dev <package>    # Add development dependency
uv add --optional <group> <package>  # Add to optional dependency group
uv remove <package>       # Remove dependency
uv lock                   # Update lockfile
uv lock --upgrade-package <pkg>  # Update specific package

# Code Execution
uv run <command>          # Run command in project environment
uv run python script.py   # Execute Python script
uv run pytest            # Run test suite
uv run ruff check        # Run linter

# PEP 723 Script Management
uv add package-name --script script.py     # Add dependency to script
uv remove package-name --script script.py  # Remove dependency from script
```

## Code Quality Tools

### RUFF Configuration

Our project uses RUFF with these settings:

- **Line length**: 88 characters
- **Rule sets**: E (pycodestyle errors), F (pyflakes), W (warnings), N (naming), I (import sorting), UP (pyupgrade), B (bugbear), C4 (comprehensions), SIM (simplifications)

```bash
# Linting Workflow
uv run ruff check              # Check all files
uv run ruff check --fix        # Auto-fix issues where possible
uv run ruff check --watch      # Continuous checking
uv run ruff check <file>       # Check specific file

# Formatting Workflow  
uv run ruff format             # Format all files
uv run ruff format --check     # Verify formatting only
uv run ruff format <file>      # Format specific file

# Recommended Combined Workflow
uv run ruff check --fix && uv run ruff format
```

### TY Type Checking (Development Only)

**Note**: TY is in alpha/preview. Use for development feedback but not in CI/CD.

```bash
uv run ty                      # Check entire project
uv run ty <file>              # Check specific file  
uv run ty --watch             # Continuous type checking
uv run ty --python-version 3.10  # Specify Python version
```

## Development Workflows

### Starting New Work

```bash
# Branch creation and setup
git checkout -b feature/descriptive-name
uv sync --dev

# Check for existing context
ls scratchpads/issues/        # Look for related issue work
ls scratchpads/planning/      # Check feature planning docs
```

### Active Development Cycle

**Fast Iteration** (use during coding):

```bash
# Code formatting and basic fixes
uv run python scripts/dev.py format
uv run python scripts/dev.py fix

# Run relevant tests
uv run pytest tests/test_specific.py -v
uv run pytest tests/ -k "test_function_name"
```

**Continuous Feedback** (run in separate terminals):

```bash
# Terminal 1: Continuous type checking
uv run ty --watch

# Terminal 2: Continuous linting  
uv run ruff check --watch

# Terminal 3: Test watching (if using pytest-watch)
uv run ptw tests/
```

### Pre-Commit Validation

**Required before commits**:

```bash
# Full pipeline check (mirrors CI/CD exactly)
uv run python scripts/dev.py full

# Manual alternative if needed
uv run ruff check --fix && uv run ruff format && uv run ty && uv run pytest
```

**Commit best practices**:

- Ensure all checks pass locally before pushing
- Use conventional commit messages: `feat:`, `fix:`, `docs:`, `refactor:`
- Reference issues: `Closes #123` or `Fixes #456`

## Scratchpad System

Use structured markdown files for complex work planning and documentation.

### Directory Structure

```bash
scratchpads/
├── issues/          # Issue-specific work (/project:fix-issue command)
├── planning/        # Feature planning and design
└── research/        # Technical exploration and spikes
```

### Naming Convention

`scratchpads/{type}/{brief-description}.md`

Examples:

- `scratchpads/issues/fix-data-validation-bug-123.md`
- `scratchpads/planning/user-authentication-system.md`  
- `scratchpads/research/performance-optimization-analysis.md`

### Standard Template

```markdown
# [Task Name]

## Objective
[Clear description of what needs to be accomplished]

## Context
[Background information, links to issues, previous work]

## Plan
- [ ] Step 1: [Specific actionable item]
- [ ] Step 2: [Specific actionable item]
- [ ] Step 3: [Specific actionable item]

## Implementation Notes
[Code snippets, architectural decisions, API changes]

## Testing Strategy
[How to verify the solution works]

## Review Points
[Areas requiring special attention during code review]
```

## Custom Slash Commands

Available project-specific commands:

### `/project:fix-issue <issue-number>`

Complete issue-driven development workflow following TDD principles.

**Usage**: `/project:fix-issue 123`

**What it does**:

1. Analyzes GitHub issue via `gh issue view`
2. Searches codebase for relevant context
3. Creates implementation plan in scratchpad
4. Implements solution using TDD approach
5. Creates pull request with proper documentation

### `/project:create-pr [description]`  

Generate comprehensive pull request with best practices.

**Usage**: `/project:create-pr "Add user authentication"`

**Features**:

- Auto-detects linked issues from branch name
- Generates structured PR description
- Adds appropriate reviewers and labels
- Includes security and testing checklists

## CI/CD Pipeline

Our GitHub Actions pipeline (`.github/workflows/ci.yaml`) runs identical checks to `scripts/dev.py full`:

**Validation Steps**:

1. **Lint & Format Check** - Ruff validation on code style
2. **Test Suite** - pytest across multiple Python versions (3.10, 3.11, 3.12)  
3. **Build Verification** - Package build validation

**Triggers**:

- Push to main branch
- Pull requests targeting main
- Manual workflow dispatch

**Key Difference**: Type checking with TY is excluded from CI/CD due to alpha status.

## Advanced Analysis with Gemini

For comprehensive codebase analysis that exceeds Claude's context limits, use Gemini CLI:

### When to Use Gemini

- Analyzing >50 files or >100KB total codebase
- Comprehensive security audits  
- Architecture assessments across entire project
- Performance analysis of large systems
- When explicitly requested: "use gemini"

### Gemini Command Patterns

```bash
# Security analysis
gemini -p "@src/ @api/ Complete security analysis with specific vulnerabilities and fixes"

# Performance optimization
gemini -p "@src/ @config/ Identify bottlenecks with optimization strategies"  

# Architecture assessment
gemini -p "@src/ Analyze patterns and technical debt with actionable recommendations"

# Feature verification
gemini -p "@src/feature/ @tests/ Verify implementation completeness with gap analysis"

# Full project analysis
gemini --all_files -p "Comprehensive code quality assessment with prioritized recommendations"
```

### File Inclusion Syntax

- `@src/main.py` - Single file
- `@src/` - Entire directory  
- `@./` - Complete project
- `@src/ @tests/` - Multiple directories

## Extended Thinking for Complex Problems

Use thinking triggers for deep analysis:

| Trigger | Depth | Use Case |
|---------|-------|----------|
| "think" | Basic | Standard problem analysis |
| "think hard" | Deep | Complex architectural decisions |
| "think more" | Extended | Multi-faceted problem solving |
| "think harder" | Maximum | Critical system design choices |

**Best for**: Complex debugging, architectural planning, performance optimization, security analysis.

## Configuration Reference

### Python Requirements

- **Minimum Version**: Python >=3.10 (see `.python-version`)
- **Virtual Environment**: `.venv/` (managed by UV)
- **Source Layout**: `src/` directory structure

### Tool Configuration

- **RUFF**: Line length 88, comprehensive rule set (E,F,W,N,I,UP,B,C4,SIM)
- **TY**: Basic type checking with gradual adoption
- **pytest**: Test discovery in `tests/` directory
- **UV**: Lock file at `uv.lock`, dependencies in `pyproject.toml`

### Key Configuration Files

- `@pyproject.toml` - Project dependencies and tool settings
- `@.python-version` - Python version specification
- `@scripts/dev.py` - Development workflow automation
- `@.github/workflows/ci.yaml` - CI/CD pipeline configuration

## Troubleshooting

### Common Issues and Solutions

**Development script fails**:

```bash
# Reinstall dependencies
uv run python scripts/dev.py install

# Verify tool availability  
uv pip list | grep -E "(pytest|ruff|ty)"

# Check individual tools
uv run ruff --version
uv run pytest --version
uv run ty --version
```

**Import errors after adding dependencies**:

```bash
# Sync environment with lockfile
uv sync --dev

# Clear cache if needed
uv cache clean
```

**Test failures in CI but not locally**:

```bash
# Run exact CI commands locally
uv sync --dev
uv run ruff format --check --diff  
uv run ruff check
uv run pytest tests/ -v
```

**Type checking inconsistencies**:

```bash
# TY is alpha - inconsistencies expected
# Focus on runtime correctness over type perfection
# Use TY output as guidance, not strict requirements
```

### Error Handling Philosophy

The `scripts/dev.py` includes robust error handling:

- **Graceful degradation**: Warns about missing tools instead of failing
- **Detailed diagnostics**: Shows exact commands and exit codes
- **Clear reporting**: ✅/❌ status summary for all operations
- **CI compatibility**: Identical behavior locally and in GitHub Actions

## Best Practices Summary

### Daily Development

1. **Start with**: `uv sync --dev` to ensure consistent environment
2. **During coding**: Use fast iteration commands (`format`, `fix`, `test`)
3. **Before committing**: Always run `uv run python scripts/dev.py full`
4. **Use scratchpads**: Document complex work for context preservation

### Code Quality

1. **Format early and often**: Run `ruff format` frequently
2. **Fix automatically**: Use `ruff check --fix` for auto-fixable issues  
3. **Test incrementally**: Run specific tests during development
4. **Think deeply**: Use extended thinking for complex architectural decisions

### Collaboration  

1. **Document decisions**: Use scratchpads for complex reasoning
2. **Create clear PRs**: Use `/project:create-pr` for consistent formatting
3. **Link issues**: Always reference relevant GitHub issues
4. **Review thoroughly**: Focus on areas highlighted in PR descriptions
