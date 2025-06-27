---
allowed-tools: Bash(gh:*), Bash(git:*)
description: Complete issue-driven development workflow with TDD and subagent review
---

# Issue-Driven Development Workflow

Current branch: !`git branch --show-current`
Git status: !`git status --porcelain`

Analyze and implement a solution for GitHub issue: $ARGUMENTS

## PLAN

1. **Analyze**: `gh issue view $ARGUMENTS` to understand the problem
2. **Research**: Search scratchpads, PRs, and codebase for context
3. **Break down**: Split into small, manageable tasks  
4. **Document**: Create `scratchpad-issue-{issue-number}-plan.md` with your approach

## CODE  

1. **Branch**: `git checkout -b fix/issue-{issue-number}-{description}`
2. **TDD**: Write tests first, then implement to pass them
3. **Implement**: Work in small commits following your plan
4. **Quality check**: Use this approach for robust solutions:

> "Please write a high quality, general purpose solution. Implement a solution that works correctly for all valid inputs, not just the test cases. Do not hard-code values or create solutions that only work for specific test inputs. Instead, implement the actual logic that solves the problem generally.
>
> Focus on understanding the problem requirements and implementing the correct algorithm. Tests are there to verify correctness, not to define the solution. Provide a principled implementation that follows best practices and software design principles."

## TEST

1. **Run tests**: Ensure all tests pass and no regressions
2. **Manual testing**: Verify the issue requirements are met
3. **Subagent review**: Create a new conversation and ask a fresh Claude instance to review your implementation for:
   - Code quality and best practices
   - Whether the solution overfits to specific test cases
   - Potential edge cases or improvements

## DEPLOY

1. **Self-review**: Check your changes thoroughly
2. **PR**: `gh pr create` with clear description linking to issue
3. **Monitor**: Watch for CI results and review feedback

**Always use `gh` CLI for GitHub operations. Update your scratchpad throughout the process.**
