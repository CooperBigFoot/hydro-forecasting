---
allowed-tools: Bash(gh:*), Bash(git:*)
description: Smart PR merge with quality checks and cleanup
---

# Merge Pull Request

Current branch: !`git branch --show-current`
PR status: !`gh pr status --json state,statusCheckRollup,reviewDecision`

Merge current branch PR or specified PR number: $ARGUMENTS

## ARGUMENT HANDLING

**Determine target PR:**

- **No arguments**: Merge PR for current branch (`gh pr merge`)
- **PR number**: Merge specific PR (`gh pr merge $ARGUMENTS`)
- **Merge strategy flag**: Apply strategy (`gh pr merge --$ARGUMENTS`)
- **Multiple args**: Parse PR number and strategy from $ARGUMENTS

**Valid Arguments:**

- `123` - Merge PR #123
- `--squash` - Force squash merge strategy
- `--merge` - Force merge commit strategy  
- `--rebase` - Force rebase merge strategy
- `123 --squash` - Merge PR #123 with squash strategy

## PRE-MERGE VALIDATION

1. **Check PR status**: Verify all CI checks passed and required reviews approved
2. **Validate merge readiness**: Ensure no conflicts and branch is up-to-date
3. **Review strategy**: Auto-select merge strategy based on team preferences
4. **Final checks**: Security scan, no draft status, linked issues resolved

## SMART MERGE STRATEGY

- **Squash merge**: Clean, linear history with single commit per feature
- **Auto-cleanup**: Delete branch after successful merge
- **Issue linking**: Automatically close linked issues

**Strategy Selection:**

```bash
# Default: squash for cleaner history
gh pr merge --squash --delete-branch

# Alternative: preserve commit history
gh pr merge --merge --delete-branch

# For linear history: rebase commits
gh pr merge --rebase --delete-branch
```

## MERGE EXECUTION

1. **Auto-merge if ready**: `gh pr merge --auto` when checks pending
2. **Manual merge**: Direct merge when all requirements met
3. **Cleanup branches**: Remove local and remote feature branches
4. **Update main**: Ensure local main branch is current

## POST-MERGE ACTIONS

1. **Verify merge**: Confirm successful integration to main branch
2. **Update local**: `git checkout main && git pull origin main`
3. **Clean workspace**: Remove merged branch locally
4. **Update scratchpad**: Record completion and any follow-up tasks
5. **Notify team**: Optional notification of successful deployment

## QUALITY SAFEGUARDS

- **CI/CD Integration**: Only merge when all automated checks pass
- **Review Requirements**: Respect branch protection rules
- **Conflict Detection**: Alert if manual conflict resolution needed
- **Rollback Plan**: Document process for emergency rollbacks

**Best Practice**: Squash merge keeps history clean and makes rollbacks easier.
