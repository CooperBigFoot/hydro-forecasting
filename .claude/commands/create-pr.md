---
allowed-tools: Bash(gh:*), Bash(git:*)
description: Create comprehensive pull request with best practices
---

# Create Pull Request

Current branch: !`git branch --show-current`
Recent commits: !`git log --oneline -3`

Create a well-structured pull request for current branch: $ARGUMENTS

## PREPARE

1. **Review changes**: `git diff main..HEAD` and ensure all committed
2. **Check size**: Warn if >200 lines, suggest splitting for faster review
3. **Security scan**: No credentials, API keys, or sensitive data
4. **Run tests**: Verify functionality before creating PR

## ANALYZE & CREATE

1. **Auto-detect issue**: Extract from branch name or commits
2. **Generate description** with clear What/Why/How/Testing sections
3. **Smart title**: Format as `type: description (#issue)`
4. **Create draft**: `gh pr create --draft` for review before publishing

## FINALIZE

1. **Assign reviewers**: Use CODEOWNERS and recent contributors
2. **Add labels**: Based on files changed and PR type
3. **Link issues**: Ensure "Closes #123" syntax works
4. **AI enhance**: Add `/copilot summary` comment for better description
5. **Mark ready**: `gh pr ready` when satisfied

## TEMPLATE

```markdown
## What & Why
- **Problem**: [Issue being solved]
- **Solution**: [How this addresses it]

## Changes
- [Key modifications made]
- [Files changed and why]

## Testing
- [How you verified this works]
- [Edge cases considered]

## Review Focus
- [Areas needing careful attention]
```

**Best Practice**: Keep PRs small, focused, and well-documented for faster reviews.

**Update scratchpad with PR details and next steps.**
