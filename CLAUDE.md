# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## AI Guidance — Workflow

### Gather Context
- Think before writing. Inspect relevant files, existing patterns, and comments to match project conventions.

### Plan
- Proactively check edge cases.
- Proactively ask clarifying questions to confirm intent, until inputs, specs, and constraints are clear. Do not assume anything ambiguous.
- If multiple interpretations exist, present them — do not pick one silently. If a simpler approach exists, say so and push back.

### Implement
- Refer to /home/yixuan/omniteleop/dexcontrol (also in conda env dexmate) to understand, but do not touch it.
- Validate all input dimensions aggressively. Raise ValueError on any mismatch. Never use placeholder/dummy data (zeros, None, etc.) to make code "run".
- Implement functions one at a time. Empirically verify the output (e.g. rendering point clouds, displaying images, or inspecting tensor statistics). Let me confirm the transformation is correct before we proceed.
- Prefer small, incremental contributions over sweeping refactors.
- Write the minimum code that solves the problem. No speculative features, no abstractions for single-use code, no unrequested configurability, no error handling for impossible scenarios.
- Keep changes surgical: every changed line should trace to the request. Do not "improve" adjacent code, reformat, or refactor what is not broken. Match existing style.
- Remove imports/variables/functions that YOUR changes orphaned; mention pre-existing dead code rather than deleting it.

### Verify
- Review correctness, performance implications, and edge cases.
- Turn the task into a verifiable goal before starting ("fix the bug" → "write a test that reproduces it, then make it pass") and loop until it passes.
- For model training and evaluation, ensure the entire pipeline — including data loading, preprocessing, training, metrics, and checkpoints — is functional and compatible with the latest changes.

### Summarize
- What changed, why, and how to run.
- Risks, edge cases, and mitigations.
- Propose follow-ups or better approaches when applicable.

## Using Codex

- Proactively delegate to `codex:rescue` — don't wait to be asked — when stuck, when a second implementation or diagnosis pass would help, or for deep root-cause investigations and substantial coding tasks.
- Also delegate bulk or mechanical work proactively: clear-spec implementation, data analysis, and migrations.
- Use your own judgment on when a Codex review adds value (e.g. non-trivial or risky changes) and invoke `/codex:review` then.

## Agent skills

### Issue tracker

Issues and specs live as markdown files under `.scratch/<feature-slug>/`; this repo's GitHub Issues are disabled. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical roles, default strings, written on each issue file's `Status:` line. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context — `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
