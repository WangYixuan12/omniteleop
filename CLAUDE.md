# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## AI Guidance — Workflow

### Gather Context
- Think before writing. Inspect relevant files, existing patterns, and comments to match project conventions.

### Plan
- Proactively check edge cases. When inputs, specs, or constraints are unclear, pause and ask clarifying questions to confirm intent.

### Implement
- Refer to /home/yixuan/omniteleop/dexcontrol (also in conda env dexmate) to understand, but do not touch it.
- Validate all input dimensions aggressively. Raise ValueError on any mismatch. Never use placeholder/dummy data (zeros, None, etc.) to make code "run".
- Implement functions one at a time. Empirically verify the output (e.g. rendering point clouds, displaying images, or inspecting tensor statistics). Let me confirm the transformation is correct before we proceed.
- Prefer small, incremental contributions over sweeping refactors.

### Verify
- Review correctness, performance implications, and edge cases.
- For model training and evaluation, ensure the entire pipeline — including data loading, preprocessing, training, metrics, and checkpoints — is functional and compatible with the latest changes.

### Summarize
- What changed, why, and how to run.
- Risks, edge cases, and mitigations.
- Propose follow-ups or better approaches when applicable.