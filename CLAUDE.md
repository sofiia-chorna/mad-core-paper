## Key conventions / gotchas
- **Reproducibility**: every HPC run gets its own committed self-contained `submit.sh` (not ad-hoc).
- **Never `git commit`/`push`** — Sofiia does that; only `git add` + advise.
- **Long login-node jobs**: run detached (`setsid nohup … &`). Note CSCS rotates login nodes, so a
  nohup proc lives on one node and may be invisible from a later session (the SLURM jobs are unaffected).
General rules:
# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

Don't write comments in the code unless requested. No commit or push allowed.
Write code with clear spaces between the lines (logical) and give good variable names
Keep all imports on top
never mention anyone's names

/capstor/store/cscs/swissai/aa002/ - never modify anything without asking and examplaing what you do

## 5. Code Style

- All imports at the top of the file, never inside functions.
- No descriptive comment or docstring at the top of a file. Usage and descriptions go in the README.
- No docstrings on functions. Use a short `#` comment above the code instead, only where it helps.
- Comments are minimal, short and start with a lowercase letter.
- Comments use plain words only: no `;`, no colons, no arrows, math or other special symbols.
- Code is not dense. Separate logical steps with blank lines, give each step its own line,
  and break long calls or literals over several lines.
- The same rules apply to bash and Slurm scripts. `#!` and `#SBATCH` lines are not comments.
  Write `if` blocks over several lines instead of `if ...; then ...; fi` or `a; b` on one line.

no commits allowed
