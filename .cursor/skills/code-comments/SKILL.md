---
name: code-comments
description: House style for writing code comments (and internal docstrings) in this repo. Use whenever you add or edit a `#` comment or a non-rendered docstring in src/. Comments must be standalone and timeless — written for a stranger reading the code cold, never a log of the conversation, review round, or edit that produced them.
---

# Code Comments

A comment exists to tell a future reader something the code cannot: **why** it is
this way, what constraint or tradeoff it is honoring, or what footgun it is
avoiding. Write every comment so it stands on its own for someone reading the
file cold — no access to this conversation, the PR, the review, or the diff that
produced it.

(For rendered API doc strings — route `summary=`, route docstrings, `Field`/
`Path`/`Query` descriptions — follow [api-writing-style](../api-writing-style/SKILL.md)
instead; this skill is for `#` comments and internal/non-rendered docstrings.)

## The rules

1. **Comment the why, not the what.** The code already says what it does. A
   comment that restates it is noise. Explain the non-obvious reason, the
   invariant, or the reason a simpler-looking approach is wrong.

2. **Standalone and timeless.** Write the fact, not the story of arriving at it.
   A reader six months later has none of the context you have now. Banned:
   - Conversational / review artifacts: "we", "as discussed", "the reviewer
     flagged", "per the feedback", "as requested".
   - Change narration: "previously X, now Y", "changed to…", "was moved",
     "introduced here", "on this branch", "computed lower down / above".
   - Pointers to the moment of writing: "for now", "recently", "the new…".
   State the current behavior and its reason as a permanent fact.

3. **Don't narrate the edit or the history.** "Kept a subclass instead of
   changing the base" → say what the subclass *is for*. Keep a prior state only
   when a future editor must respect it, and then phrase it as a live rule
   ("must stay Optional so the compact projection can null it"), not as a diff.

4. **One idea, fewest words.** Cut hedging, restatement, and throat-clearing. Aim
   for one line; a multi-line comment must earn every clause. Delete anything the
   type signature, the function name, or the next line of code already conveys.

5. **Keep the decision-critical bit.** If there is a real tradeoff, a
   non-obvious constraint, or a reason not to "simplify" the code, that is
   exactly what the comment is for — keep it, tightly. Losing it is worse than
   verbosity.

6. **Cross-reference by name, not by story.** "Mirrors `_row_agreement`" beats
   "see the run endpoint where we explain the rationale for this". Point at the
   symbol; let the reader jump.

## Before / after

```python
# BAD — narrates the conversation and the edit, restates the code
# Keep only failing cases (`passed is False`). A case that errored comes back
# `False` too, so it's included. `passed is None` means the case hasn't finished
# yet (pending placeholder) — NOT a failure — so a client polling mid-run doesn't
# see unfinished cases as problems. See the run endpoint for rationale.
data["results"] = [r for r in data["results"] if r.get("passed") is False]

# GOOD — the one fact a reader needs, stated permanently
# `passed is None` is a pending case, not a failure — exclude it so a mid-run
# poll doesn't report unfinished cases as failures.
data["results"] = [r for r in data["results"] if r.get("passed") is False]
```

```python
# BAD — history + restatement
# `total_items` and the `paged_items` slice are computed LOWER DOWN — after the
# optional `disagreement_only` item filter — so paging covers the disagreeing set.

# GOOD — the invariant, once, where the slice happens
# Slice after the disagreement filter so `total` and paging cover only the
# matching items.
```

```python
# BAD — explains why it's a subclass as a decision story
# Version model for the evaluator-detail endpoint's `versions[]`. Kept a subclass
# (not a change to the base) so only that endpoint's contract advertises
# `system_prompt` as nullable, while the always-full endpoints keep it required.
# Deliberately no docstring — it would become the public schema description.

# GOOD — what it's for + the one constraint to preserve
# Compact-mode shape for GET /evaluators/{uuid}: `system_prompt` is nullable here
# only. The base model stays required so the always-full endpoints keep it.
# No docstring: Pydantic would publish it as the schema description.
```

## When NOT to comment

If tightening a comment down to its real point leaves nothing the code doesn't
already say, delete it. A well-named function and a clear line need no gloss.
