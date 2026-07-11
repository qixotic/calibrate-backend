---
name: api-docs-audit
description: Audit every rendered API doc string changed on the current branch/diff against the api-writing-style house style, and publish a field-level HTML conformance report (grouped by file then method, one row per param/field/summary/docstring, with verdict, reasoning, and a suggested fix). Use before claiming any change that touched route summaries, docstrings, or Field/Path/Query descriptions is done.
---

# API Docs Audit

Produce a verifiable, field-level conformance report for **every rendered API
doc string** a change touches. "Rendered" = anything that reaches `/docs`,
`/redoc`, or the public OpenAPI spec (and therefore the generated SDK/CLI/MCP).
The audit checks each string against [api-writing-style](../api-writing-style/SKILL.md)
and hands back a scannable HTML report so a reviewer can see, per field, what was
written, whether it passes, why, and the exact fix if not.

This complements the mechanical gate ([check_api_docs_style.py](../../../scripts/check_api_docs_style.py) /
`tests/test_api_docs_style.py`), which only catches the machine-checkable subset
(em-dashes, semicolons, unit echoes, a few banned phrases). The audit covers the
**judgment-call** rules the gate can't — explanatory parentheticals, per-X
phrasing, null caveats, purpose-first phrasing, enum re-listing, cryptic labels,
consistency across models — which is exactly where real violations hide.

## When to run it

Run it whenever a change added or edited any of the in-scope strings below and
**before reporting that change done**. It is required for public-API doc changes
and recommended for any route/model doc change. It does not replace the
mechanical test — run both.

## What is in scope (audit these)

Only strings that render into the API docs / OpenAPI:

- Route `summary=`
- Route function **docstrings** (the endpoint description)
- `Field(description=...)` on every request/response **Pydantic model** field
- `Path(...)` / `Query(...)` `description=` on path and query **params**
- **Pydantic model class docstrings** (Pydantic renders these as the schema
  description) — if a model has one, audit it
- **Factory-generated** descriptions (e.g. `make_projection_params` builds the
  `?compact` description at runtime): render the actual string the endpoint
  ships and audit that, not the f-string source

## What is OUT of scope (skip these)

- `#` code comments (never render)
- Docstrings on **non-route helper / private functions** (`_foo`, utilities in
  `pagination.py`, `db.py`, etc.) — they don't reach the spec
- Variable names, log messages, test text

## Procedure

### 1. Inventory the changed strings

Diff against the base branch (usually `main`) and extract the in-scope strings
the change added or modified. Start from:

```
git diff <base>...HEAD -- src/ | grep -E '^\+[^+]' | grep -iE 'summary=|description=|Query\(|Field\(|Path\(|"""'
```

Then read each hit in context to decide scope: keep route summaries/docstrings,
model-field/param descriptions, model class docstrings, and factory-generated
descriptions; drop `#` comments and private-helper docstrings. For a
factory-generated string, read the factory and reconstruct the exact rendered
text (with a realistic field list). Group the survivors by **file → method /
model**.

### 2. Verify each string (one sub-agent per string)

For each in-scope string, launch **one sub-agent** (run them in parallel — they
are independent). Give each agent: the file+line, the exact string, its context
(which route/model, param vs field vs summary vs docstring, public or JWT-only),
and instruct it to:

1. Read [api-writing-style/SKILL.md](../api-writing-style/SKILL.md) in full.
2. Audit the string against **every applicable dimension**, at minimum:
   em-dashes, clause-splitting semicolons, trailing period on `description=`,
   **explanatory parentheticals** (only bare unit/format/enum/example allowed),
   **per-X phrasing** ("per-model" → "for each model"), null caveats (the `|null`
   chip already conveys it), repeated units, internal/undocumented concepts,
   purpose-first + plain-not-cryptic phrasing, enum/`Literal` value re-listing,
   second person + ID-not-UUID + workspace-not-org terminology, optional
   params/fields explaining what omission does, summary verb vocabulary
   (`Get`/`List`/`Create`/…, imperative, no period), consistency with sibling
   params and with a base model when the string is a subclass override.
3. Return, per dimension, PASS/FAIL + one-line reason, then an **OVERALL VERDICT
   = ADHERES | VIOLATES**, and for VIOLATES the exact **suggested replacement**.

### 3. Publish the HTML report

Copy [report-template.html](report-template.html) and fill it in, then publish
with the **Artifact** tool (load the `artifact-design` skill first per its own
rules). The report is **grouped by file, then by method/model**, with **one table
row per audited string**. Each row's columns are fixed:

| Column | Content |
|---|---|
| Target | the param/field name + kind (query param, field, route title, route description, class docstring) |
| What is written | the exact rendered string, in a mono block |
| Verdict | a Pass / Fail pill |
| Reasoning | why, with the violated rule name tagged on failures |
| Suggested change | the corrected string on Fail rows; `—` on Pass rows |

Fill the four scorecards at the top: strings audited, adhere, violate, files.
The template is theme-aware and self-contained (system fonts, mono for the doc
strings) — do not add external assets. Keep the `<title>` and favicon stable
across re-runs of the same audit.

### 4. Report the outcome

State the pass/violate count and list the violating fields with their fixes in
the chat too (the Artifact isn't shown inline). If anything violated, offer to
apply the fixes (and update the matching test descriptions). Applying the fixes
is a separate step — the audit only reports.

## Notes

- **Generalize every fix.** A violation is a class, not an instance: when you fix
  one, grep `src/routers/*.py` for the same pattern and fix all of them, per the
  api-writing-style "generalize every fix" rule.
- Re-run the audit after applying fixes to confirm the report comes back clean.
- The audit reads the diff, so it naturally scopes to the current change; to
  audit the whole surface instead, point step 1 at the full file set.
