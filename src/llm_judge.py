"""Helpers for shaping evaluator definitions for the calibrate CLI.

Handles `{{variable}}` substitution and rubric assembly. The actual LLM-as-judge
HTTP call lives in the calibrate CLI — this module only renders prompts and
builds the config payload the CLI expects.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_VARIABLE_PATTERN = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


def render_template(template: str, variables: Dict[str, Any]) -> str:
    """Replace `{{name}}` in `template` with values from `variables`. Missing vars render as ''."""

    def _sub(match: re.Match) -> str:
        name = match.group(1)
        value = variables.get(name, "")
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    return _VARIABLE_PATTERN.sub(_sub, template)


def _format_scale_rubric(output_config: Optional[Dict[str, Any]]) -> str:
    """If any scale entry has a `description`, return a multi-line rubric block the judge
    can read to understand what each value means. Empty string when no descriptions present.

    Format:
        Rubric:
          1 (Poor): Garbled or missing content.
          2 (Weak): Significant issues.
          ...
    """
    scale = (output_config or {}).get("scale")
    if not isinstance(scale, list):
        return ""
    lines = []
    for entry in scale:
        desc = entry.get("description")
        if not desc:
            continue
        value = entry.get("value")
        name = entry.get("name")
        label = f"{value}" + (f" ({name})" if name else "")
        lines.append(f"  {label}: {desc}")
    if not lines:
        return ""
    return "\n\nRubric:\n" + "\n".join(lines)


def _scale_bounds(output_config: Optional[Dict[str, Any]]) -> tuple[Optional[float], Optional[float]]:
    """For rating evaluators, derive scale_min/scale_max from output_config.scale.

    Calibrate constrains the judge to integers in [scale_min, scale_max] (Pydantic Literal).
    """
    scale = (output_config or {}).get("scale")
    if not isinstance(scale, list) or not scale:
        return (None, None)
    numeric_values = [e.get("value") for e in scale if isinstance(e.get("value"), (int, float))]
    if not numeric_values:
        return (None, None)
    return (min(numeric_values), max(numeric_values))


def _calibrate_evaluator_def(
    ev: Dict[str, Any],
    rendered_prompt: str,
) -> Dict[str, Any]:
    """Shape a single evaluator dict into calibrate's expected contract."""
    out: Dict[str, Any] = {
        "name": ev.get("name"),
        "system_prompt": rendered_prompt,
        "judge_model": ev.get("judge_model"),
        "type": ev.get("output_type", "binary"),
    }
    if ev.get("uuid"):
        out["id"] = ev["uuid"]
    if out["type"] == "rating":
        scale_min, scale_max = _scale_bounds(ev.get("output_config"))
        if scale_min is not None and scale_max is not None:
            out["scale_min"] = scale_min
            out["scale_max"] = scale_max
    return out


def _render_with_rubric(
    ev: Dict[str, Any],
    extra_vars: Optional[Dict[str, Any]] = None,
) -> str:
    """Substitute {{variable}} placeholders in the evaluator's system_prompt and append the
    per-level rubric block when scale entries have descriptions."""
    variables_spec = ev.get("variables") or []
    variable_values = dict(ev.get("variable_values") or {})
    if extra_vars:
        variable_values.update(extra_vars)
    render_vars: Dict[str, Any] = {}
    for spec in variables_spec:
        name = spec.get("name")
        if not name:
            continue
        render_vars[name] = variable_values.get(name, spec.get("default", ""))
    for k, v in variable_values.items():
        render_vars.setdefault(k, v)

    rendered = render_template(ev.get("system_prompt", ""), render_vars)
    rubric = _format_scale_rubric(ev.get("output_config"))
    if rubric:
        rendered = rendered.rstrip() + rubric
    return rendered


def build_evaluator_cli_payload(
    evaluators: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Render each linked evaluator into the dict shape calibrate expects.

    Use for STT, TTS, and simulation runs — variables are pre-substituted into the
    system_prompt because those flows don't have a per-row arguments mechanism. For LLM
    tests/benchmarks, use `build_test_evaluators_payload` instead so each test case can
    pass its own `arguments`.
    """
    payload = []
    for ev in evaluators:
        rendered_prompt = _render_with_rubric(ev)
        payload.append(_calibrate_evaluator_def(ev, rendered_prompt))
    return payload


def build_test_evaluators_payload(
    tests_with_evaluators: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """For LLM tests/benchmarks (deduped top-level evaluators + per-test criteria refs).

    Args:
        tests_with_evaluators: list of {"test_uuid", "evaluators": [evaluator-link-dicts]}
            where each evaluator-link is the row produced by `get_evaluators_for_test`.

    Returns:
        (top_level_evaluators, criteria_per_test)

        - `top_level_evaluators` — unique evaluator definitions (deduped by evaluator uuid).
          Each entry's `system_prompt` is **NOT** rendered with variables; the rubric block
          (per-level scale descriptions) is still appended. Calibrate substitutes
          `{{variable}}` placeholders per test case using each criterion's `arguments`.

        - `criteria_per_test` — `{test_uuid: [{"name", "arguments"?}]}` references that go
          into each test case's `evaluation.criteria`.

    Names must be unique across the run (calibrate keys outputs by name); when two evaluators
    share the same `name`, a `-{shortuuid}` suffix is appended to disambiguate.
    """
    name_for_uuid: Dict[str, str] = {}
    used_names: set = set()
    top_level: List[Dict[str, Any]] = []
    criteria_per_test: Dict[str, List[Dict[str, Any]]] = {}

    for entry in tests_with_evaluators:
        test_uuid = entry["test_uuid"]
        criteria_refs: List[Dict[str, Any]] = []
        for ev in entry.get("evaluators") or []:
            evaluator_uuid = ev.get("uuid")
            if not evaluator_uuid:
                continue

            if evaluator_uuid not in name_for_uuid:
                base_name = ev.get("name") or evaluator_uuid
                name = base_name
                if name in used_names:
                    name = f"{base_name}-{evaluator_uuid[:8]}"
                used_names.add(name)
                name_for_uuid[evaluator_uuid] = name

                # Top-level definition: leave {{variable}} placeholders unrendered so calibrate
                # can substitute per-test arguments. Rubric block IS appended.
                rendered_prompt = ev.get("system_prompt", "")
                rubric = _format_scale_rubric(ev.get("output_config"))
                if rubric:
                    rendered_prompt = rendered_prompt.rstrip() + rubric

                ev_with_name = {**ev, "name": name}
                top_level.append(_calibrate_evaluator_def(ev_with_name, rendered_prompt))

            ref: Dict[str, Any] = {"name": name_for_uuid[evaluator_uuid]}
            arguments = ev.get("variable_values") or {}
            if arguments:
                ref["arguments"] = arguments
            criteria_refs.append(ref)

        criteria_per_test[test_uuid] = criteria_refs

    return top_level, criteria_per_test


