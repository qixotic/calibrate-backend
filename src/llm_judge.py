"""LLM-as-judge invocation for evaluator version prompts.

Handles `{{variable}}` substitution and routes calls to OpenRouter. Supports both
`single` (one output judged) and `side_by_side` (multiple outputs compared) evaluator kinds,
and both `binary` and `rating` output types.
"""

import base64
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_TIMEOUT_SECONDS = 120

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


def _build_judgement_instruction(
    output_type: str,
    kind: str,
    output_config: Optional[Dict[str, Any]],
) -> str:
    """Instruction appended to the judge prompt telling it exactly what JSON to return.

    Includes a per-level rubric block when scale entries have descriptions.
    """
    scale = (output_config or {}).get("scale")
    rubric = _format_scale_rubric(output_config)

    if kind == "side_by_side":
        return (
            rubric
            + "\n\nReturn a JSON object of the form "
            '{"winner": "<label>", "reasoning": "..."}. '
            "`winner` must be the label of the best output (or \"tie\"). "
            "Do not include any text outside the JSON object."
        )
    if output_type == "rating":
        scale_txt = ""
        if isinstance(scale, list):
            scale_txt = " Scale: " + ", ".join(
                f"{entry.get('value')}={entry.get('name', '')}" for entry in scale
            )
        return (
            rubric
            + "\n\nReturn a JSON object of the form "
            '{"value": <number>, "reasoning": "..."}.'
            + scale_txt
            + " Do not include any text outside the JSON object."
        )
    # binary
    return (
        rubric
        + "\n\nReturn a JSON object of the form "
        '{"pass": true|false, "reasoning": "..."}. '
        "Do not include any text outside the JSON object."
    )


def _parse_json_response(text: str) -> Dict[str, Any]:
    """Best-effort JSON parse: strip code fences, extract the first {...} block if needed."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9]*\n", "", cleaned)
        cleaned = re.sub(r"\n```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def invoke_evaluator(
    version: Dict[str, Any],
    evaluator: Dict[str, Any],
    variables: Dict[str, Any],
    output: Optional[str] = None,
    outputs: Optional[List[Dict[str, str]]] = None,
    audio_base64: Optional[str] = None,
    audio_mime: str = "audio/wav",
) -> Dict[str, Any]:
    """Invoke an evaluator version against concrete inputs.

    For kind=single: `output` is the content being judged.
    For kind=side_by_side: `outputs` is a list of {"label": ..., "content": ...} dicts.
    For data_type=audio: `audio_base64` + `audio_mime` carry the audio clip (ignored if
    `output` is also provided). `evaluator_type` (tts|stt|llm|simulation) is the semantic
    category and is independent of the medium — only `data_type` controls audio routing.

    Returns the parsed JSON judgement plus raw text, e.g.
    {"pass": true, "reasoning": "...", "raw": "..."}.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")

    kind = evaluator.get("kind", "single")
    data_type = evaluator.get("data_type", "text")
    output_type = evaluator.get("output_type", "binary")
    # output_config lives on the version — it's the rubric frozen at link time.
    output_config = version.get("output_config")

    # Build template variables
    template_vars = dict(variables or {})
    if kind == "single" and output is not None and "output" not in template_vars:
        template_vars["output"] = output
    if kind == "side_by_side" and outputs is not None:
        template_vars.setdefault(
            "outputs",
            "\n\n".join(f"[{o['label']}]\n{o['content']}" for o in outputs),
        )

    rendered_prompt = render_template(version["system_prompt"], template_vars)
    rendered_prompt += _build_judgement_instruction(output_type, kind, output_config)

    user_content: Any
    if data_type == "audio" and audio_base64:
        user_content = [
            {"type": "text", "text": "Please evaluate the attached audio."},
            {
                "type": "input_audio",
                "input_audio": {"data": audio_base64, "format": audio_mime.split("/")[-1]},
            },
        ]
    else:
        user_content = "Please return the judgement now."

    payload = {
        "model": version["judge_model"],
        "messages": [
            {"role": "system", "content": rendered_prompt},
            {"role": "user", "content": user_content},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }

    with httpx.Client(timeout=OPENROUTER_TIMEOUT_SECONDS) as client:
        response = client.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    raw_text = data["choices"][0]["message"]["content"]
    try:
        parsed = _parse_json_response(raw_text)
    except json.JSONDecodeError:
        logger.warning(f"Judge returned non-JSON output: {raw_text!r}")
        parsed = {"reasoning": raw_text}

    parsed["raw"] = raw_text
    parsed["judge_model"] = version["judge_model"]
    return parsed


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


def encode_audio_from_url(url: str) -> tuple[str, str]:
    """Download an audio URL (http(s) or local path) and return (base64, mime)."""
    if url.startswith("http://") or url.startswith("https://"):
        with httpx.Client(timeout=60) as client:
            resp = client.get(url)
            resp.raise_for_status()
            mime = resp.headers.get("Content-Type", "audio/wav").split(";")[0].strip()
            return base64.b64encode(resp.content).decode("ascii"), mime
    with open(url, "rb") as f:
        data = f.read()
    ext = os.path.splitext(url)[1].lstrip(".").lower() or "wav"
    return base64.b64encode(data).decode("ascii"), f"audio/{ext}"
