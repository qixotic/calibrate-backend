#!/usr/bin/env python3
"""Integration-testing stand-in for the ``calibrate-agent`` eval CLI.

Enabled by ``FAKE_AI_PROVIDERS=1`` (see ``get_calibrate_agent_cli`` in
``src/utils.py``) so integration tests can drive the full run → results pipeline
with no real LLM/STT/TTS call, key, or cost. Standalone — imports nothing from
the backend. For each ``[cli, <subcommand>, ...flags]`` a worker spawns, it
writes the output files that worker's reader expects, then exits 0.

Output shapes mirror the readers in ``routers/agent_tests.py``, ``stt.py``,
``tts.py``, ``simulations.py``, and ``annotation_eval_runner.py``. The constants
below are asserted by the frontend integration tests — keep them stable.
"""

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# --- Canned constants (frontend integration tests assert these) -------------
FAKE_RESPONSE = "Simulated agent reply."
FAKE_REASONING = "Simulated judge reasoning: criteria satisfied."
FAKE_LATENCY_MS = 100
FAKE_COST = 0.001
FAKE_TOKENS = 42
FAKE_WER = 0.0
FAKE_TTFB = 0.5
# Sarvam judge bundle (included by default; suppressed by `--skip-llm-judges`).
FAKE_SARVAM_LLM_WER = 0.0
FAKE_SARVAM_LLM_CER = 0.0
# Every evaluator verdict is a PASS; every rating is scale_max.


# --- Argument parsing -------------------------------------------------------
# Flags that take no value (their presence is the signal). Everything else that
# starts with "-" consumes the following non-dash tokens as its value(s).
_BOOL_FLAGS = {"--eval-only", "--skip-verify", "--skip-llm-judges"}


def _parse_args(argv: List[str]) -> tuple[str, Dict[str, List[str]]]:
    """Return (subcommand, {flag: [values]}). Lenient by design — the real CLI
    accepts many more flags; we only need the few the workers pass."""
    sub = argv[1] if len(argv) > 1 else ""
    opts: Dict[str, List[str]] = {}
    i = 2
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("-"):
            if tok in _BOOL_FLAGS:
                opts[tok] = []
                i += 1
                continue
            vals: List[str] = []
            i += 1
            while i < len(argv) and not argv[i].startswith("-"):
                vals.append(argv[i])
                i += 1
            opts[tok] = vals
        else:
            i += 1  # stray positional — ignore
    return sub, opts


def _first(opts: Dict[str, List[str]], *keys: str) -> Optional[str]:
    for k in keys:
        vals = opts.get(k)
        if vals:
            return vals[0]
    return None


def _many(opts: Dict[str, List[str]], *keys: str) -> List[str]:
    for k in keys:
        vals = opts.get(k)
        if vals:
            return vals
    return []


def _load_json(path: Optional[str]) -> Any:
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# --- Evaluator helpers ------------------------------------------------------
# The calibrate evaluator contract keys the uuid as ``id`` and the output kind
# as ``type`` ("binary" | "rating"), with ``scale_min``/``scale_max`` on rating.
def _evaluators_from_config(config: Any) -> List[Dict[str, Any]]:
    if isinstance(config, dict):
        evs = config.get("evaluators")
        if isinstance(evs, list):
            return [e for e in evs if isinstance(e, dict)]
    return []


def _ev_uuid(ev: Dict[str, Any]) -> Optional[str]:
    return ev.get("id") or ev.get("uuid")


def _ev_scale_max(ev: Dict[str, Any]) -> Any:
    return ev.get("scale_max", 5)


def _ev_scale_min(ev: Dict[str, Any]) -> Any:
    return ev.get("scale_min", 1)


def _evaluators_map(evaluators: List[Dict[str, Any]]) -> Dict[str, str]:
    """calibrate's ``config.json`` map is ``{evaluator_uuid: name}``."""
    out: Dict[str, str] = {}
    for ev in evaluators:
        uid = _ev_uuid(ev)
        name = ev.get("name")
        if uid and name:
            out[str(uid)] = str(name)
    return out


def _write_config_json(output_dir: Path, evaluators: List[Dict[str, Any]]) -> None:
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({"evaluators_map": _evaluators_map(evaluators)}, f)


def _metrics_aggregate(ev: Dict[str, Any], n: int) -> Dict[str, Any]:
    """Per-evaluator aggregate entry for metrics.json. Must carry a ``type`` key
    so the readers recognise it as an evaluator output (not a built-in scalar)."""
    uid = _ev_uuid(ev)
    if ev.get("type") == "rating":
        sm = _ev_scale_max(ev)
        return {
            "type": "rating",
            "evaluator_id": uid,
            "mean": sm,
            "min": sm,
            "max": sm,
            "count": n,
            "scale_min": _ev_scale_min(ev),
            "scale_max": sm,
        }
    return {
        "type": "binary",
        "evaluator_id": uid,
        "passed": n,
        "total": n,
        "pass_rate": 1.0,
    }


# --- Subcommand: llm (unit-test run AND benchmark) --------------------------
def _safe_model(model: str) -> str:
    """Filesystem-safe folder name matching ``_match_model_to_folder``."""
    return model.replace("/", "__").replace(":", "_")


def _llm_judge_results(
    criteria: List[Dict[str, Any]], ev_by_name: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    judge_results: Dict[str, Any] = {}
    for ref in criteria:
        if not isinstance(ref, dict):
            continue
        name = ref.get("name")
        if not name:
            continue
        ev = ev_by_name.get(name, {})
        if ev.get("type") == "rating":
            judge_results[name] = {
                "reasoning": FAKE_REASONING,
                "score": _ev_scale_max(ev),
            }
        else:
            judge_results[name] = {"reasoning": FAKE_REASONING, "match": True}
    return judge_results


def _write_llm_model_dir(
    model_dir: Path,
    test_cases: List[Dict[str, Any]],
    evaluators: List[Dict[str, Any]],
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    ev_by_name = {ev.get("name"): ev for ev in evaluators if ev.get("name")}

    results: List[Dict[str, Any]] = []
    ref_count: Dict[str, int] = {}
    for tc in test_cases:
        if not isinstance(tc, dict):
            continue
        evaluation = tc.get("evaluation") or {}
        ev_type = evaluation.get("type")
        criteria = evaluation.get("criteria") or []
        judge_results = _llm_judge_results(criteria, ev_by_name)
        for name in judge_results:
            ref_count[name] = ref_count.get(name, 0) + 1

        output: Dict[str, Any] = {
            "response": FAKE_RESPONSE,
            "tool_calls": [],
            "cost": FAKE_COST,
        }
        if ev_type == "tool_call":
            output["tool_calls"] = evaluation.get("tool_calls") or []

        results.append(
            {
                "test_case_id": tc.get("id"),
                "test_case": tc,
                "output": output,
                "metrics": {
                    "passed": True,
                    "reasoning": FAKE_REASONING,
                    "judge_results": judge_results,
                },
                "latency_ms": FAKE_LATENCY_MS,
            }
        )

    with open(model_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f)

    n = len(results)
    criteria_agg: Dict[str, Any] = {}
    for ev in evaluators:
        name = ev.get("name")
        if not name:
            continue
        criteria_agg[name] = _metrics_aggregate(ev, ref_count.get(name, n))

    with open(model_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "total": n,
                "passed": n,
                # Aggregate perf blocks are dicts, not scalars: latency is
                # percentiles {p50,p95,p99,count}; cost/tokens are
                # {mean,min,max,count}. The run-status/benchmark readers pass
                # these straight into Optional[Dict] response fields, so a
                # scalar 500s response validation.
                "latency_ms": {
                    "p50": FAKE_LATENCY_MS,
                    "p95": FAKE_LATENCY_MS,
                    "p99": FAKE_LATENCY_MS,
                    "count": n,
                },
                "cost": {"mean": FAKE_COST, "min": FAKE_COST, "max": FAKE_COST, "count": n},
                "total_tokens": {
                    "mean": FAKE_TOKENS,
                    "min": FAKE_TOKENS,
                    "max": FAKE_TOKENS,
                    "count": n,
                },
                "criteria": criteria_agg,
            },
            f,
        )


def _write_leaderboard(
    output_dir: Path, safe_models: List[str], evaluators: List[Dict[str, Any]]
) -> None:
    leaderboard_dir = output_dir / "leaderboard"
    leaderboard_dir.mkdir(parents=True, exist_ok=True)
    ev_names = [ev.get("name") for ev in evaluators if ev.get("name")]
    header = ["model", "test_pass_rate"] + ev_names
    with open(leaderboard_dir / "leaderboard.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for sm in safe_models:
            writer.writerow([sm, "1.0"] + ["1.0" for _ in ev_names])


def _cmd_llm(opts: Dict[str, List[str]]) -> None:
    output_dir = Path(_first(opts, "-o", "--output"))
    config = _load_json(_first(opts, "-c", "--config"))
    test_cases = []
    if isinstance(config, dict):
        test_cases = [t for t in (config.get("test_cases") or []) if isinstance(t, dict)]
    evaluators = _evaluators_from_config(config)

    models = _many(opts, "-m", "--model", "--models")
    if not models:
        # Agent-connection unit-test mode: no -m. The unit-test reader walks the
        # tree for the first results.json, so a single folder name is enough.
        models = ["default"]

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_models: List[str] = []
    for model in models:
        safe = _safe_model(model) if model != "default" else "default"
        safe_models.append(safe)
        _write_llm_model_dir(output_dir / safe, test_cases, evaluators)

    # Benchmark reads a leaderboard; a unit test ignores it. Harmless either way.
    if _many(opts, "-m", "--model", "--models"):
        _write_leaderboard(output_dir, safe_models, evaluators)

    _write_config_json(output_dir, evaluators)


# --- Subcommand: stt --------------------------------------------------------
def _read_id_text_csv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(dict(row))
    except OSError:
        pass
    return rows


def _evaluator_row_cols(
    evaluators: List[Dict[str, Any]],
) -> tuple[List[str], Dict[str, str]]:
    """Column names + a per-column canned value for evaluator outputs in a
    results.csv row (``<name>`` score + ``<name>_reasoning``)."""
    cols: List[str] = []
    values: Dict[str, str] = {}
    for ev in evaluators:
        name = ev.get("name")
        if not name:
            continue
        cols.append(name)
        cols.append(f"{name}_reasoning")
        values[name] = str(_ev_scale_max(ev)) if ev.get("type") == "rating" else "1"
        values[f"{name}_reasoning"] = FAKE_REASONING
    return cols, values


# Extra scalar metrics + per-row columns the real CLI writes for the Sarvam
# judge bundle (on by default). Mirrors `_score_and_write_results` in calibrate-agent.
_SARVAM_METRICS = {"sarvam_llm_wer": FAKE_SARVAM_LLM_WER, "sarvam_llm_cer": FAKE_SARVAM_LLM_CER}
_SARVAM_ROW_COLS = ["sarvam_llm_wer", "sarvam_llm_cer", "sarvam_llm_wer_reasoning"]
_SARVAM_ROW_VALUES = {
    "sarvam_llm_wer": str(FAKE_SARVAM_LLM_WER),
    "sarvam_llm_cer": str(FAKE_SARVAM_LLM_CER),
    "sarvam_llm_wer_reasoning": "[]",
}


def _stt_metrics(evaluators: List[Dict[str, Any]], sarvam: bool) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {"wer": FAKE_WER}
    if sarvam:
        metrics.update(_SARVAM_METRICS)
    for ev in evaluators:
        name = ev.get("name")
        if name:
            metrics[name] = _metrics_aggregate(ev, 1)
    return metrics


def _cmd_stt(opts: Dict[str, List[str]]) -> None:
    output_dir = Path(_first(opts, "-o", "--output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    providers = _many(opts, "-p", "--provider", "--providers") or ["openai"]
    evaluators = _evaluators_from_config(_load_json(_first(opts, "--config")))
    sarvam = "--skip-llm-judges" not in opts

    input_dir = _first(opts, "-i", "--input")
    utterances = _read_id_text_csv(Path(input_dir) / "stt.csv") if input_dir else []
    if not utterances:
        utterances = [{"id": "audio_1", "text": "x"}]

    ev_cols, ev_values = _evaluator_row_cols(evaluators)
    extra_cols = _SARVAM_ROW_COLS if sarvam else []
    extra_values = _SARVAM_ROW_VALUES if sarvam else {}
    for provider in providers:
        sub = output_dir / f"{provider}_results"
        sub.mkdir(parents=True, exist_ok=True)
        with open(sub / "results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "gt", "pred"] + ev_cols + extra_cols)
            for utt in utterances:
                text = utt.get("text", "")
                writer.writerow(
                    [utt.get("id"), text, text]
                    + [ev_values[c] for c in ev_cols]
                    + [extra_values[c] for c in extra_cols]
                )
        with open(sub / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(_stt_metrics(evaluators, sarvam), f)

    _write_config_json(output_dir, evaluators)


# --- Subcommand: tts --------------------------------------------------------
def _tts_metrics(evaluators: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {"ttfb": FAKE_TTFB}
    for ev in evaluators:
        name = ev.get("name")
        if name:
            metrics[name] = _metrics_aggregate(ev, 1)
    return metrics


def _cmd_tts(opts: Dict[str, List[str]]) -> None:
    output_dir = Path(_first(opts, "-o", "--output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    providers = _many(opts, "-p", "--provider", "--providers") or ["openai"]
    evaluators = _evaluators_from_config(_load_json(_first(opts, "--config")))

    input_file = _first(opts, "-i", "--input")
    rows = _read_id_text_csv(Path(input_file)) if input_file else []
    if not rows:
        rows = [{"id": "row_1", "text": "hi"}]

    ev_cols, ev_values = _evaluator_row_cols(evaluators)
    for provider in providers:
        sub = output_dir / f"{provider}_results"
        audios = sub / "audios"
        audios.mkdir(parents=True, exist_ok=True)
        with open(sub / "results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "text", "audio_path"] + ev_cols)
            for idx, row in enumerate(rows):
                # Write a real (empty) .wav and reference its ABSOLUTE path — the
                # TTS worker keys audio files by exact local path and marks a
                # provider failed if no row's audio_path matches a walked file.
                wav = audios / f"row_{idx + 1}.wav"
                wav.write_bytes(b"")
                writer.writerow(
                    [row.get("id"), row.get("text", ""), str(wav)]
                    + [ev_values[c] for c in ev_cols]
                )
        with open(sub / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(_tts_metrics(evaluators), f)

    _write_config_json(output_dir, evaluators)


# --- Subcommand: simulations (normal run) -----------------------------------
def _sim_eval_row(ev: Dict[str, Any]) -> List[Any]:
    if ev.get("type") == "rating":
        return [_ev_uuid(ev), ev.get("name"), "rating", _ev_scale_max(ev), FAKE_REASONING]
    return [_ev_uuid(ev), ev.get("name"), "binary", "Pass", FAKE_REASONING]


def _write_sim_case(
    case_dir: Path,
    persona: Any,
    scenario: Any,
    evaluators: List[Dict[str, Any]],
) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    with open(case_dir / "transcript.json", "w", encoding="utf-8") as f:
        json.dump(
            [
                {"role": "user", "content": "Simulated user turn."},
                {"role": "assistant", "content": FAKE_RESPONSE},
            ],
            f,
        )
    with open(case_dir / "evaluation_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["evaluator_id", "name", "type", "value", "reasoning"])
        for ev in evaluators:
            writer.writerow(_sim_eval_row(ev))
    with open(case_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({"persona": persona, "scenario": scenario}, f)


def _cmd_simulations(opts: Dict[str, List[str]]) -> None:
    output_dir = Path(_first(opts, "-o", "--output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _load_json(_first(opts, "-c", "--config")) or {}
    personas = config.get("personas") or []
    scenarios = config.get("scenarios") or []
    evaluators = _evaluators_from_config(config)

    for i, persona in enumerate(personas, start=1):
        for j, scenario in enumerate(scenarios, start=1):
            case_dir = output_dir / f"simulation_persona_{i}_scenario_{j}"
            _write_sim_case(case_dir, persona, scenario, evaluators)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"total": len(personas) * len(scenarios)}, f)


# --- Annotation eval-only flows ---------------------------------------------
def _cmd_annotation_tts(opts: Dict[str, List[str]]) -> None:
    """`tts --eval-only` reads a prior run dir's ``results.csv`` (id, text,
    audio_path) and writes judged ``results.csv`` + ``metrics.json`` under
    ``-o`` — same flat layout as STT eval-only, but with TTS base columns."""
    output_dir = Path(_first(opts, "-o", "--output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(_first(opts, "--dataset"))
    evaluators = _evaluators_from_config(_load_json(_first(opts, "-c", "--config")))

    csv_path = run_dir / "results.csv"
    rows: List[Dict[str, Any]] = []
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    ev_cols, ev_values = _evaluator_row_cols(evaluators)
    with open(output_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text", "audio_path"] + ev_cols)
        for row in rows:
            writer.writerow(
                [row.get("id"), row.get("text"), row.get("audio_path")]
                + [ev_values[c] for c in ev_cols]
            )

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(_tts_metrics(evaluators), f)
    _write_config_json(output_dir, evaluators)


def _cmd_annotation_stt_or_general(opts: Dict[str, List[str]]) -> None:
    """`stt --eval-only` and `general` both write a flat run-root results.csv
    (id + built-in cols + per-evaluator score/reasoning) keyed by dataset id."""
    output_dir = Path(_first(opts, "-o", "--output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluators = _evaluators_from_config(_load_json(_first(opts, "-c", "--config")))
    dataset = _load_json(_first(opts, "--dataset")) or []

    ev_cols, ev_values = _evaluator_row_cols(evaluators)
    with open(output_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "gt", "pred"] + ev_cols)
        for row in dataset if isinstance(dataset, list) else []:
            if not isinstance(row, dict):
                continue
            gt = row.get("gt", row.get("input", ""))
            pred = row.get("pred", row.get("output", ""))
            writer.writerow(
                [row.get("id"), gt, pred] + [ev_values[c] for c in ev_cols]
            )

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        # Sarvam judges are an STT-eval-only feature; the annotation eval-only
        # and `general` flows never include the Sarvam bundle.
        json.dump(_stt_metrics(evaluators, sarvam=False), f)
    _write_config_json(output_dir, evaluators)


def _cmd_annotation_llm(opts: Dict[str, List[str]]) -> None:
    """`llm --eval-only` writes a flat run-root results.json keyed by
    ``test_case_id``, with judge_results per referenced evaluator."""
    output_dir = Path(_first(opts, "-o", "--output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluators = _evaluators_from_config(_load_json(_first(opts, "-c", "--config")))
    ev_by_name = {ev.get("name"): ev for ev in evaluators if ev.get("name")}
    dataset = _load_json(_first(opts, "--dataset")) or []

    results: List[Dict[str, Any]] = []
    for row in dataset if isinstance(dataset, list) else []:
        if not isinstance(row, dict):
            continue
        test_case = row.get("test_case") or {}
        criteria = (test_case.get("evaluation") or {}).get("criteria") or []
        judge_results = _llm_judge_results(criteria, ev_by_name)
        results.append(
            {
                "test_case_id": test_case.get("id"),
                "test_case": test_case,
                "output": {"response": FAKE_RESPONSE, "tool_calls": [], "cost": FAKE_COST},
                "metrics": {
                    "passed": True,
                    "reasoning": FAKE_REASONING,
                    "judge_results": judge_results,
                },
                "latency_ms": FAKE_LATENCY_MS,
            }
        )

    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"total": len(results), "passed": len(results)}, f)
    _write_config_json(output_dir, evaluators)


def _cmd_annotation_simulation(opts: Dict[str, List[str]]) -> None:
    """`simulations --eval-only` writes one ``row_<i>/`` dir per dataset row plus
    a ``dataset_map.json`` mapping ``row_<i>`` → {index, name}."""
    output_dir = Path(_first(opts, "-o", "--output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluators = _evaluators_from_config(_load_json(_first(opts, "-c", "--config")))
    dataset = _load_json(_first(opts, "--dataset")) or []

    dataset_map: Dict[str, Any] = {}
    for idx, row in enumerate(dataset if isinstance(dataset, list) else []):
        row_id = f"row_{idx + 1}"
        name = row.get("name") if isinstance(row, dict) else None
        dataset_map[row_id] = {"index": idx, "name": name}
        case_dir = output_dir / row_id
        case_dir.mkdir(parents=True, exist_ok=True)
        with open(case_dir / "evaluation_results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["evaluator_id", "name", "type", "value", "reasoning"])
            for ev in evaluators:
                writer.writerow(_sim_eval_row(ev))

    with open(output_dir / "dataset_map.json", "w", encoding="utf-8") as f:
        json.dump(dataset_map, f)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"total": len(dataset_map)}, f)
    _write_config_json(output_dir, evaluators)


# --- status (provider health) -----------------------------------------------
def _cmd_status() -> None:
    """Placeholder healthy payload. `provider_status.run_check` short-circuits
    `status` under the flag (returning `_fake_healthy_providers()`), so the fake
    is never spawned for it; this stays only to keep `status` a handled subcommand."""
    sys.stdout.write(json.dumps({"openai": {"status": "pass"}}))


# Every `calibrate-agent <subcommand>` the backend workers spawn. The guard test
# `tests/test_fake_matches_real_usage.py` statically scans the real call sites and
# fails if any spawns a subcommand missing from this set — so a new real usage
# can't ship without the fake growing a handler for it.
SUPPORTED_SUBCOMMANDS = frozenset(
    {"status", "general", "llm", "stt", "tts", "simulations"}
)


def main(argv: List[str]) -> int:
    sub, opts = _parse_args(argv)
    eval_only = "--eval-only" in opts

    if sub not in SUPPORTED_SUBCOMMANDS:
        sys.stderr.write(f"fake_calibrate_agent: unknown subcommand {sub!r}\n")
        return 1

    if sub == "status":
        _cmd_status()
    elif sub == "general":
        _cmd_annotation_stt_or_general(opts)
    elif sub == "llm":
        _cmd_annotation_llm(opts) if eval_only else _cmd_llm(opts)
    elif sub == "stt":
        _cmd_annotation_stt_or_general(opts) if eval_only else _cmd_stt(opts)
    elif sub == "tts":
        _cmd_annotation_tts(opts) if eval_only else _cmd_tts(opts)
    elif sub == "simulations":
        _cmd_annotation_simulation(opts) if eval_only else _cmd_simulations(opts)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
