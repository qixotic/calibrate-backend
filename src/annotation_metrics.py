"""Human-vs-human and human-vs-evaluator agreement metrics for annotation tasks.

Pairwise mean agreement on shared rows. Each (item, evaluator_id) is treated as
its own annotation slot; a row-level overall annotation has evaluator_id=None.

For binary judgements: agreement = 1.0 if equal else 0.0.
For numeric (rating) judgements: agreement = 1 - |a - b| / span, where span is
the observed range across all paired numeric judgements (defaults to 1.0 if all
ratings are equal). This keeps the metric in [0, 1] without needing the
evaluator's scale config.

For mixed types within a slot, only same-type pairs contribute.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple


# All agreement floats returned to callers are rounded to this many decimal
# digits so the API surface is stable and free of float-formatting noise.
_AGREEMENT_DECIMALS = 4


def _round_agreement(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), _AGREEMENT_DECIMALS)


def _scalar(value: Any) -> Any:
    """Extract the comparable payload from an annotation `value` JSON.

    Conventions accepted: a bare bool/number/string, or a dict with one of
    {value, score, rating, label, binary}. Comments and other keys are ignored.
    """
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        for key in ("value", "score", "rating", "label", "binary"):
            if key in value and value[key] is not None:
                return value[key]
    return None


def _classify(scalar: Any) -> Optional[str]:
    if scalar is None:
        return None
    if isinstance(scalar, bool):
        return "binary"
    if isinstance(scalar, (int, float)):
        return "numeric"
    if isinstance(scalar, str):
        return "categorical"
    return None


def _pairwise_agreement(
    values: List[Any],
) -> Tuple[float, int]:
    """Mean pairwise agreement and pair count over a single annotation slot.

    Returns (mean_agreement, pair_count). pair_count == 0 ⇒ no comparable pairs.
    """
    classified = [(_classify(v), v) for v in values if v is not None]
    if len(classified) < 2:
        return (0.0, 0)

    # Find numeric span (across this slot) for normalisation.
    numerics = [v for k, v in classified if k == "numeric"]
    if numerics:
        span = max(numerics) - min(numerics)
        if span <= 0:
            span = 1.0
    else:
        span = 1.0

    total = 0.0
    pairs = 0
    for i in range(len(classified)):
        for j in range(i + 1, len(classified)):
            ki, vi = classified[i]
            kj, vj = classified[j]
            if ki != kj:
                continue
            if ki == "binary" or ki == "categorical":
                total += 1.0 if vi == vj else 0.0
            elif ki == "numeric":
                total += 1.0 - abs(vi - vj) / span
            pairs += 1
    if pairs == 0:
        return (0.0, 0)
    return (total / pairs, pairs)


def _slot_key(annotation: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    return (annotation["item_id"], annotation.get("evaluator_id"))


def filter_runs_to_live_versions(
    runs: Iterable[Dict[str, Any]],
    live_version_by_evaluator: Dict[str, Optional[str]],
) -> List[Dict[str, Any]]:
    """Drop evaluator_runs whose `evaluator_version_id` doesn't match the
    evaluator's current live version.

    Used so that experimental/non-live runs don't silently contaminate the
    "evaluator agreement" surfaced anywhere in the API. When a user says
    "evaluator agreement" without qualifying a version, they mean the live
    version's agreement.

    Runs for evaluators with no live version (or evaluator_id missing in the
    map) are dropped — without a live version there's no contribution to
    "live agreement" by definition.
    """
    out: List[Dict[str, Any]] = []
    for r in runs:
        ev_id = r.get("evaluator_id")
        if not ev_id:
            continue
        live_v = live_version_by_evaluator.get(ev_id)
        if not live_v:
            continue
        if r.get("evaluator_version_id") == live_v:
            out.append(r)
    return out


def aggregate_agreement(
    annotations: Iterable[Dict[str, Any]],
) -> Tuple[Optional[float], int]:
    """Mean pairwise agreement across every (item, evaluator) slot that has ≥2
    distinct annotators in the supplied annotation list.

    Returns (mean_agreement | None if no comparable pairs, total_pair_count).
    """
    # Group by (item_id, evaluator_id), dedup per annotator (latest wins — but
    # input is already the upserted row, so one annotator contributes once).
    by_slot: Dict[Tuple[str, Optional[str]], Dict[str, Any]] = {}
    for a in annotations:
        slot = _slot_key(a)
        annotator = a.get("annotator_id")
        if annotator is None:
            continue
        scalar = _scalar(a.get("value"))
        if scalar is None:
            continue
        # Pick the latest per (slot, annotator) — annotations are sorted by
        # updated_at ascending, so we just overwrite.
        by_slot.setdefault(slot, {})[annotator] = scalar

    total_weighted = 0.0
    total_pairs = 0
    for slot, per_annotator in by_slot.items():
        if len(per_annotator) < 2:
            continue
        mean, pairs = _pairwise_agreement(list(per_annotator.values()))
        if pairs == 0:
            continue
        total_weighted += mean * pairs
        total_pairs += pairs
    if total_pairs == 0:
        return (None, 0)
    return (_round_agreement(total_weighted / total_pairs), total_pairs)


# ============ Bucketing for trend series ============


_BUCKET_DELTAS = {
    "week": timedelta(days=7),
    "month": timedelta(days=30),       # calendar-month-ish; keeps math simple
    "year": timedelta(days=365),
}


def _parse_ts(ts: str) -> datetime:
    """Parse a SQLite CURRENT_TIMESTAMP string ('YYYY-MM-DD HH:MM:SS', UTC)."""
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def _fmt_ts(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def build_buckets(
    bucket: str, days: int, now: Optional[datetime] = None
) -> List[Tuple[datetime, datetime]]:
    """Build a list of (start, end) UTC bucket boundaries spanning the last
    `days` ending now, in ascending order. Each bucket is [start, end)."""
    if bucket not in _BUCKET_DELTAS:
        raise ValueError(
            f"bucket must be one of {list(_BUCKET_DELTAS)}, got {bucket!r}"
        )
    if days <= 0:
        raise ValueError("days must be > 0")
    delta = _BUCKET_DELTAS[bucket]
    now = (now or datetime.now(timezone.utc)).replace(microsecond=0)
    window_start = now - timedelta(days=days)
    buckets: List[Tuple[datetime, datetime]] = []
    cur_end = now
    while cur_end > window_start:
        cur_start = cur_end - delta
        if cur_start < window_start:
            cur_start = window_start
        buckets.append((cur_start, cur_end))
        cur_end = cur_start
    return list(reversed(buckets))


def _latest_evaluator_value_per_slot(
    evaluator_runs: Iterable[Dict[str, Any]],
    evaluator_id: Optional[str] = None,
) -> Dict[Tuple[str, str], Tuple[Any, str]]:
    """Pick the most recent evaluator_run per (item_id, evaluator_id) pair.
    Returns `{(item_id, evaluator_id): (scalar_value, completed_at_or_created_at)}`.
    Filters to a single evaluator_id when supplied."""
    out: Dict[Tuple[str, str], Tuple[Any, str]] = {}
    for r in evaluator_runs:
        ev_id = r.get("evaluator_id")
        if not ev_id:
            continue
        if evaluator_id is not None and ev_id != evaluator_id:
            continue
        item_id = r.get("item_id")
        if not item_id:
            continue
        scalar = _scalar(r.get("value"))
        if scalar is None:
            continue
        ts = r.get("completed_at") or r.get("created_at") or ""
        slot = (item_id, ev_id)
        existing = out.get(slot)
        if existing is None or ts > existing[1]:
            out[slot] = (scalar, ts)
    return out


def evaluator_human_pair_agreement(
    eval_value: Any,
    human_values: Iterable[Any],
) -> Tuple[float, int]:
    """One-slot evaluator-vs-human agreement: pair the evaluator's value with
    each human's value on the same slot and return (sum_of_scores, pair_count).

    Same scoring rules as `_pairwise_agreement`:
      - binary / categorical: 1.0 if equal, 0.0 otherwise
      - numeric: 1 - |a - b| / span, where span spans this slot's values

    Pairs of mismatched type are skipped (don't contribute to the count).

    Returns the SUM (not mean) so callers aggregating across slots can weight
    correctly: account-level mean = sum_of_sums / sum_of_counts. Single-slot
    callers that want the mean compute `sum / count` themselves (and treat
    count==0 as `None`).
    """
    eval_kind = _classify(eval_value)
    if eval_kind is None:
        return (0.0, 0)
    humans = list(human_values)
    if not humans:
        return (0.0, 0)
    numerics = [v for v in (eval_value, *humans) if _classify(v) == "numeric"]
    span = (max(numerics) - min(numerics)) if numerics else 1.0
    if span <= 0:
        span = 1.0
    total = 0.0
    pairs = 0
    for hv in humans:
        if _classify(hv) != eval_kind:
            continue
        if eval_kind in ("binary", "categorical"):
            total += 1.0 if eval_value == hv else 0.0
        elif eval_kind == "numeric":
            total += 1.0 - abs(eval_value - hv) / span
        pairs += 1
    return (total, pairs)


def aggregate_human_evaluator_agreement(
    annotations: Iterable[Dict[str, Any]],
    evaluator_runs: Iterable[Dict[str, Any]],
    evaluator_id: str,
) -> Tuple[Optional[float], int]:
    """Mean pairwise agreement between one evaluator's machine judgements and
    every human annotation on the matching slot.

    Slot = (item_id, evaluator_id). For each slot where the evaluator ran AND
    at least one human annotated, pair the evaluator value with each human's
    latest value and score:
      - bool / categorical: 1.0 if equal else 0.0
      - numeric: 1 - |a - b| / span (span derived per slot from the values
        participating in pairs)
    Mean weighted by pair count. (None, 0) if no comparable pairs."""
    eval_by_slot = _latest_evaluator_value_per_slot(evaluator_runs, evaluator_id)
    if not eval_by_slot:
        return (None, 0)

    # Group human annotations by (item, evaluator) slot, keeping latest per
    # annotator. Annotations are sorted ascending by updated_at so overwriting
    # gives us "latest wins" without an explicit timestamp comparison.
    human_by_slot: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for a in annotations:
        ev_id = a.get("evaluator_id")
        if ev_id != evaluator_id:
            continue
        slot = (a["item_id"], ev_id)
        annotator = a.get("annotator_id")
        if annotator is None:
            continue
        scalar = _scalar(a.get("value"))
        if scalar is None:
            continue
        human_by_slot.setdefault(slot, {})[annotator] = scalar

    total_weighted = 0.0
    total_pairs = 0
    for slot, (eval_val, _ts) in eval_by_slot.items():
        humans = human_by_slot.get(slot)
        if not humans:
            continue
        slot_total, slot_pairs = evaluator_human_pair_agreement(
            eval_val, humans.values()
        )
        total_weighted += slot_total
        total_pairs += slot_pairs
    if total_pairs == 0:
        return (None, 0)
    return (_round_agreement(total_weighted / total_pairs), total_pairs)


def has_any_comparable_pair(
    annotations: Iterable[Dict[str, Any]],
    evaluator_runs: Iterable[Dict[str, Any]],
    evaluator_ids: Iterable[str],
) -> bool:
    """True if any agreement pair exists: a shared slot with ≥2 human
    annotators (human-vs-human), or a slot where a human and one of the given
    evaluators both judged (human-vs-evaluator). False otherwise."""
    annotations_list = list(annotations)
    runs_list = list(evaluator_runs)
    if aggregate_agreement(annotations_list)[1] > 0:
        return True
    for ev_id in evaluator_ids:
        if aggregate_human_evaluator_agreement(
            annotations_list, runs_list, ev_id
        )[1] > 0:
            return True
    return False


def per_item_agreement(
    annotations_for_item: Iterable[Dict[str, Any]],
    evaluator_runs_for_item: Iterable[Dict[str, Any]],
    evaluator_ids: Iterable[str],
) -> Dict[str, Any]:
    """Compute, for one item: human-vs-human agreement across all evaluators
    annotated, plus per-evaluator human-vs-evaluator agreement. Returns:

        {
          "human_human": {"agreement": float|None, "pair_count": int},
          "evaluators": [
            {"evaluator_id": str, "agreement": float|None, "pair_count": int}
          ]
        }
    """
    annotations_list = list(annotations_for_item)
    runs_list = list(evaluator_runs_for_item)
    hh, hh_pairs = aggregate_agreement(annotations_list)
    out_evaluators: List[Dict[str, Any]] = []
    for ev_id in evaluator_ids:
        agree, pairs = aggregate_human_evaluator_agreement(
            annotations_list, runs_list, ev_id
        )
        out_evaluators.append(
            {
                "evaluator_id": ev_id,
                "agreement": agree,
                "pair_count": pairs,
            }
        )
    return {
        "human_human": {"agreement": hh, "pair_count": hh_pairs},
        "evaluators": out_evaluators,
    }


def trend_series_human_evaluator(
    annotations: List[Dict[str, Any]],
    evaluator_runs: List[Dict[str, Any]],
    evaluator_ids: List[str],
    bucket: str,
    days: int,
    now: Optional[datetime] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Per-evaluator cumulative agreement trend over time, bucketed.

    For each bucket end T and each evaluator E, compute alignment using:
      - every annotation with `updated_at <= T`
      - every evaluator_run with `completed_at <= T` (falls back to created_at)

    Returns `{evaluator_id: [{bucket_start, bucket_end, agreement, pair_count}, ...]}`.
    """
    buckets = build_buckets(bucket, days, now=now)
    if not evaluator_ids:
        return {}
    if not annotations and not evaluator_runs:
        empty = [
            {
                "bucket_start": _fmt_ts(s),
                "bucket_end": _fmt_ts(e),
                "agreement": None,
                "pair_count": 0,
            }
            for (s, e) in buckets
        ]
        return {ev_id: list(empty) for ev_id in evaluator_ids}

    parsed_ann: List[Tuple[datetime, Dict[str, Any]]] = []
    for a in annotations:
        ts = a.get("updated_at") or a.get("created_at")
        if not ts:
            continue
        try:
            parsed_ann.append((_parse_ts(ts), a))
        except ValueError:
            continue
    parsed_ann.sort(key=lambda x: x[0])

    parsed_runs: List[Tuple[datetime, Dict[str, Any]]] = []
    for r in evaluator_runs:
        ts = r.get("completed_at") or r.get("created_at")
        if not ts:
            continue
        try:
            parsed_runs.append((_parse_ts(ts), r))
        except ValueError:
            continue
    parsed_runs.sort(key=lambda x: x[0])

    out: Dict[str, List[Dict[str, Any]]] = {ev_id: [] for ev_id in evaluator_ids}
    ann_idx = 0
    run_idx = 0
    visible_ann: List[Dict[str, Any]] = []
    visible_runs: List[Dict[str, Any]] = []
    for (start, end) in buckets:
        while ann_idx < len(parsed_ann) and parsed_ann[ann_idx][0] <= end:
            visible_ann.append(parsed_ann[ann_idx][1])
            ann_idx += 1
        while run_idx < len(parsed_runs) and parsed_runs[run_idx][0] <= end:
            visible_runs.append(parsed_runs[run_idx][1])
            run_idx += 1
        for ev_id in evaluator_ids:
            agreement, pair_count = aggregate_human_evaluator_agreement(
                visible_ann, visible_runs, ev_id
            )
            out[ev_id].append(
                {
                    "bucket_start": _fmt_ts(start),
                    "bucket_end": _fmt_ts(end),
                    "agreement": agreement,
                    "pair_count": pair_count,
                }
            )
    return out


def trend_series_evaluator_breakdown(
    annotations: List[Dict[str, Any]],
    runs: List[Dict[str, Any]],
    evaluator_id: str,
    version_ids: List[str],
    task_ids: List[str],
    bucket: str,
    days: int,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Single-pass cumulative agreement trend for one evaluator, broken down by
    version and task simultaneously.

    `runs` must carry a `task_id` field (as returned by
    `get_evaluator_runs_for_evaluator_org_scoped`).

    Returns:
        {
            "overall":    [{bucket_start, bucket_end, agreement, pair_count}, ...],
            "by_version": {version_id: [series ...]},
            "by_task":    {task_id:    [series ...]},
        }
    """
    buckets = build_buckets(bucket, days, now=now)

    def _empty() -> List[Dict[str, Any]]:
        return [
            {
                "bucket_start": _fmt_ts(s),
                "bucket_end": _fmt_ts(e),
                "agreement": None,
                "pair_count": 0,
            }
            for (s, e) in buckets
        ]

    if not annotations and not runs:
        return {
            "overall": _empty(),
            "by_version": {vid: _empty() for vid in version_ids},
            "by_task": {tid: _empty() for tid in task_ids},
        }

    parsed_ann: List[Tuple[datetime, Dict[str, Any]]] = []
    for a in annotations:
        ts = a.get("updated_at") or a.get("created_at")
        if not ts:
            continue
        try:
            parsed_ann.append((_parse_ts(ts), a))
        except ValueError:
            continue
    parsed_ann.sort(key=lambda x: x[0])

    parsed_runs: List[Tuple[datetime, Dict[str, Any]]] = []
    for r in runs:
        ts = r.get("completed_at") or r.get("created_at")
        if not ts:
            continue
        try:
            parsed_runs.append((_parse_ts(ts), r))
        except ValueError:
            continue
    parsed_runs.sort(key=lambda x: x[0])

    version_id_set = set(version_ids)
    task_id_set = set(task_ids)

    # Pre-separated visible accumulators — populated as we advance through
    # buckets so each aggregate call works on an already-filtered slice.
    visible_ann_all: List[Dict[str, Any]] = []
    visible_ann_by_task: Dict[str, List[Dict[str, Any]]] = {tid: [] for tid in task_ids}
    visible_runs_all: List[Dict[str, Any]] = []
    visible_runs_by_version: Dict[str, List[Dict[str, Any]]] = {vid: [] for vid in version_ids}
    visible_runs_by_task: Dict[str, List[Dict[str, Any]]] = {tid: [] for tid in task_ids}

    out_overall: List[Dict[str, Any]] = []
    out_by_version: Dict[str, List[Dict[str, Any]]] = {vid: [] for vid in version_ids}
    out_by_task: Dict[str, List[Dict[str, Any]]] = {tid: [] for tid in task_ids}

    ann_idx = 0
    run_idx = 0

    for start, end in buckets:
        while ann_idx < len(parsed_ann) and parsed_ann[ann_idx][0] <= end:
            a = parsed_ann[ann_idx][1]
            visible_ann_all.append(a)
            tid = a.get("task_id")
            if tid in task_id_set:
                visible_ann_by_task[tid].append(a)
            ann_idx += 1

        while run_idx < len(parsed_runs) and parsed_runs[run_idx][0] <= end:
            r = parsed_runs[run_idx][1]
            visible_runs_all.append(r)
            vid = r.get("evaluator_version_id")
            if vid in version_id_set:
                visible_runs_by_version[vid].append(r)
            tid = r.get("task_id")
            if tid in task_id_set:
                visible_runs_by_task[tid].append(r)
            run_idx += 1

        meta = {"bucket_start": _fmt_ts(start), "bucket_end": _fmt_ts(end)}

        agree, pairs = aggregate_human_evaluator_agreement(
            visible_ann_all, visible_runs_all, evaluator_id
        )
        out_overall.append({**meta, "agreement": agree, "pair_count": pairs})

        for vid in version_ids:
            agree, pairs = aggregate_human_evaluator_agreement(
                visible_ann_all, visible_runs_by_version[vid], evaluator_id
            )
            out_by_version[vid].append({**meta, "agreement": agree, "pair_count": pairs})

        for tid in task_ids:
            agree, pairs = aggregate_human_evaluator_agreement(
                visible_ann_by_task[tid], visible_runs_by_task[tid], evaluator_id
            )
            out_by_task[tid].append({**meta, "agreement": agree, "pair_count": pairs})

    return {
        "overall": out_overall,
        "by_version": out_by_version,
        "by_task": out_by_task,
    }


def aggregate_agreement_for_annotator(
    annotations: Iterable[Dict[str, Any]],
    annotator_id: str,
) -> Tuple[Optional[float], int]:
    """Mean pairwise agreement between `annotator_id` and every other annotator
    on slots where both participated.

    Slots = (item_id, evaluator_id). Only slots that include `annotator_id`
    AND at least one other annotator contribute. Each pair is `annotator_id`
    vs one other annotator.
    """
    by_slot: Dict[Tuple[str, Optional[str]], Dict[str, Any]] = {}
    for a in annotations:
        slot = _slot_key(a)
        annotator = a.get("annotator_id")
        if annotator is None:
            continue
        scalar = _scalar(a.get("value"))
        if scalar is None:
            continue
        by_slot.setdefault(slot, {})[annotator] = scalar

    total_weighted = 0.0
    total_pairs = 0
    for slot, per_annotator in by_slot.items():
        if annotator_id not in per_annotator:
            continue
        if len(per_annotator) < 2:
            continue
        my_val = per_annotator[annotator_id]
        my_kind = _classify(my_val)
        if my_kind is None:
            continue
        # Numeric span across this slot, for normalisation.
        numerics = [
            v
            for v in per_annotator.values()
            if _classify(v) == "numeric"
        ]
        span = max(numerics) - min(numerics) if numerics else 1.0
        if span <= 0:
            span = 1.0
        for other_id, other_val in per_annotator.items():
            if other_id == annotator_id:
                continue
            other_kind = _classify(other_val)
            if other_kind != my_kind:
                continue
            if my_kind in ("binary", "categorical"):
                total_weighted += 1.0 if my_val == other_val else 0.0
            elif my_kind == "numeric":
                total_weighted += 1.0 - abs(my_val - other_val) / span
            total_pairs += 1
    if total_pairs == 0:
        return (None, 0)
    return (_round_agreement(total_weighted / total_pairs), total_pairs)


def trend_series(
    annotations: List[Dict[str, Any]],
    bucket: str,
    days: int,
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Group annotations into buckets by `updated_at`, compute per-bucket
    pairwise agreement using ALL annotations whose updated_at <= bucket end.

    The "as-of" semantics make the trend monotone in coverage: once a row has
    enough annotators, every subsequent bucket still benefits from those
    judgements. This matches the intuition "average alignment so far."
    """
    buckets = build_buckets(bucket, days, now=now)
    if not annotations:
        return [
            {
                "bucket_start": _fmt_ts(s),
                "bucket_end": _fmt_ts(e),
                "agreement": None,
                "pair_count": 0,
            }
            for (s, e) in buckets
        ]
    # Pre-parse timestamps once.
    parsed: List[Tuple[datetime, Dict[str, Any]]] = []
    for a in annotations:
        ts = a.get("updated_at") or a.get("created_at")
        if not ts:
            continue
        try:
            parsed.append((_parse_ts(ts), a))
        except ValueError:
            continue
    parsed.sort(key=lambda x: x[0])

    out: List[Dict[str, Any]] = []
    idx = 0
    visible: List[Dict[str, Any]] = []
    for (start, end) in buckets:
        while idx < len(parsed) and parsed[idx][0] <= end:
            visible.append(parsed[idx][1])
            idx += 1
        agreement, pair_count = aggregate_agreement(visible)
        out.append(
            {
                "bucket_start": _fmt_ts(start),
                "bucket_end": _fmt_ts(end),
                "agreement": agreement,
                "pair_count": pair_count,
            }
        )
    return out


def trend_series_for_annotator(
    annotations: List[Dict[str, Any]],
    annotator_id: str,
    bucket: str,
    days: int,
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Cumulative trend of `annotator_id`'s pairwise agreement with other
    annotators, bucketed by updated_at."""
    buckets = build_buckets(bucket, days, now=now)
    if not annotations:
        return [
            {
                "bucket_start": _fmt_ts(s),
                "bucket_end": _fmt_ts(e),
                "agreement": None,
                "pair_count": 0,
            }
            for (s, e) in buckets
        ]
    parsed: List[Tuple[datetime, Dict[str, Any]]] = []
    for a in annotations:
        ts = a.get("updated_at") or a.get("created_at")
        if not ts:
            continue
        try:
            parsed.append((_parse_ts(ts), a))
        except ValueError:
            continue
    parsed.sort(key=lambda x: x[0])

    out: List[Dict[str, Any]] = []
    idx = 0
    visible: List[Dict[str, Any]] = []
    for (start, end) in buckets:
        while idx < len(parsed) and parsed[idx][0] <= end:
            visible.append(parsed[idx][1])
            idx += 1
        agreement, pair_count = aggregate_agreement_for_annotator(
            visible, annotator_id
        )
        out.append(
            {
                "bucket_start": _fmt_ts(start),
                "bucket_end": _fmt_ts(end),
                "agreement": agreement,
                "pair_count": pair_count,
            }
        )
    return out
