from fastapi import APIRouter, Depends, Query

from db import (
    get_annotations_for_user,
    get_evaluator,
    get_evaluator_runs_for_user,
)
from auth_utils import get_current_user_id
from annotation_metrics import (
    aggregate_agreement,
    aggregate_human_evaluator_agreement,
    trend_series,
    trend_series_human_evaluator,
)


router = APIRouter(prefix="/annotation-agreement", tags=["annotation-agreement"])


@router.get("/trend")
async def agreement_trend(
    bucket: str = Query("week", pattern="^(week|month|year)$"),
    days: int = Query(90, ge=1, le=3650),
    user_id: str = Depends(get_current_user_id),
):
    """Account-wide human-vs-human agreement trend across all of the user's
    annotation tasks, plus per-evaluator human-vs-evaluator alignment for
    every evaluator that has produced at least one run on the user's data.

    Returns:
      - `human_human`: `{ current, pair_count, series }`.
      - `evaluators`: list of `{ evaluator_id, name, current, pair_count, series }`,
        one per evaluator that's been run at least once on this account's data.
    """
    annotations = get_annotations_for_user(user_id)
    runs = get_evaluator_runs_for_user(user_id)

    hh_current, hh_pairs = aggregate_agreement(annotations)
    hh_series = trend_series(annotations, bucket=bucket, days=days)

    # Distinct evaluator_ids that have produced runs on the user's data — that's
    # the natural set to include in the account-wide rollup. Names are resolved
    # live from the evaluators table so renames show up.
    evaluator_ids = []
    seen = set()
    for r in runs:
        ev_id = r.get("evaluator_id")
        if ev_id and ev_id not in seen:
            seen.add(ev_id)
            evaluator_ids.append(ev_id)
    series_by_id = trend_series_human_evaluator(
        annotations, runs, evaluator_ids, bucket=bucket, days=days
    )
    evaluators_block = []
    for ev_id in evaluator_ids:
        ev = get_evaluator(ev_id)
        cur, pairs = aggregate_human_evaluator_agreement(annotations, runs, ev_id)
        evaluators_block.append(
            {
                "evaluator_id": ev_id,
                "name": ev.get("name") if ev else None,
                "current": cur,
                "pair_count": pairs,
                "series": series_by_id.get(ev_id, []),
            }
        )

    return {
        "bucket": bucket,
        "days": days,
        "human_human": {
            "current": hh_current,
            "pair_count": hh_pairs,
            "series": hh_series,
        },
        "evaluators": evaluators_block,
    }
