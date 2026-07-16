"""Tests for src/annotation_metrics.py — pure pairwise agreement math."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from annotation_metrics import (
    _classify,
    _pairwise_agreement,
    _scalar,
    aggregate_agreement,
    aggregate_agreement_for_annotator,
    aggregate_human_evaluator_agreement,
    build_buckets,
    evaluator_human_pair_agreement,
    filter_runs_to_live_versions,
    has_any_comparable_pair,
    per_item_agreement,
    trend_series,
    trend_series_evaluator_breakdown,
    trend_series_for_annotator,
    trend_series_human_evaluator,
)


def test_scalar_extracts_value_variants():
    assert _scalar(None) is None
    assert _scalar(1) == 1
    assert _scalar(True) is True
    assert _scalar(3.14) == 3.14
    assert _scalar("foo") == "foo"
    assert _scalar({"value": 5, "reasoning": "x"}) == 5
    assert _scalar({"score": 4}) == 4
    assert _scalar({"label": "good"}) == "good"
    assert _scalar({"binary": True}) is True
    assert _scalar({"other": "x"}) is None
    assert _scalar([1, 2]) is None


def test_classify_kinds():
    assert _classify(None) is None
    assert _classify(True) == "binary"
    assert _classify(1) == "numeric"
    assert _classify(3.0) == "numeric"
    assert _classify("x") == "categorical"
    assert _classify([1, 2]) is None


def test_pairwise_agreement_branches():
    # < 2 → 0
    assert _pairwise_agreement([]) == (0.0, 0)
    assert _pairwise_agreement([1]) == (0.0, 0)
    # binary match
    mean, pairs = _pairwise_agreement([True, True])
    assert mean == 1.0 and pairs == 1
    # binary mismatch
    mean, pairs = _pairwise_agreement([True, False])
    assert mean == 0.0 and pairs == 1
    # numeric with span = max - min
    mean, pairs = _pairwise_agreement([1, 5])
    assert pairs == 1 and mean == 0.0
    # numeric all equal -> span 1, agreement = 1
    mean, pairs = _pairwise_agreement([3, 3, 3])
    assert pairs == 3 and mean == 1.0
    # categorical
    mean, pairs = _pairwise_agreement(["a", "a", "b"])
    assert pairs == 3 and 0 < mean < 1
    # mixed types skip mismatched pairs
    mean, pairs = _pairwise_agreement([True, 1])
    # bool/int don't match in _classify (bool is binary, int is numeric)
    assert pairs == 0
    # all None → 0
    assert _pairwise_agreement([None, None]) == (0.0, 0)


def test_filter_runs_to_live_versions():
    runs = [
        {"evaluator_id": "e1", "evaluator_version_id": "v-live"},
        {"evaluator_id": "e1", "evaluator_version_id": "v-stale"},
        {"evaluator_id": "e2", "evaluator_version_id": "vx"},  # not in map
        {"evaluator_id": None, "evaluator_version_id": "v"},  # no eval id
    ]
    out = filter_runs_to_live_versions(runs, {"e1": "v-live", "e2": None})
    assert len(out) == 1
    assert out[0]["evaluator_id"] == "e1"


def test_aggregate_agreement_empty_and_full():
    assert aggregate_agreement([]) == (None, 0)
    # Two annotators agreeing on one slot
    annotations = [
        {"item_id": "i1", "evaluator_id": "e1", "annotator_id": "a", "value": {"value": True}},
        {"item_id": "i1", "evaluator_id": "e1", "annotator_id": "b", "value": {"value": True}},
    ]
    agree, pairs = aggregate_agreement(annotations)
    assert agree == 1.0 and pairs == 1
    # Skip rows missing scalar / annotator
    annotations.append({"item_id": "i2", "evaluator_id": "e1", "annotator_id": None, "value": {"value": True}})
    annotations.append({"item_id": "i3", "evaluator_id": "e1", "annotator_id": "a", "value": {"other": 1}})
    agree, pairs = aggregate_agreement(annotations)
    assert pairs == 1


def test_build_buckets_validation():
    with pytest.raises(ValueError):
        build_buckets("decade", 7)
    with pytest.raises(ValueError):
        build_buckets("week", 0)
    now = datetime(2024, 1, 31, tzinfo=timezone.utc)
    buckets = build_buckets("week", 14, now=now)
    assert len(buckets) == 2
    # Buckets are returned chronologically
    assert buckets[0][1] <= buckets[1][1]


def test_evaluator_human_pair_agreement():
    # eval=binary, humans=binary
    total, pairs = evaluator_human_pair_agreement(True, [True, False])
    assert pairs == 2 and total == 1.0
    # eval=None classification
    assert evaluator_human_pair_agreement(None, [True]) == (0.0, 0)
    # eval present but no humans
    assert evaluator_human_pair_agreement(True, []) == (0.0, 0)
    # eval = numeric; mismatched-type humans are skipped
    total, pairs = evaluator_human_pair_agreement(3, [3, "foo"])
    assert pairs == 1


def test_aggregate_human_evaluator_agreement_empty():
    agree, pairs = aggregate_human_evaluator_agreement([], [], "e1")
    assert agree is None and pairs == 0

    annotations = [
        {"item_id": "i1", "evaluator_id": "e1", "annotator_id": "a", "value": {"value": True}},
    ]
    # Latest-per-slot picks the most recent evaluator run for the slot
    runs = [
        {"item_id": "i1", "evaluator_id": "e1", "evaluator_version_id": "v1", "value": {"value": True}, "created_at": "2024-01-01 00:00:00"},
        {"item_id": "i1", "evaluator_id": "e1", "evaluator_version_id": "v1", "value": {"value": False}, "completed_at": "2024-02-01 00:00:00"},
        # Different evaluator — filtered out
        {"item_id": "i1", "evaluator_id": "e2", "evaluator_version_id": "v", "value": {"value": True}},
        # Missing fields — filtered
        {"evaluator_id": "e1", "value": {"value": True}},
    ]
    agree, pairs = aggregate_human_evaluator_agreement(annotations, runs, "e1")
    # Latest evaluator value is False; human said True → disagreement
    assert agree == 0.0 and pairs == 1


def test_has_any_comparable_pair_human_human():
    annotations = [
        {"item_id": "i1", "evaluator_id": "e1", "annotator_id": "a", "value": {"value": True}},
        {"item_id": "i1", "evaluator_id": "e1", "annotator_id": "b", "value": {"value": False}},
    ]
    assert has_any_comparable_pair(annotations, [], ["e1"]) is True


def test_has_any_comparable_pair_human_evaluator():
    annotations = [
        {"item_id": "i1", "evaluator_id": "e1", "annotator_id": "a", "value": {"value": True}},
    ]
    runs = [
        {"item_id": "i1", "evaluator_id": "e1", "value": {"value": True}, "created_at": "2024-01-01 00:00:00"},
    ]
    assert has_any_comparable_pair(annotations, runs, ["e1"]) is True


def test_has_any_comparable_pair_none():
    # Single annotation, no evaluator run → no pair
    annotations = [
        {"item_id": "i1", "evaluator_id": "e1", "annotator_id": "a", "value": {"value": True}},
    ]
    assert has_any_comparable_pair(annotations, [], ["e1"]) is False
    # Empty inputs
    assert has_any_comparable_pair([], [], []) is False
    assert has_any_comparable_pair([], [], ["e1"]) is False


def test_has_any_comparable_pair_accepts_iterables():
    # Generators are consumed multiple times internally — must be materialized.
    annotations = [
        {"item_id": "i1", "evaluator_id": "e1", "annotator_id": "a", "value": {"value": True}},
    ]
    runs = [
        {"item_id": "i1", "evaluator_id": "e1", "value": {"value": True}, "created_at": "2024-01-01 00:00:00"},
    ]
    ann_gen = (a for a in annotations)
    run_gen = (r for r in runs)
    ev_gen = (e for e in ["e1"])
    assert has_any_comparable_pair(ann_gen, run_gen, ev_gen) is True


def test_per_item_agreement_shape():
    ann = [
        {"item_id": "i1", "evaluator_id": "e1", "annotator_id": "a", "value": {"value": True}},
        {"item_id": "i1", "evaluator_id": "e1", "annotator_id": "b", "value": {"value": False}},
    ]
    runs = [
        {"item_id": "i1", "evaluator_id": "e1", "value": {"value": True}, "created_at": "2024-01-01 00:00:00"},
    ]
    out = per_item_agreement(ann, runs, ["e1"])
    assert "human_human" in out
    assert out["evaluators"][0]["evaluator_id"] == "e1"


def test_aggregate_agreement_for_annotator():
    # No data
    assert aggregate_agreement_for_annotator([], "a") == (None, 0)
    annotations = [
        # Slot with annotator a + b
        {"item_id": "i1", "evaluator_id": "e1", "annotator_id": "a", "value": {"value": True}},
        {"item_id": "i1", "evaluator_id": "e1", "annotator_id": "b", "value": {"value": True}},
        # Slot without target annotator — skipped
        {"item_id": "i2", "evaluator_id": "e1", "annotator_id": "c", "value": {"value": True}},
        # Numeric type
        {"item_id": "i3", "evaluator_id": "e1", "annotator_id": "a", "value": {"score": 3}},
        {"item_id": "i3", "evaluator_id": "e1", "annotator_id": "b", "value": {"score": 5}},
        # Mismatched type — skipped pair
        {"item_id": "i4", "evaluator_id": "e1", "annotator_id": "a", "value": {"value": True}},
        {"item_id": "i4", "evaluator_id": "e1", "annotator_id": "b", "value": {"value": "yes"}},
        # No scalar — skipped
        {"item_id": "i5", "evaluator_id": "e1", "annotator_id": None, "value": None},
    ]
    agree, pairs = aggregate_agreement_for_annotator(annotations, "a")
    assert pairs >= 2


def test_trend_series_empty_returns_buckets():
    now = datetime(2024, 1, 31, tzinfo=timezone.utc)
    series = trend_series([], "week", 14, now=now)
    assert all(b["agreement"] is None for b in series)
    assert all(b["pair_count"] == 0 for b in series)


def test_trend_series_populated():
    now = datetime(2024, 1, 31, tzinfo=timezone.utc)
    annotations = [
        {
            "item_id": "i1",
            "evaluator_id": "e1",
            "annotator_id": "a",
            "value": {"value": True},
            "updated_at": "2024-01-25 12:00:00",
        },
        {
            "item_id": "i1",
            "evaluator_id": "e1",
            "annotator_id": "b",
            "value": {"value": True},
            "updated_at": "2024-01-26 12:00:00",
        },
        # Malformed timestamp → skipped
        {
            "item_id": "i2",
            "annotator_id": "a",
            "value": {"value": True},
            "updated_at": "garbage",
        },
        # No timestamp → skipped
        {"item_id": "i3", "annotator_id": "a", "value": {"value": True}},
    ]
    series = trend_series(annotations, "week", 14, now=now)
    assert any(b["pair_count"] > 0 for b in series)


def test_trend_series_for_annotator():
    now = datetime(2024, 1, 31, tzinfo=timezone.utc)
    annotations = [
        {
            "item_id": "i1",
            "evaluator_id": "e1",
            "annotator_id": "a",
            "value": {"value": True},
            "updated_at": "2024-01-26 12:00:00",
        },
        {
            "item_id": "i1",
            "evaluator_id": "e1",
            "annotator_id": "b",
            "value": {"value": False},
            "updated_at": "2024-01-26 12:01:00",
        },
        {
            "item_id": "i2",
            "annotator_id": "a",
            "value": "garbage-ts",
            "updated_at": "not a date",
        },
    ]
    series = trend_series_for_annotator(annotations, "a", "week", 14, now=now)
    assert len(series) == 2

    # Empty inputs go through the empty-series branch
    empty = trend_series_for_annotator([], "a", "week", 14, now=now)
    assert all(b["agreement"] is None for b in empty)


def test_trend_series_human_evaluator_branches():
    now = datetime(2024, 1, 31, tzinfo=timezone.utc)
    # Empty evaluator list
    assert trend_series_human_evaluator([], [], [], "week", 14, now=now) == {}

    # No data path
    empty = trend_series_human_evaluator([], [], ["e1"], "week", 14, now=now)
    assert all(b["agreement"] is None for b in empty["e1"])

    annotations = [
        {
            "item_id": "i1",
            "evaluator_id": "e1",
            "annotator_id": "a",
            "value": {"value": True},
            "updated_at": "2024-01-26 12:00:00",
        }
    ]
    runs = [
        {
            "item_id": "i1",
            "evaluator_id": "e1",
            "evaluator_version_id": "v1",
            "value": {"value": True},
            "completed_at": "2024-01-26 13:00:00",
        },
        # malformed ts → skipped
        {
            "item_id": "i2",
            "evaluator_id": "e1",
            "evaluator_version_id": "v1",
            "value": {"value": True},
            "completed_at": "garbage",
        },
        # no ts → skipped
        {
            "item_id": "i3",
            "evaluator_id": "e1",
            "value": {"value": True},
        },
    ]
    series = trend_series_human_evaluator(
        annotations, runs, ["e1"], "week", 14, now=now
    )
    assert "e1" in series


def test_trend_series_evaluator_breakdown():
    now = datetime(2024, 1, 31, tzinfo=timezone.utc)
    # Empty data path
    out = trend_series_evaluator_breakdown(
        [], [], "e1", ["v1"], ["t1"], "week", 14, now=now
    )
    assert "overall" in out
    assert all(b["agreement"] is None for b in out["overall"])

    annotations = [
        {
            "item_id": "i1",
            "evaluator_id": "e1",
            "annotator_id": "a",
            "value": {"value": True},
            "task_id": "t1",
            "updated_at": "2024-01-26 12:00:00",
        }
    ]
    runs = [
        {
            "item_id": "i1",
            "evaluator_id": "e1",
            "evaluator_version_id": "v1",
            "task_id": "t1",
            "value": {"value": True},
            "completed_at": "2024-01-26 12:00:00",
        },
        # malformed ts annotation
        {
            "item_id": "i2",
            "evaluator_id": "e1",
            "evaluator_version_id": "v1",
            "task_id": "t1",
            "value": {"value": True},
            "completed_at": "bad",
        },
        # no ts at all → skipped
        {
            "item_id": "i3",
            "evaluator_id": "e1",
            "evaluator_version_id": "v1",
            "task_id": "t1",
            "value": {"value": True},
        },
    ]
    out = trend_series_evaluator_breakdown(
        annotations, runs, "e1", ["v1"], ["t1"], "week", 14, now=now
    )
    assert "v1" in out["by_version"]
    assert "t1" in out["by_task"]
