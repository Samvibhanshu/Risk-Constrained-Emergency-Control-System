"""Screening a single candidate."""
import pytest

from predict import EXAMPLE, TREAT, load, screen
from simulate import LABELS
import json


def test_hard_constraint_rejects_before_the_model_runs():
    candidate = dict(EXAMPLE, occupancy=1, airflow_direction_to_protected=1)
    decision, proba, reason = screen(candidate, bundle=None)   # bundle=None: the model must not be touched
    assert decision == "Rejected" and proba is None and "occupied" in reason


def test_example_candidate_gets_a_gated_decision():
    decision, proba, reason = screen(EXAMPLE, load())
    assert decision in LABELS and reason == TREAT[decision]
    assert sum(proba.values()) == pytest.approx(1.0)
    tau = load()["protective_tau"]
    assert (decision == "Protective") == (proba["Protective"] >= tau)


def test_saved_model_matches_the_committed_summary():
    bundle = load()
    summary = json.load(open("outputs/results_summary.json"))
    assert bundle["protective_tau"] == summary["protective_threshold_tau"]
    assert type(bundle["pipeline"].named_steps["clf"]).__name__ == "GradientBoostingClassifier"
