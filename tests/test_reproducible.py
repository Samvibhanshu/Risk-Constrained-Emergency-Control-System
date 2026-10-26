"""Re-running the experiment gives the committed results."""
import json

import pytest

COMMITTED = json.load(open("outputs/results_summary.json"))


def by_model(summary):
    return {m["model"]: m for m in summary["model_comparison"]}


def test_same_model_and_threshold_selected(fresh_run):
    s = fresh_run["summary"]
    assert s["selected_model"] == COMMITTED["selected_model"] == "Gradient Boosting"
    assert s["protective_threshold_tau"] == COMMITTED["protective_threshold_tau"]


def test_selected_model_scores_exactly_as_committed(fresh_run):
    new, old = by_model(fresh_run["summary"]), by_model(COMMITTED)
    best = COMMITTED["selected_model"]
    for k, v in old[best].items():
        if k != "inference_s_1000":
            assert new[best][k] == pytest.approx(v, abs=1e-4), k


def test_other_models_within_a_few_test_records(fresh_run):
    # Random Forest and Logistic Regression can move by a record or two across machines and BLAS builds
    # (the pinned versions in requirements.txt reproduce them exactly); 0.006 is six records out of 1,000.
    new, old = by_model(fresh_run["summary"]), by_model(COMMITTED)
    for name in old:
        for k in ("accuracy", "harmful_recall", "false_safe_rate", "weighted_f1"):
            assert new[name][k] == pytest.approx(old[name][k], abs=0.006), (name, k)


def test_control_layer_numbers_match(fresh_run):
    s = fresh_run["summary"]
    for k in ("screening", "admitted_protective_actually_unsafe", "harmful_test_cases",
              "harmful_caught_below_abs_threshold", "rollback_eligible", "rollback_success", "mean_trust_by_outcome"):
        assert s[k] == COMMITTED[k], k


def test_headline_numbers_quoted_in_the_readme(fresh_run):
    best = by_model(fresh_run["summary"])["Gradient Boosting"]
    assert round(best["false_safe_rate"] * 100, 1) == 0.8
    assert round(best["harmful_recall"] * 100, 1) == 92.9
    assert fresh_run["summary"]["harmful_test_cases"] == 396
