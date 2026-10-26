"""The gate threshold is chosen on training data only, and is the smallest one that meets the 1 % target."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import train_eval
from simulate import generate
from train_eval_utils import TAU_GRID, false_safe_rate, safety_gated_predict, tune_tau


def test_tau_is_the_smallest_threshold_meeting_the_target():
    num, X_tr, X_te, y_tr, y_te, _ = train_eval.split(generate(2000, 5))
    model = train_eval.pipe(LogisticRegression(max_iter=2000, class_weight="balanced"), num, scale=True)
    tau = tune_tau(model, X_tr, y_tr)
    oof = cross_val_predict(model, X_tr, y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                            method="predict_proba")
    classes = np.sort(y_tr.unique())
    fs = lambda t: false_safe_rate(y_tr.values, safety_gated_predict(oof, classes, t))
    assert fs(tau) <= 0.01
    lower = [t for t in TAU_GRID if t < tau]
    assert all(fs(t) > 0.01 for t in lower)


def test_tau_does_not_depend_on_the_test_rows():
    df = generate(2000, 6)
    num, X_tr, X_te, y_tr, y_te, idx_te = train_eval.split(df)
    model = train_eval.pipe(LogisticRegression(max_iter=2000, class_weight="balanced"), num, scale=True)
    before = tune_tau(model, X_tr, y_tr)
    # change every test row; the split is fixed, so the training rows are the same
    df.loc[idx_te, num] = 0.0
    num2, X_tr2, *_ = train_eval.split(df)
    assert X_tr2.equals(X_tr)
    assert tune_tau(model, X_tr2, y_tr) == before


def test_full_run_tuned_every_gate_on_training_rows_only(fresh_run):
    df = generate(5000, 42)
    *_, idx_te = train_eval.split(df)
    test_rows = set(idx_te)
    assert len(fresh_run["tune_tau_rows"]) == 3            # one per model
    for rows in fresh_run["tune_tau_rows"]:
        assert len(rows) == 4000 and not rows & test_rows
