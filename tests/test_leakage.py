"""Post-intervention measurements must never reach the model."""
import numpy as np
from sklearn.tree import DecisionTreeClassifier

import train_eval
from predict import load
from simulate import generate
from train_eval_utils import LEAKAGE, feature_columns

POST = ["dC_protected", "command_achievement_pct"]


def test_feature_list_excludes_leakage_and_ids():
    cols = feature_columns(generate(50, 0).columns)
    assert not set(cols) & set(LEAKAGE + ["record_id", "actuator_type"])
    assert "airflow_direction_to_protected" in cols and "actuator_trust" in cols


def test_split_hands_the_model_no_leakage_columns():
    num, X_tr, X_te, *_ = train_eval.split(generate(500, 3))
    for X in (X_tr, X_te):
        assert not set(X.columns) & set(LEAKAGE)


def test_saved_model_reads_no_leakage_columns():
    pre = load()["pipeline"].named_steps["pre"]
    used = [c for _, _, cols in pre.transformers_ for c in (cols if isinstance(cols, list) else [])]
    assert used and not set(used) & set(LEAKAGE)


def test_predictions_ignore_post_intervention_columns_even_if_present():
    model = load()["pipeline"]
    df = generate(300, 9)
    shuffled = df.copy()
    rng = np.random.default_rng(0)
    for c in POST:
        shuffled[c] = rng.permutation(shuffled[c].values)
    assert np.array_equal(model.predict_proba(df), model.predict_proba(shuffled))


def test_why_it_matters_the_leaked_columns_alone_almost_give_the_answer():
    # The labels are a function of the post-intervention measurements, so a model that saw them would look
    # near-perfect and be useless before the action is tried. This is the number the guard keeps out.
    df = generate(5000, 42)
    X = df[POST + ["airflow_direction_to_protected", "sensor_fault"]]
    tree = DecisionTreeClassifier(max_depth=6, random_state=0).fit(X[:4000], df.outcome[:4000])
    assert tree.score(X[4000:], df.outcome[4000:]) > 0.97
