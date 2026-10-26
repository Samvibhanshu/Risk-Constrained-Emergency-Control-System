"""Shared pieces of the training pipeline and the screening step: which columns the model may see, the safety-gated
decision rule, the false-safe metric and the threshold search. Kept here so that training, prediction and the tests
all use exactly the same definitions."""
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict

SEED = 42

# Measured only AFTER an intervention is tried, so they must never be model inputs. outcome is the label itself.
LEAKAGE = ["dC_protected", "command_achievement_pct", "outcome"]
ID_COLS = ["record_id"]
CAT = ["actuator_type"]


def feature_columns(columns):
    """Numeric input columns: everything except identifiers, leakage columns and the categorical column."""
    return [c for c in columns if c not in LEAKAGE + ID_COLS + CAT]


def safety_gated_predict(proba, classes, tau):
    """Cost-sensitive decision rule: a candidate is labelled Protective only if the model is
    confident (P(Protective) >= tau). Otherwise it falls back to the most likely non-Protective
    class. A false rejection only delays escalation; a false-safe could expose a protected zone."""
    classes = np.asarray(classes)
    pi = int(np.where(classes == "Protective")[0][0])
    masked = proba.copy(); masked[:, pi] = -1
    fallback = classes[masked.argmax(1)]
    return np.where(proba[:, pi] >= tau, "Protective", fallback)


def false_safe_rate(y_true, y_pred):
    """Share of actual Harmful interventions predicted Protective (the most dangerous error)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    h = (y_true == "Harmful")
    return float(((y_pred == "Protective") & h).sum() / h.sum())


TAU_GRID = np.round(np.arange(0.30, 0.96, 0.01), 2)


def tune_tau(model, X, y, target_fs=0.01, seed=SEED):
    """Pick the smallest tau whose out-of-fold false-safe rate on the TRAINING set <= target.
    Only X and y are seen here: the caller passes the training split, so the test set never influences tau."""
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
    classes = np.sort(y.unique())
    for tau in TAU_GRID:
        if false_safe_rate(y.values, safety_gated_predict(oof, classes, tau)) <= target_fs:
            return float(tau)
    return 0.95


def hard_constraint(candidate):
    """Protected-zone rule applied before the model: never push air toward an occupied protected zone.
    Returns the reason for rejection, or None if the candidate may go to the model."""
    if int(candidate["occupancy"]) == 1 and int(candidate["airflow_direction_to_protected"]) == 1:
        return "airflow toward an occupied protected zone (hard constraint)"
    return None
