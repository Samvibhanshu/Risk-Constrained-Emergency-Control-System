"""
predict.py - Screen one candidate micro-intervention with the saved model.
Usage: python predict.py
"""
import pickle
import pandas as pd
from train_eval_utils import safety_gated_predict, hard_constraint

TREAT = {"Protective": "ADMIT to bounded micro-intervention test (escalate only if measured response confirms)",
         "Harmful": "REJECT before execution",
         "Ineffective": "HOLD as alternative / diagnostic action",
         "Inconsistent": "REVIEW: extra sensing, fault attribution or operator approval"}

EXAMPLE = {
    "source_zone": 1, "protected_zone": 3, "hop_distance": 2,
    "initial_hazard_concentration": 32.0, "spatial_gradient": -0.6, "temporal_gradient": -0.2,
    "airflow_speed_mps": 1.4, "airflow_direction_to_protected": 0, "pressure_diff_pa": 3.0,
    "oxygen_pct": 20.1, "door_open": 0, "blocked_path": 0, "sensor_fault": 0, "actuator_fault": 0,
    "communication_delay_ms": 80.0, "actuator_type": "damper", "intervention_magnitude_pct": 25.0,
    "intervention_duration_s": 8.0, "actuator_trust": 0.86, "occupancy": 1, "evacuation_route_exposure": 0,
}


def load(path="outputs/selected_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def screen(candidate, bundle):
    """Returns (decision, probabilities or None, reason). The hard constraint is checked before the model runs."""
    reason = hard_constraint(candidate)
    if reason:
        return "Rejected", None, reason
    model, tau = bundle["pipeline"], bundle["protective_tau"]
    proba = model.predict_proba(pd.DataFrame([candidate]))
    label = safety_gated_predict(proba, model.classes_, tau)[0]
    return label, {c: float(p) for c, p in zip(model.classes_, proba[0])}, TREAT[label]


if __name__ == "__main__":
    decision, proba, reason = screen(EXAMPLE, load())
    if proba is None:
        print("REJECT:", reason)
    else:
        print("Class probabilities:", {c: round(p, 3) for c, p in proba.items()})
        print(f"Decision: {decision} -> {reason}")
