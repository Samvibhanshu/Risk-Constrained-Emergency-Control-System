"""The simulator: deterministic, physically consistent, and labelled exactly by the stated rules."""
import pandas as pd
import pytest

from simulate import generate, label_record, LABELS


def test_same_seed_same_data_and_different_seed_different_data():
    a, b, c = generate(400, 7), generate(400, 7), generate(400, 8)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


def test_committed_dataset_is_what_the_generator_produces():
    committed = pd.read_csv("data/intervention_records.csv")
    pd.testing.assert_frame_equal(committed, generate(5000, 42), check_dtype=False, atol=1e-9)


def test_every_label_follows_the_rules():
    d = generate(3000, 1)
    expected = [label_record(r.dC_protected, r.airflow_direction_to_protected, r.sensor_fault, r.command_achievement_pct)
                for r in d.itertuples()]
    assert list(d.outcome) == expected
    assert set(d.outcome) == set(LABELS)


def test_zones_are_never_the_same_and_hop_distance_matches_the_corridor():
    d = generate(3000, 2)
    assert (d.source_zone != d.protected_zone).all()
    assert (d.hop_distance == (d.source_zone - d.protected_zone).abs()).all()
    assert d.hop_distance.between(1, 3).all()


@pytest.mark.parametrize("dC, toward, fault, achieved, expected", [
    (-20, 0, 1, 99, "Inconsistent"),     # a sensor fault overrides everything
    (-20, 0, 0, 44.99, "Inconsistent"),  # command not achieved
    (-20, 0, 0, 45, "Protective"),       # 45 % is enough
    (6.01, 0, 0, 90, "Harmful"),
    (6.0, 0, 0, 90, "Ineffective"),      # rise of exactly 6 away from the zone is not harmful
    (1.01, 1, 0, 90, "Harmful"),         # with airflow toward the zone, any rise above 1 is harmful
    (1.0, 1, 0, 90, "Ineffective"),
    (-5.01, 0, 0, 90, "Protective"),
    (-5.0, 0, 0, 90, "Ineffective"),
])
def test_label_rule_boundaries(dC, toward, fault, achieved, expected):
    assert label_record(dC, toward, fault, achieved) == expected
