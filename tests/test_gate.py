"""The safety gate: Protective only above the threshold, otherwise the best non-Protective class."""
import numpy as np

from train_eval_utils import safety_gated_predict, false_safe_rate

CLASSES = np.array(["Harmful", "Inconsistent", "Ineffective", "Protective"])


def test_protective_needs_at_least_tau_even_when_it_is_the_most_likely_class():
    proba = np.array([[0.10, 0.05, 0.15, 0.70],    # most likely Protective, but below tau
                      [0.05, 0.05, 0.10, 0.80],    # exactly tau
                      [0.30, 0.10, 0.10, 0.50]])
    assert list(safety_gated_predict(proba, CLASSES, 0.80)) == ["Ineffective", "Protective", "Harmful"]


def test_never_protective_below_tau_on_random_probabilities():
    rng = np.random.default_rng(0)
    proba = rng.dirichlet(np.ones(4), size=5000)
    for tau in (0.3, 0.5, 0.84, 0.95):
        pred = safety_gated_predict(proba, CLASSES, tau)
        assert not ((pred == "Protective") & (proba[:, 3] < tau)).any()
        assert ((pred == "Protective") == (proba[:, 3] >= tau)).all()


def test_raising_tau_only_removes_protective_decisions():
    rng = np.random.default_rng(1)
    proba = rng.dirichlet(np.ones(4), size=5000)
    previous = None
    for tau in np.round(np.arange(0.30, 0.96, 0.05), 2):
        admitted = safety_gated_predict(proba, CLASSES, tau) == "Protective"
        if previous is not None:
            assert not (admitted & ~previous).any()
        previous = admitted


def test_class_order_does_not_matter():
    rng = np.random.default_rng(2)
    proba = rng.dirichlet(np.ones(4), size=500)
    order = [3, 0, 2, 1]
    a = safety_gated_predict(proba, CLASSES, 0.6)
    b = safety_gated_predict(proba[:, order], CLASSES[order], 0.6)
    assert (a == b).all()


def test_false_safe_rate_counts_harmful_cases_admitted_as_protective():
    y = np.array(["Harmful", "Harmful", "Harmful", "Harmful", "Protective", "Ineffective"])
    p = np.array(["Protective", "Harmful", "Ineffective", "Harmful", "Protective", "Protective"])
    assert false_safe_rate(y, p) == 0.25   # 1 of 4 harmful; the non-harmful mistakes do not count
