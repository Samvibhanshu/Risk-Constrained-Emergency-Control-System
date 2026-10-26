import os, sys
from unittest import mock

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # the scripts use paths relative to the repository root

import train_eval  # noqa: E402


@pytest.fixture(scope="session")
def fresh_run(tmp_path_factory):
    """Runs the whole experiment once (about 1.5 minutes) into a temporary folder, recording which rows every call
    to tune_tau received, so the tests can check the gate was tuned on training rows only."""
    out = tmp_path_factory.mktemp("run")
    seen = []
    real = train_eval.tune_tau

    def spy(model, X, y, *a, **k):
        seen.append(set(X.index))
        return real(model, X, y, *a, **k)

    with mock.patch.object(train_eval, "tune_tau", spy):
        summary = train_eval.run(out=str(out), data_path=str(out / "data.csv"), figures=False, verbose=False)
    return {"summary": summary, "out": out, "tune_tau_rows": seen}
