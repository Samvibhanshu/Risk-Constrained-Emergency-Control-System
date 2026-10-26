"""The browser demo (app/) runs an exported copy of the model; it must give scikit-learn's answers."""
import json
import shutil
import subprocess

import numpy as np
import pytest

from export_model import export, predict_exported
from predict import load
from simulate import generate
from train_eval_utils import feature_columns

DATA = generate(300, 11)


def sklearn_proba(bundle):
    return bundle["pipeline"].predict_proba(DATA)


def test_exported_trees_reproduce_scikit_learn():
    bundle = load()
    m = export(bundle)
    Xt = np.asarray(bundle["pipeline"].named_steps["pre"].transform(DATA), dtype=float)
    assert np.abs(predict_exported(m, Xt) - sklearn_proba(bundle)).max() < 1e-12


def test_committed_export_is_the_current_model():
    committed = json.load(open("outputs/model_export.json"))
    assert committed == json.loads(json.dumps(export(load())))
    js = open("app/model_export.js").read()
    assert js.startswith("const MODEL = ") and json.loads(js[len("const MODEL = "):].rstrip().rstrip(";")) == committed


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_browser_inference_matches_scikit_learn():
    bundle = load()
    rows = DATA[feature_columns(DATA.columns) + ["actuator_type"]].to_dict(orient="records")
    script = """
const fs = require('fs'), vm = require('vm');
const ctx = {}; vm.createContext(ctx);
vm.runInContext(fs.readFileSync('app/model_export.js', 'utf8') + fs.readFileSync('app/model.js', 'utf8')
  + '; this.predictProba = predictProba; this.gated = gated; this.MODEL = MODEL;', ctx);
const rows = JSON.parse(fs.readFileSync(0, 'utf8'));
process.stdout.write(JSON.stringify(rows.map(r => { const p = ctx.predictProba(r); return [ctx.MODEL.classes.map(c => p[c]), ctx.gated(p)]; })));
"""
    out = subprocess.run(["node", "-e", script], input=json.dumps(rows), capture_output=True, text=True, check=True)
    js = json.loads(out.stdout)
    js_proba = np.array([p for p, _ in js])
    assert np.abs(js_proba - sklearn_proba(bundle)).max() < 1e-12
    from train_eval_utils import safety_gated_predict
    py_decisions = safety_gated_predict(sklearn_proba(bundle), bundle["pipeline"].classes_, bundle["protective_tau"])
    assert [d for _, d in js] == list(py_decisions)
