"""
export_model.py - Exports the selected Gradient Boosting model to JSON so the browser demo (app/) can run the same
model with no server. Checks that the exported trees reproduce scikit-learn's probabilities before writing.

    python export_model.py        # after train_eval.py; writes app/model_export.js and outputs/model_export.json
"""
import json, pickle
import numpy as np
import pandas as pd


def export(bundle):
    """The model as plain lists: feature order, one-hot categories, class order, initial scores and every tree."""
    pipe, tau = bundle["pipeline"], bundle["protective_tau"]
    pre, gb = pipe.named_steps["pre"], pipe.named_steps["clf"]
    if not hasattr(gb, "estimators_"):
        raise SystemExit("export_model.py exports Gradient Boosting only; the selected model is " + type(gb).__name__)
    num = list(pre.transformers_[0][2])
    cats = list(pre.named_transformers_["cat"].categories_[0])
    xt0 = pre.transform(pd.DataFrame([dict.fromkeys(num, 0) | {"actuator_type": cats[0]}]))
    init = gb._raw_predict_init(np.asarray(xt0, dtype=float))[0].tolist()
    trees = []
    for stage in gb.estimators_:
        trees.append([[t.tree_.feature.tolist(), [float(v) for v in t.tree_.threshold.tolist()],
                       t.tree_.children_left.tolist(), t.tree_.children_right.tolist(),
                       [float(v) for v in t.tree_.value[:, 0, 0].tolist()]] for t in stage])
    return {"num": num, "cats": cats, "classes": [str(c) for c in gb.classes_], "lr": float(gb.learning_rate),
            "init": init, "tau": float(tau), "trees": trees}


def predict_exported(m, X):
    """Pure-Python inference over the exported structure, the same algorithm as app/model.js."""
    out = []
    X = np.asarray(X, dtype=np.float32)   # scikit-learn's trees compare in float32; so must we
    for x in X:
        r = np.array(m["init"], dtype=float)
        for stage in m["trees"]:
            for k, (f, th, left, right, val) in enumerate(stage):
                n = 0
                while left[n] != -1:
                    n = left[n] if float(x[f[n]]) <= th[n] else right[n]
                r[k] += m["lr"] * val[n]
        e = np.exp(r - r.max())
        out.append(e / e.sum())
    return np.array(out)


if __name__ == "__main__":
    bundle = pickle.load(open("outputs/selected_model.pkl", "rb"))
    m = export(bundle)
    df = pd.read_csv("data/intervention_records.csv")
    pre = bundle["pipeline"].named_steps["pre"]
    Xt = np.asarray(pre.transform(df[m["num"] + ["actuator_type"]][:200]), dtype=float)
    diff = np.abs(predict_exported(m, Xt) - bundle["pipeline"].named_steps["clf"].predict_proba(Xt)).max()
    print("max abs difference from scikit-learn on 200 records:", diff)
    assert diff < 1e-9, "exported model does not match"
    text = json.dumps(m, separators=(",", ":"))
    open("outputs/model_export.json", "w").write(text)
    open("app/model_export.js", "w").write("const MODEL = " + text + ";\n")
    print("wrote app/model_export.js and outputs/model_export.json,", len(text) // 1024, "KB")
