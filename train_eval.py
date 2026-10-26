"""
train_eval.py - Trains the intervention-outcome classifiers, selects the model on
safety (false-safe rate first), and evaluates the control layer:
screening, early harmful-gradient detection, rollback and actuator trust.

    python train_eval.py            # writes data/ and outputs/ (about 1.5 minutes on 2 cores)
"""
import json, os, time, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)
from simulate import generate, LABELS
from train_eval_utils import (SEED, LEAKAGE, CAT, feature_columns, safety_gated_predict,
                              false_safe_rate, tune_tau)

TREAT = {"Protective": "Eligible for bounded micro-intervention validation",
         "Harmful": "Rejected before autonomous escalation",
         "Ineffective": "Retained only as alternative / diagnostic action",
         "Inconsistent": "Additional sensing, fault attribution or operator review"}
ABS_THRESHOLD = 10.0
TRUST_FACTOR = {"Protective": 1.04, "Ineffective": 0.95, "Harmful": 0.85, "Inconsistent": 0.59}


def validate(df):
    """Checks run on every generated dataset before training."""
    assert df.isna().sum().sum() == 0, "missing values"
    assert not df.duplicated(subset=[c for c in df.columns if c != "record_id"]).any(), "duplicates"
    assert (df.protected_zone != df.source_zone).all(), "impossible zone combination"


def split(df, seed=SEED):
    """Stratified 80:20 split on the model inputs only (no leakage columns)."""
    num = feature_columns(df.columns)
    X, y = df[num + CAT], df["outcome"]
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, df.index, test_size=0.2, stratify=y, random_state=seed)
    return num, X_tr, X_te, y_tr, y_te, idx_te


def pipe(model, num, scale=False):
    pre = ColumnTransformer([("num", StandardScaler() if scale else "passthrough", num),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
    return Pipeline([("pre", pre), ("clf", model)])


def make_models(num, seed=SEED):
    return {
        "Random Forest": pipe(RandomForestClassifier(
            n_estimators=400, max_depth=12, min_samples_leaf=3,
            class_weight="balanced_subsample", random_state=seed, n_jobs=-1), num),
        "Gradient Boosting": pipe(GradientBoostingClassifier(
            n_estimators=250, learning_rate=0.05, max_depth=3, random_state=seed), num),
        "Logistic Regression": pipe(LogisticRegression(
            max_iter=3000, class_weight="balanced", C=1.0), num, scale=True),
    }


def evaluate(models, X_tr, y_tr, X_te, y_te):
    """Tune each model's gate on the training split, fit it, and score it once on the test split."""
    results, preds, taus = [], {}, {}
    for name, m in models.items():
        taus[name] = tune_tau(m, X_tr, y_tr)
        m.fit(X_tr, y_tr)
        t0 = time.perf_counter()
        p = safety_gated_predict(m.predict_proba(X_te), m.classes_, taus[name])
        dt = time.perf_counter() - t0
        preds[name] = p
        raw = m.predict(X_te)   # plain argmax, no safety gate (baseline for comparison)
        pw, rw, fw, _ = precision_recall_fscore_support(y_te, p, average="weighted", zero_division=0)
        _, rc, _, _ = precision_recall_fscore_support(y_te, p, labels=LABELS, zero_division=0)
        results.append({"model": name, "accuracy": accuracy_score(y_te, p),
                        "weighted_precision": pw, "weighted_recall": rw, "weighted_f1": fw,
                        "harmful_recall": rc[1], "protective_tau": taus[name],
                        "false_safe_rate": false_safe_rate(y_te.values, p),
                        "inference_s_1000": dt,
                        "accuracy_ungated": accuracy_score(y_te, raw),
                        "false_safe_rate_ungated": false_safe_rate(y_te.values, raw)})
    res = pd.DataFrame(results)
    # safety-first selection: lowest false-safe rate, then highest Harmful recall, then F1
    res = res.sort_values(["false_safe_rate", "harmful_recall", "weighted_f1"],
                          ascending=[True, False, False]).reset_index(drop=True)
    return res, preds, taus


def control_layer(test, predicted):
    """Screening counts, early harmful detection, rollback and trust update on the test split."""
    test = test.copy()
    test["predicted"] = predicted
    screen = test.predicted.value_counts().reindex(LABELS).fillna(0).astype(int)
    admitted = test[test.predicted == "Protective"]
    admitted_unsafe = int(admitted.outcome.isin(["Harmful", "Inconsistent"]).sum())
    # protected-zone hard filter on top of the model: occupied zone + airflow toward it -> reject
    hard_reject = (admitted.occupancy == 1) & (admitted.airflow_direction_to_protected == 1)
    # early harmful-gradient detection: harmful cases still under the absolute threshold
    harm = test[test.outcome == "Harmful"]
    early = harm[harm.dC_protected <= ABS_THRESHOLD]
    # rollback: eligible = Harmful or Inconsistent in test set
    elig = test[test.outcome.isin(["Harmful", "Inconsistent"])]
    ok = (elig.command_achievement_pct >= 45) & (elig.communication_delay_ms <= 300)
    test["trust_after"] = np.clip(test.actuator_trust * test.outcome.map(TRUST_FACTOR), 0, 1)
    trust = test.groupby("outcome")[["actuator_trust", "trust_after"]].mean().reindex(LABELS)
    return test, dict(screen=screen, admitted_unsafe=admitted_unsafe, hard_reject=int(hard_reject.sum()),
                      harm=harm, early=early, elig=elig, ok=ok, trust=trust)


def run(out="outputs/", data_path="data/intervention_records.csv", figures=True, verbose=True, seed=SEED):
    """The whole experiment. Returns the summary that is also written to <out>/results_summary.json."""
    say = print if verbose else (lambda *a, **k: None)
    os.makedirs(out, exist_ok=True)
    df = generate(5000, seed)
    if data_path:
        os.makedirs(os.path.dirname(data_path) or ".", exist_ok=True)
        df.to_csv(data_path, index=False)
    validate(df)

    num, X_tr, X_te, y_tr, y_te, idx_te = split(df, seed)
    test = df.loc[idx_te].copy()
    say(f"Train {len(X_tr)} | Test {len(X_te)}")
    say("Class counts (full):\n", df["outcome"].value_counts().reindex(LABELS), "\n")

    models = make_models(num, seed)
    res, preds, taus = evaluate(models, X_tr, y_tr, X_te, y_te)
    best = res.loc[0, "model"]
    say(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    say(f"\nSelected model (safety-first): {best}  (Protective gate tau = {taus[best]})\n")
    res.to_csv(os.path.join(out, "model_comparison.csv"), index=False)

    bm = models[best]; bp = preds[best]
    say(classification_report(y_te, bp, labels=LABELS, digits=4))
    with open(os.path.join(out, "selected_model.pkl"), "wb") as f:
        pickle.dump({"pipeline": bm, "protective_tau": taus[best]}, f)

    clf = bm.named_steps["clf"]
    names = num + list(bm.named_steps["pre"].named_transformers_["cat"].get_feature_names_out(CAT))
    imp = (clf.feature_importances_ if hasattr(clf, "feature_importances_")
           else np.abs(clf.coef_).mean(axis=0))
    fi = pd.Series(imp, index=names).sort_values(ascending=False)
    fi.to_csv(os.path.join(out, "feature_importance.csv"), header=["importance"])

    test, c = control_layer(test, bp)
    if figures:
        draw_figures(out, df["outcome"], res, list(models), y_te, bp, best, fi, c["trust"])

    summary = {
        "selected_model": best,
        "protective_threshold_tau": taus[best],
        "model_comparison": res.round(4).to_dict(orient="records"),
        "screening": {k: {"records": int(c["screen"][k]), "treatment": TREAT[k]} for k in LABELS},
        "admitted_protective_actually_unsafe": c["admitted_unsafe"],
        "extra_hard_filter_rejections_among_admitted": c["hard_reject"],
        "harmful_test_cases": int(len(c["harm"])),
        "harmful_caught_below_abs_threshold": int(len(c["early"])),
        "harmful_caught_below_abs_threshold_pct": round(100 * len(c["early"]) / len(c["harm"]), 2),
        "rollback_eligible": int(len(c["elig"])), "rollback_success": int(c["ok"].sum()),
        "rollback_success_pct": round(100 * c["ok"].mean(), 2),
        "mean_trust_by_outcome": c["trust"].round(4).to_dict(orient="index"),
        "top_features": fi.head(10).round(4).to_dict(),
    }
    with open(os.path.join(out, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        spread = pd.crosstab(df.actuator_type, df.outcome)[LABELS]   # per-actuator outcome spread (shortcut check)
        test["_d"] = (test.command_achievement_pct - 92).abs()        # one real test record per class
        scen = (test[test.predicted == test.outcome].sort_values("_d")
                    .groupby("outcome").head(1).set_index("outcome").reindex(LABELS)
                    [["record_id", "actuator_type", "intervention_magnitude_pct",
                      "dC_protected", "command_achievement_pct", "predicted"]])
        harm, early, elig, ok = c["harm"], c["early"], c["elig"], c["ok"]
        print("Screening:\n", c["screen"].to_string())
        print(f"Admitted as Protective but actually Harmful/Inconsistent: {c['admitted_unsafe']}")
        print(f"Harmful caught below {ABS_THRESHOLD}-unit threshold: {len(early)}/{len(harm)} "
              f"({100*len(early)/len(harm):.2f}%)")
        print(f"Rollback success: {ok.sum()}/{len(elig)} ({100*ok.mean():.2f}%)")
        print("Mean trust:\n", c["trust"].round(4).to_string())
        print("\nOutcome spread by actuator:\n", spread.to_string())
        print("\nRepresentative scenarios:\n", scen.to_string())
        print("\nTop features:\n", fi.head(10).round(4).to_string())
    return summary


def draw_figures(out, y, res, order, y_te, bp, best, fi, trust):
    plt.rcParams.update({"figure.dpi": 150, "font.size": 10})
    p = lambda name: os.path.join(out, name)

    vc = y.value_counts().reindex(LABELS)
    plt.figure(figsize=(7, 4)); b = plt.bar(LABELS, vc.values, color="#1f77b4")
    plt.bar_label(b); plt.ylabel("Number of records"); plt.title("Class distribution of the simulated dataset")
    plt.tight_layout(); plt.savefig(p("fig_class_distribution.png")); plt.close()

    metrics = ["accuracy", "weighted_f1", "harmful_recall"]
    r = res.set_index("model").loc[order]
    x = np.arange(len(r)); w = 0.26
    plt.figure(figsize=(8, 4.2))
    for i, mt in enumerate(metrics):
        bb = plt.bar(x + (i - 1) * w, r[mt] * 100, w, label=mt.replace("_", " ").title())
        plt.bar_label(bb, fmt="%.1f", fontsize=7)
    plt.xticks(x, r.index); plt.ylabel("%"); plt.ylim(0, 112); plt.legend(loc="upper center", ncol=3, fontsize=8)
    plt.title("Model performance comparison"); plt.tight_layout()
    plt.savefig(p("fig_model_comparison.png")); plt.close()

    cm = confusion_matrix(y_te, bp, labels=LABELS)
    plt.figure(figsize=(6, 5.2)); plt.imshow(cm, cmap="viridis")
    for i in range(4):
        for j in range(4):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color="black" if cm[i, j] > cm.max() * 0.6 else "white")
    plt.xticks(range(4), LABELS, rotation=30); plt.yticks(range(4), LABELS)
    plt.xlabel("Predicted class"); plt.ylabel("Actual class")
    plt.title(f"Confusion matrix - {best}"); plt.colorbar(); plt.tight_layout()
    plt.savefig(p("fig_confusion_matrix.png")); plt.close()

    top = fi.head(10)[::-1]
    plt.figure(figsize=(7.5, 4.8)); plt.barh(top.index, top.values)
    plt.xlabel("Importance"); plt.title(f"Top-10 feature importances - {best}")
    plt.tight_layout(); plt.savefig(p("fig_feature_importance.png")); plt.close()

    plt.figure(figsize=(7.5, 4.2)); xx = np.arange(4)
    b1 = plt.bar(xx - 0.2, trust.actuator_trust, 0.4, label="Before update")
    b2 = plt.bar(xx + 0.2, trust.trust_after, 0.4, label="After update")
    plt.bar_label(b1, fmt="%.3f", fontsize=7); plt.bar_label(b2, fmt="%.3f", fontsize=7)
    plt.xticks(xx, LABELS); plt.ylabel("Mean actuator trust"); plt.legend()
    plt.title("Mean actuator trust before and after validation"); plt.tight_layout()
    plt.savefig(p("fig_actuator_trust.png")); plt.close()


if __name__ == "__main__":
    run()
