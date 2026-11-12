"""
Ensemble detector: LightGBM on readable features + CNN-LSTM
===========================================================
1. Builds readable per-customer features (zero-use streaks, monthly profile, ...)
   for the same customers and the same train/val/test split as
   11_train_lstm_detector_v2.py (--source raw, 50/50, seed 42).
2. Trains LightGBM on them (early stopping on validation).
3. Averages its theft probability with a finished LSTM run's probability.
   The decision threshold is tuned on validation only (best MCC).

The ensemble is the hybrid system's decision-maker. 12_explain_hybrid.py reads
the files saved here and uses LightGBM's per-feature contributions (SHAP values,
computed by LightGBM itself) as extra evidence for the LLM.

Needs : py -m pip install lightgbm
Usage : python 14_gbm_ensemble.py
        python 14_gbm_ensemble.py --lstm-run data/lstm-hybrid-v2/raw/<run> --no-push
Output: <lstm run>/ensemble/
          gbm_model.txt          LightGBM model (text; stays local)
          ensemble_config.json   threshold, weights, feature names
          ensemble_results.json  test metrics: LightGBM, LSTM, ensemble
          test_predictions.csv   per-customer probabilities and decision
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LSTM_RUN = SCRIPT_DIR / "data" / "lstm-hybrid-v2" / "raw" / "cnnlstm_minmax_col"
LSTM_WEIGHT = 0.5          # ensemble = 0.5 * LSTM + 0.5 * LightGBM
SEED = 42
GBM_PARAMS = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 31,
              "min_child_samples": 20, "feature_fraction": 0.7, "bagging_fraction": 0.8,
              "bagging_freq": 1, "verbose": -1, "seed": SEED}
DAYS = ["Mondays", "Tuesdays", "Wednesdays", "Thursdays", "Fridays", "Saturdays", "Sundays"]


def load_v2():
    spec = importlib.util.spec_from_file_location("detector_v2", SCRIPT_DIR / "11_train_lstm_detector_v2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────
# Readable features (names are what the LLM sees)
# ─────────────────────────────────────────────
def _longest_run(mask):
    run = best = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def _corr(a, b):
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.nan_to_num(np.corrcoef(a, b)[0, 1]))


def customer_features(x, m, dates):
    """One customer -> {readable name: value}. x = daily kWh (gaps filled), m = 1 where missing."""
    x = np.asarray(x, float)
    mean = x.mean()
    zero = (x <= 0.01) & (m == 0)
    f = {
        "average daily use (kWh)": mean,
        "highest daily use (kWh)": x.max(),
        "median daily use (kWh)": float(np.median(x)),
        "day-to-day variability (coefficient of variation)": x.std() / (mean + 1e-6),
        "share of days with zero or near-zero use (%)": 100 * zero.mean(),
        "longest run of zero-use days": _longest_run(zero),
        "missing meter readings (days)": int(m.sum()),
        "largest one-day drop in use (kWh)": float(-np.diff(x).min()),
        "similarity of each day to the day before (correlation)": _corr(x[:-1], x[1:]),
        "similarity of each day to the same day a week before (correlation)": _corr(x[:-7], x[7:]),
        "change in use per year (% of average)": 100 * 365 * np.polyfit(np.arange(len(x)), x, 1)[0] / (mean + 1e-6),
        "average daily use in the last 90 days (kWh)": x[-90:].mean(),
        "use in the last 90 days compared with own average (%)": 100 * x[-90:].mean() / (mean + 1e-6),
    }
    month = pd.Series(x).groupby(dates.to_period("M")).mean()
    for period, v in month.items():
        f[f"use in {period.strftime('%b %Y')} compared with own average (%)"] = 100 * v / (mean + 1e-6)
    dow = pd.Series(x).groupby(dates.dayofweek).mean()
    for d, v in dow.items():
        f[f"use on {DAYS[d]} compared with own average (%)"] = 100 * v / (mean + 1e-6)
    return {k: float(v) for k, v in f.items()}


def feature_table(X, M, cols):
    dates = pd.to_datetime(cols)
    return pd.DataFrame([customer_features(X[i], M[i], dates) for i in range(len(X))])


def lgb_frame(F):
    """LightGBM wants plain column names; the readable ones live in the config."""
    G = F.copy()
    G.columns = [f"f{i}" for i in range(F.shape[1])]
    return G


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    ap = argparse.ArgumentParser(description="LightGBM + LSTM ensemble detector")
    ap.add_argument("--lstm-run", default=str(DEFAULT_LSTM_RUN))
    ap.add_argument("--data", default=None)
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    v2 = load_v2()
    run_dir = Path(args.lstm_run)
    out_dir = run_dir / "ensemble"
    out_dir.mkdir(parents=True, exist_ok=True)

    path = Path(args.data) if args.data else v2.find_file("data/data.csv", "data.csv")
    print(f"Loading {path} ...")
    ids, X, M, y, cols = v2.load_raw(path)
    i_tr, i_va, i_te = v2._split(y)
    splits = json.loads((run_dir / "splits.json").read_text(encoding="utf-8"))
    if list(ids[i_te]) != splits["test"]:
        raise SystemExit("[ERROR] the LSTM run used a different split; train it with --source raw.")

    print("Building readable features ...")
    F = feature_table(X, M, cols)
    G = lgb_frame(F)
    model = lgb.train(GBM_PARAMS, lgb.Dataset(G.iloc[i_tr], y[i_tr]), 3000,
                      valid_sets=[lgb.Dataset(G.iloc[i_va], y[i_va])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    g_va, g_te = model.predict(G.iloc[i_va]), model.predict(G.iloc[i_te])

    def lstm_probs(name, ii):
        p = pd.read_csv(run_dir / f"{name}_predictions.csv", dtype={"CONS_NO": str}).set_index("CONS_NO")
        return p.theft_probability.loc[ids[ii]].to_numpy()

    l_va, l_te = lstm_probs("val", i_va), lstm_probs("test", i_te)
    e_va = LSTM_WEIGHT * l_va + (1 - LSTM_WEIGHT) * g_va
    e_te = LSTM_WEIGHT * l_te + (1 - LSTM_WEIGHT) * g_te

    results, thresholds = {}, {}
    for name, pv, pt in (("lightgbm", g_va, g_te), ("lstm", l_va, l_te), ("ensemble", e_va, e_te)):
        thr = v2.tune_threshold(y[i_va], pv)
        m = v2.metrics_at(y[i_te], pt, thr)
        m["val_auc"] = float(roc_auc_score(y[i_va], pv))
        results[name], thresholds[name] = m, thr
        print(f"{name:<9} valAUC={m['val_auc']:.3f}  test acc={m['accuracy']:.3f} F1={m['f1']:.3f} "
              f"AUC={m['auc']:.3f} MCC={m['mcc']:.3f}")

    model.save_model(str(out_dir / "gbm_model.txt"))
    (out_dir / "ensemble_config.json").write_text(json.dumps({
        "lstm_run": str(run_dir), "lstm_weight": LSTM_WEIGHT, "threshold": thresholds["ensemble"],
        "feature_names": list(F.columns), "best_iteration": int(model.best_iteration),
    }, indent=2), encoding="utf-8")
    (out_dir / "ensemble_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    pd.DataFrame({"CONS_NO": ids[i_te], "true_label": y[i_te], "lstm_probability": l_te,
                  "gbm_probability": g_te, "theft_probability": e_te,
                  "predicted_label": (e_te >= thresholds["ensemble"]).astype(int)}).to_csv(
        out_dir / "test_predictions.csv", index=False)
    print(f"Saved to {out_dir}")

    if not args.no_push:
        try:
            from repo_sync import push_run
        except ImportError:
            print("[repo_sync] repo_sync.py not found, skipping GitHub upload.")
        else:
            e = results["ensemble"]
            push_run(__file__, "lstm-hybrid-v2-ensemble",
                     [out_dir / "ensemble_results.json", out_dir / "ensemble_config.json"],
                     message=f"Ensemble: test F1={e['f1']:.3f} AUC={e['auc']:.3f}", allow_weights=False)


if __name__ == "__main__":
    main()
