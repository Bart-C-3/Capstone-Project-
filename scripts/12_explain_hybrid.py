"""
Hybrid pipeline, Steps 3-6: evidence extractor + LLM narrator + guardrails
=========================================================================
The detector has already made every decision. It is either the LSTM alone
(11_train_lstm_detector_v2.py) or, if 14_gbm_ensemble.py has been run for that
LSTM run, the ensemble (average of the LSTM and LightGBM). This script never
changes a decision or a probability. For each flagged customer it:

  3. Builds an EVIDENCE JSON:
       * decision + probability, copied from the detector
       * ensemble mode: the features that pushed LightGBM toward "theft" most
         (LightGBM's own per-feature contributions, i.e. SHAP values), with the
         customer's value next to the typical normal customer's value
       * "suspicious periods": 4-week windows found by occlusion. Each window is
         replaced by that customer's own typical level and the LSTM is re-run;
         the bigger the drop in theft probability, the more that window drove
         the decision.
       * how the customer's usage compares with NORMAL customers from the
         training split (percentile ranks of simple, readable features).
  4. Turns the evidence into short plain fact sentences and asks a small LLM
     (TinyLlama-1.1B-Chat by default) to rewrite them as one plain paragraph,
     with one worked example in the prompt. The LLM never sees anything else.
  5. Guardrails on the text:
       * every number and date in the text must appear in the evidence
       * no contradiction of the decision, no wrong "higher/lower" claims
       * no invented content seen in earlier runs (talking about the JSON file,
         the "language model", locations, training data, the future, jargon)
       * must mention the main suspicious period, a warning-sign value and an
         on-site check; must end with a full sentence (cut-off replies are
         trimmed to their last full sentence first)
     If the LLM fails twice, a fixed template writes the summary instead, so the
     output is always faithful. Each result records which writer was used.
  6. Explanation checks (written to explanation_eval.json):
       * deletion test: does hiding the top periods lower the theft probability
         more than hiding random periods? (are the periods really what the LSTM used)
       * ensemble mode, feature deletion test: does setting the top features to
         typical normal values lower LightGBM's theft score more than doing the
         same to random features?
       * guardrail pass rate of the LLM, and readability (Flesch reading ease)

Inputs : the run folder of 11_ (default data/lstm-hybrid-v2/raw/cnnlstm_minmax_col)
         and the same data file it was trained on.
Outputs: <run folder>/explanations/
           evidence/<CONS_NO>.json, explanations.jsonl, explanations.md,
           explanation_eval.json

Usage
    python 12_explain_hybrid.py                     top 20 flagged test customers, TinyLlama
    python 12_explain_hybrid.py --no-llm            template writer only (fast, no download)
    python 12_explain_hybrid.py --customer-id <CONS_NO>
    python 12_explain_hybrid.py --top-n 50 --no-push
    python 12_explain_hybrid.py --mode lstm         ignore the ensemble, explain the LSTM alone
    python 12_explain_hybrid.py --llm-id Qwen/Qwen2.5-1.5B-Instruct   stronger small model (fits 8 GB)
"""

import argparse
import collections
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN = SCRIPT_DIR / "data" / "lstm-hybrid-v2" / "raw" / "cnnlstm_minmax_col"
LLM_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
WINDOW_DAYS = 28
TOP_PERIODS = 3
SEED = 42
OVERALL_KEY = "customer's overall average daily use (kWh)"


def load_v2():
    """Import 11_train_lstm_detector_v2.py (its name starts with a digit)."""
    path = SCRIPT_DIR / "11_train_lstm_detector_v2.py"
    spec = importlib.util.spec_from_file_location("detector_v2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────
# Model + data
# ─────────────────────────────────────────────
def restore_scaler(v2, cfg):
    sc = v2.Scaler(cfg["scaler"]["kind"])
    st = {k: v for k, v in cfg["scaler"].items() if k != "kind"}
    sc.state = {k: (np.asarray(v, np.float32) if isinstance(v, list) else v) for k, v in st.items()}
    return sc


class Scorer:
    """Wraps the trained LSTM: raw kWh rows in, theft probabilities out."""

    def __init__(self, v2, run_dir, device):
        self.cfg = json.loads((run_dir / "model_config.json").read_text(encoding="utf-8"))
        self.scaler = restore_scaler(v2, self.cfg)
        self.model = v2.build_model(self.cfg["arch"], self.cfg["n_days"], self.cfg["use_mask"]).to(device)
        self.model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True))
        self.model.eval()
        self.device = device
        self.threshold = float(self.cfg["threshold"])

    @torch.no_grad()
    def __call__(self, X, M):
        S, C = self.scaler.transform(np.asarray(X, np.float32))
        out = []
        for i in range(0, len(S), 256):
            xb = torch.from_numpy(S[i:i + 256]).to(self.device)
            mb = torch.from_numpy(np.asarray(M[i:i + 256], np.float32)).to(self.device)
            cb = torch.from_numpy(C[i:i + 256]).to(self.device)
            out.append(torch.softmax(self.model(xb, mb, cb), 1)[:, 1].cpu().numpy())
        return np.concatenate(out)


# ─────────────────────────────────────────────
# Step 3: evidence
# ─────────────────────────────────────────────
def readable_features(x, m):
    """Simple per-customer features a non-expert can understand."""
    n = len(x)
    half = n // 2
    first, second = x[:half].mean(), x[half:].mean()
    zero = (x <= 0.01) & (m == 0)
    run = best = 0
    for z in zero:
        run = run + 1 if z else 0
        best = max(best, run)
    last90 = x[-90:].mean()
    return {
        "average daily use (kWh)": float(x.mean()),
        "change from first half to second half of the period (%)": float(100 * (second - first) / (first + 1e-6)),
        "days with zero or near-zero use": int(zero.sum()),
        "longest run of zero-use days": int(best),
        "missing meter readings (days)": int(m.sum()),
        "day-to-day variability (coefficient of variation)": float(x.std() / (x.mean() + 1e-6)),
        "average daily use in the last 90 days (kWh)": float(last90),
    }


def normal_baseline(X, M, y, train_idx):
    rows = [readable_features(X[i], M[i]) for i in train_idx if y[i] == 0]
    return pd.DataFrame(rows)


def occlusion_periods(scorer, x, m, base_prob, dates, window=WINDOW_DAYS):
    """Replace each window with the customer's median and measure the probability drop."""
    n = len(x)
    starts = list(range(0, n - window + 1, window))
    if starts[-1] + window < n:
        starts.append(n - window)
    med = float(np.median(x))
    Xo = np.repeat(x[None, :], len(starts), 0)
    Mo = np.repeat(m[None, :], len(starts), 0)
    for k, s in enumerate(starts):
        Xo[k, s:s + window] = med
        Mo[k, s:s + window] = 0.0
    # rank by log-odds drop: probabilities near 1.0 hide large changes
    logit = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))
    occ = scorer(Xo, Mo)
    drops = logit(base_prob) - logit(occ)
    order = np.argsort(-drops)
    periods = []
    for k in order[:TOP_PERIODS]:
        s = starts[k]
        seg = x[s:s + window]
        periods.append({
            "start_date": str(dates[s].date()), "end_date": str(dates[s + window - 1].date()),
            "lstm_theft_probability_if_period_looked_normal": float(occ[k]),
            "average daily use in this period (kWh)": float(seg.mean()),
            OVERALL_KEY: float(x.mean()),
            "missing readings in this period (days)": int(m[s:s + window].sum()),
        })
    return periods, starts, drops


class FeatureModel:
    """LightGBM half of the ensemble (from 14_gbm_ensemble.py) + normal-customer baseline."""

    def __init__(self, ens_dir, X, M, y, cols, train_idx):
        import lightgbm as lgb
        spec = importlib.util.spec_from_file_location("gbm14", SCRIPT_DIR / "14_gbm_ensemble.py")
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)
        self.cfg = json.loads((ens_dir / "ensemble_config.json").read_text(encoding="utf-8"))
        self.booster = lgb.Booster(model_file=str(ens_dir / "gbm_model.txt"))
        self.names = self.cfg["feature_names"]
        self.dates = pd.to_datetime(cols)
        normals = [i for i in train_idx if y[i] == 0]
        self.base = self.mod.feature_table(X[normals], M[normals], cols)[self.names]
        self.medians = self.base.median()

    def frame(self, feats_list):
        return self.mod.lgb_frame(pd.DataFrame(feats_list)[self.names])

    def score(self, x, m):
        feats = self.mod.customer_features(x, m, self.dates)
        G = self.frame([feats])
        it = self.cfg.get("best_iteration") or None
        prob = float(self.booster.predict(G, num_iteration=it)[0])
        contrib = self.booster.predict(G, num_iteration=it, pred_contrib=True)[0][:-1]   # last = bias
        return prob, feats, contrib

    def predict_feats(self, feats_list):
        it = self.cfg.get("best_iteration") or None
        return self.booster.predict(self.frame(feats_list), num_iteration=it)

    def evidence(self, feats, contrib, n_theft=4, n_normal=2):
        order = np.argsort(-contrib)
        def item(k):
            name = self.names[k]
            return {"feature": name, "customer_value": feats[name],
                    "typical_normal_customer_value": float(self.medians[name]),
                    "percentile_among_normal_customers": percentile_vs_normal(feats[name], self.base[name].to_numpy())}
        toward_theft = [item(k) for k in order[:n_theft] if contrib[k] > 0]
        toward_normal = [item(k) for k in order[::-1][:n_normal] if contrib[k] < 0]
        return toward_theft, toward_normal, order


def feature_deletion_test(fm, feats, order, rng, k=3):
    """Set the top-k theft features to typical normal values vs k random features (log-odds drop)."""
    logit = lambda q: float(np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6))))
    def drop(keys):
        f2 = dict(feats)
        for kk in keys:
            f2[fm.names[kk]] = float(fm.medians[fm.names[kk]])
        return f2
    top = list(order[:k])
    rand = list(rng.choice(len(fm.names), size=k, replace=False))
    p0, pt, pr = fm.predict_feats([feats, drop(top), drop(rand)])
    return logit(p0) - logit(pt), logit(p0) - logit(pr)


def percentile_vs_normal(value, col):
    return float(100.0 * (col < value).mean() + 50.0 * (col == value).mean())


def r(v, nd=2):
    """Round numbers once, here, so the LLM sees exactly the numbers it may use."""
    if isinstance(v, float):
        return round(v, nd)
    if isinstance(v, dict):
        return {k: r(x, nd) for k, x in v.items()}
    if isinstance(v, list):
        return [r(x, nd) for x in v]
    return v


def build_evidence(cid, prob, threshold, feats, base, periods, ensemble=None):
    """ensemble: None (LSTM only) or dict with lstm/gbm probabilities and feature evidence."""
    comparisons = []
    for name, val in feats.items():
        pct = percentile_vs_normal(val, base[name].to_numpy())
        comparisons.append({
            "feature": name, "customer_value": val,
            "typical_normal_customer_value": float(base[name].median()),
            "percentile_among_normal_customers": pct,
            "unusual": bool(pct <= 10 or pct >= 90),
        })
    comparisons.sort(key=lambda c: -abs(c["percentile_among_normal_customers"] - 50))
    ev = {
        "customer_id": cid,
        "decision": "flagged as possible electricity theft" if prob >= threshold else "not flagged",
        "decided_by": "LSTM detector (the language model does not change this)",
        "theft_probability": prob,
        "decision_threshold": threshold,
        "suspicious_periods": periods,
    }
    if ensemble is None:
        ev["comparison_with_normal_customers"] = comparisons[:5]
    else:
        ev["decided_by"] = ("average of an LSTM model (reads the daily usage curve) and a LightGBM model "
                            "(reads summary features); the language model does not change this")
        ev["lstm_theft_probability"] = ensemble["lstm_probability"]
        ev["feature_model_theft_probability"] = ensemble["gbm_probability"]
        ev["features_that_pushed_toward_theft"] = ensemble["toward_theft"]
        ev["features_that_pushed_toward_normal"] = ensemble["toward_normal"]
    return r(ev)


# ─────────────────────────────────────────────
# Step 4a: evidence -> short plain facts
# ─────────────────────────────────────────────
# A 1.1B model cannot reliably read nested JSON (it starts describing the file),
# so it gets short fact sentences instead. Every number in them comes from the evidence.
MONTH_RE = re.compile(r"^use in (\w{3} \d{4}) compared with own average \(%\)$")
DAY_RE = re.compile(r"^use on (\w+) compared with own average \(%\)$")
SHORT_NAMES = {
    "average daily use (kWh)": ("average daily use", "kWh"),
    "highest daily use (kWh)": ("highest daily use", "kWh"),
    "median daily use (kWh)": ("typical (median) daily use", "kWh"),
    "day-to-day variability (coefficient of variation)": ("how much use jumps from day to day", ""),
    "share of days with zero or near-zero use (%)": ("share of days with almost no use", "%"),
    "longest run of zero-use days": ("longest run of days with no use", "days"),
    "missing meter readings (days)": ("missing meter readings", "days"),
    "largest one-day drop in use (kWh)": ("biggest drop in use from one day to the next", "kWh"),
    "similarity of each day to the day before (correlation)": ("how similar each day is to the day before", ""),
    "similarity of each day to the same day a week before (correlation)": ("how similar each day is to the same day a week before", ""),
    "change in use per year (% of average)": ("yearly change in use, as a % of their average", "%"),
    "average daily use in the last 90 days (kWh)": ("average daily use in the last 90 days", "kWh"),
    "use in the last 90 days compared with own average (%)": ("use in the last 90 days, as a % of their usual use", "%"),
}


def short_name(feature):
    if feature in SHORT_NAMES:
        return SHORT_NAMES[feature]
    m = MONTH_RE.match(feature)
    if m:
        return f"use in {m.group(1)}, as a % of their usual use", "%"
    m = DAY_RE.match(feature)
    if m:
        return f"use on {m.group(1)}, as a % of their usual use", "%"
    return feature, ""


def fmt(value, unit):
    if unit == "kWh":
        return f"{value} kWh"
    if unit == "%":
        return f"{value}%"
    if unit == "days":
        return f"{value} days"
    return f"{value}"


def warning_signs(ev, k=3):
    """(plain name, customer value, typical value, unit) for the top theft-side features."""
    if "features_that_pushed_toward_theft" in ev:
        items = ev["features_that_pushed_toward_theft"]
    else:
        items = [c for c in ev.get("comparison_with_normal_customers", []) if c.get("unusual")]
    out = []
    for c in items[:k]:
        name, unit = short_name(c["feature"])
        out.append((name, c["customer_value"], c["typical_normal_customer_value"], unit))
    return out


def _more_less(a, b, up, down):
    if a == b:
        return "no different"
    ratio = a / b if b else float("inf")
    if a > b:
        return f"much {up}" if ratio >= 2 else up
    return f"much {down}" if ratio <= 0.5 else down


def evidence_facts(ev):
    pct = round(ev["theft_probability"] * 100)
    flagged = ev["decision"].startswith("flagged")
    facts = [("The detector flagged this customer as a possible electricity thief." if flagged
              else "The detector did not flag this customer."),
             f"The detector is {pct}% sure."]
    if ev.get("suspicious_periods"):
        p = ev["suspicious_periods"][0]
        facts.append(f"The most suspicious time was {p['start_date']} to {p['end_date']}.")
        a, b = p["average daily use in this period (kWh)"], p[OVERALL_KEY]
        facts.append(f"In that time the customer used {a} kWh a day. "
                     f"That is {_more_less(a, b, 'more', 'less')} than their usual {b} kWh a day.")
    for name, cust, typ, unit in warning_signs(ev):
        facts.append(f"Warning sign: {name} is {fmt(cust, unit)} for this customer. That is "
                     f"{_more_less(cust, typ, 'higher', 'lower')} than a typical honest customer, "
                     f"who has {fmt(typ, unit)}.")
    facts.append("These are warning signs, not proof. Someone should check the meter on site.")
    return facts


# ─────────────────────────────────────────────
# Step 5: guardrails
# ─────────────────────────────────────────────
NUM_RE = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?![\w])")
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
# Phrases from the first TinyLlama run that were invented or described the file.
BANNED = [
    (r"\bjson\b", "talks about the JSON file"),
    (r"\bsections?\b", "describes the file's sections"),
    (r"\blanguage model\b|\bthe lm\b|\bllm\b", "mentions the language model"),
    (r"\binstalled\b|\breported the theft\b|\brequested the investigation\b", "invents how the data was collected"),
    (r"\bnew york\b|\bcity\b|\bresidential\b|\bproperty\b|\baddress\b", "invents a location"),
    (r"\btrained\b|\btraining data\b", "invents training details"),
    (r"\bnext \d+ days\b|\bwill (steal|commit)\b|\bfuture\b", "makes a prediction about the future"),
    (r"\bthreshold\b", "uses a technical term (threshold)"),
    (r"\blstm\b|\blightgbm\b|\bshap\b|\blog-odds\b|\bcorrelation\b|\bcoefficient\b", "uses a technical term"),
    (r"kwh in a row|\bwent \d[\d.,]* kwh|kwh[^.]{0,30}without using", "describes kWh as a run of days"),
]
COMPARE_RE = re.compile(r"\b(more|higher|greater|larger|bigger|less|lower|smaller|fewer)\s+than\b")
MULT_RE = re.compile(r"^\s*(twice|double|(\w+|\d+(?:\.\d+)?) times)\b")
WORD_NUM = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
            "ten": 10, "twenty": 20, "fifty": 50, "hundred": 100}
HIGHER = re.compile(r"\b(higher|more|above|greater|larger|bigger|exceed\w*|increase\w*)\b")
LOWER = re.compile(r"\b(lower|less|below|smaller|fewer|decrease\w*|drop(ped)? below)\b")


def numbers_in(obj, key=None):
    """Numbers the text may use. Dates and the customer ID add none (dates are checked separately)."""
    out = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k != "customer_id":
                out |= numbers_in(v, k)
    elif isinstance(obj, list):
        for v in obj:
            out |= numbers_in(v, key)
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        out.add(float(obj))
        if isinstance(obj, float) and 0 <= obj <= 1:      # allow probability as a percent
            out.add(round(obj * 100, 1))
            out.add(float(round(obj * 100)))
    elif isinstance(obj, str) and not DATE_RE.fullmatch(obj):
        for tok in NUM_RE.findall(obj):                    # e.g. "last 90 days", "Oct 2016"
            out.add(float(tok.replace(",", "")))
    return out


def _sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def direction_problems(text, ev):
    """A sentence that quotes the customer's value must not point the wrong way."""
    pairs = []   # (customer value, comparison value)
    for _, cust, typ, _ in warning_signs(ev):
        pairs.append((cust, typ))
    if ev.get("suspicious_periods"):
        p = ev["suspicious_periods"][0]
        pairs.append((p["average daily use in this period (kWh)"], p[OVERALL_KEY]))
    probs = []
    for sent in _sentences(text):
        low = sent.lower()
        nums = {float(t.replace(",", "")) for t in NUM_RE.findall(DATE_RE.sub(" ", sent))}
        for cust, typ in pairs:
            if cust == typ or not any(abs(n - cust) <= 0.011 for n in nums):
                continue
            if not any(abs(n - typ) <= 0.011 for n in nums):
                continue          # only judge sentences that compare the two values
            says_higher, says_lower = bool(HIGHER.search(low)), bool(LOWER.search(low))
            if cust > typ and says_lower and not says_higher:
                probs.append(f"says lower, but {cust} is higher than {typ}")
            if cust < typ and says_higher and not says_lower:
                probs.append(f"says higher, but {cust} is lower than {typ}")
    return probs


def value_roles(ev):
    """Numbers that belong to this customer vs to the typical honest customer."""
    cust, typ = set(), set()
    if ev.get("suspicious_periods"):
        p = ev["suspicious_periods"][0]
        cust |= {float(p["average daily use in this period (kWh)"]), float(p[OVERALL_KEY])}
    for _, c, t, _ in warning_signs(ev, k=10):
        cust.add(float(c))
        typ.add(float(t))
    return cust, typ


def _close(v, pool):
    return any(abs(v - a) <= max(0.011, 0.005 * abs(a)) for a in pool)


def _nums(fragment):
    return [float(t.replace(",", "")) for t in NUM_RE.findall(DATE_RE.sub(" ", fragment))]


def comparison_problems(text, ev):
    """Every "X ... more/less than [N times] Y" must be true for the numbers it compares."""
    cust, typ = value_roles(ev)
    usual = None
    if ev.get("suspicious_periods"):
        usual = float(ev["suspicious_periods"][0][OVERALL_KEY])
    probs = []
    for sent in _sentences(text):
        low = sent.lower()
        for m in COMPARE_RE.finditer(low):
            before, after = low[:m.start()], low[m.end():]
            xs = _nums(before)
            if not xs:
                continue
            x = xs[-1]
            factor = 1.0
            mm = MULT_RE.match(after)
            if mm:
                w = mm.group(1)
                if w in ("twice", "double"):
                    factor = 2.0
                else:
                    k = mm.group(2)
                    factor = float(k) if re.fullmatch(r"\d+(?:\.\d+)?", k) else float(WORD_NUM.get(k, 1))
                after = after[mm.end():]
            ys = _nums(after.split(",")[0]) or _nums(after)
            if ys:
                y = ys[0]
            elif usual is not None and re.match(r"\s*(that of )?their (usual|typical|normal|average)", after):
                y = usual
            else:
                continue
            up = m.group(1) in ("more", "higher", "greater", "larger", "bigger")
            if up and not x > factor * y:
                probs.append(f"says {x} is {'%g times ' % factor if factor > 1 else ''}more than {y}")
            if not up and not x < y:
                probs.append(f"says {x} is less than {y}")
            if "slightly" in before[-25:] and y and not (0.7 <= x / y <= 1.4):
                probs.append(f"calls the gap between {x} and {y} slight")
    # "their usual/typical ... N" must be this customer's own value
    for sent in _sentences(text):
        # look only at the few words right after "their usual", up to a comma
        for m in re.finditer(r"\btheir (usual|typical|normal|average)\b((?:\s+[^\s,;]+){0,5})", sent.lower()):
            ns = _nums(m.group(2))
            if ns and _close(ns[0], typ) and not _close(ns[0], cust):
                probs.append(f"calls {ns[0]} 'their {m.group(1)}', but it is a typical honest customer's value")
    return probs


def check_text(text, evidence):
    """Return a list of problems (empty list = passes)."""
    problems = []
    ev_json = json.dumps(evidence)
    allowed = numbers_in(evidence) | {1.0, 2.0, 3.0, 2014.0, 2015.0, 2016.0}
    for date in DATE_RE.findall(text):
        if date not in ev_json:
            problems.append(f"date not in evidence: {date}")
    for tok in NUM_RE.findall(DATE_RE.sub(" ", text)):
        v = float(tok.replace(",", ""))
        if not any(abs(v - a) <= max(0.011, 0.005 * abs(a)) for a in allowed):
            problems.append(f"number not in evidence: {tok}")
    low = text.lower()
    for pattern, why in BANNED:
        if re.search(pattern, low):
            problems.append(why)
    flagged = evidence["decision"].startswith("flagged")
    if flagged and re.search(r"\b(not (a )?(theft|thief|suspicious|flagged)|no (evidence of )?theft|"
                             r"is normal|has been normal|(usage|use|behaviou?r) (was|is|has been) normal)\b", low):
        problems.append("text contradicts the detector's decision")
    if not flagged and re.search(r"\b(is|was) (committing|stealing)\b", low):
        problems.append("text contradicts the detector's decision")
    problems += direction_problems(text, evidence)
    problems += comparison_problems(text, evidence)
    # Must carry the key evidence, not just generic words.
    periods = evidence.get("suspicious_periods") or []
    if periods and periods[0]["start_date"] not in text:
        problems.append("does not mention the most suspicious period")
    signs = warning_signs(evidence)
    if signs and not any(re.search(rf"(?<![\d.]){re.escape(str(c))}(?![\d])", text) for _, c, _, _ in signs):
        problems.append("does not mention any warning sign value")
    if not re.search(r"\b(inspect\w*|check\w*|visit\w*|confirm\w*)\b", low):
        problems.append("does not recommend an on-site check")
    if not text.rstrip().endswith((".", "!", "?")):
        problems.append("stops mid-sentence")
    n_words = len(text.split())
    if n_words < 25:
        problems.append("too short")
    if n_words > 160:
        problems.append("too long")
    return problems


# ─────────────────────────────────────────────
# Step 4b: writers
# ─────────────────────────────────────────────
def template_writer(ev):
    """Fallback: plain, short sentences built only from the evidence."""
    return " ".join(evidence_facts(ev))


EXAMPLE_FACTS = [
    "The detector flagged this customer as a possible electricity thief.",
    "The detector is 97% sure.",
    "The most suspicious time was 2015-03-04 to 2015-03-31.",
    "In that time the customer used 3.2 kWh a day. That is much less than their usual 14.6 kWh a day.",
    "Warning sign: highest daily use is 96.4 kWh for this customer. That is much higher than a typical "
    "honest customer, who has 18.3 kWh.",
    "Warning sign: average daily use in the last 90 days is 9.1 kWh for this customer. That is higher than "
    "a typical honest customer, who has 6.2 kWh.",
    "These are warning signs, not proof. Someone should check the meter on site.",
]
EXAMPLE_ANSWER = (
    "This customer has been flagged as a possible electricity thief, and the system is 97% sure. "
    "From 2015-03-04 to 2015-03-31 they used 3.2 kWh a day, much less than their usual 14.6 kWh. "
    "Their highest daily use was 96.4 kWh, much higher than the 18.3 kWh of a typical honest customer. "
    "In the last 90 days they used 9.1 kWh a day, higher than the 6.2 kWh of a typical honest customer. "
    "These are warning signs, not proof, so someone should check the meter on site."
)
SYSTEM_PROMPT = (
    "You write short notes for electricity inspectors who are not technical. "
    "Rewrite the facts you are given as one short paragraph of 4 to 6 plain sentences. "
    "Rules: use only the facts given; copy every number and date exactly; copy the words "
    "'more', 'less', 'higher' and 'lower' from the facts and never change them; do not add any other "
    "information; do not calculate anything; do not explain how the detector works; do not use technical words; keep each "
    "sentence under 25 words; end by saying someone should check the meter on site."
)


class LLMWriter:
    def __init__(self, model_id, device):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"Loading {model_id} ...")
        self.tok = AutoTokenizer.from_pretrained(model_id)
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        try:   # transformers >= 4.56 calls it "dtype"; older versions use "torch_dtype"
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        self.model = model.to(device).eval()
        self.device = device
        # Set the generation options on the model itself. Newer transformers merge
        # any passed settings with the model's stored defaults, so the model's own
        # max_length and sampling options must be cleared here; otherwise every call
        # prints "Both max_new_tokens and max_length seem to have been set".
        gc = self.model.generation_config
        gc.max_length = None
        gc.max_new_tokens = 200
        gc.do_sample = False
        gc.temperature = None
        gc.top_p = None
        gc.top_k = None
        gc.repetition_penalty = 1.05
        gc.pad_token_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id

    def __call__(self, ev, strict=False):
        system = SYSTEM_PROMPT
        if strict:
            system += " Your last answer broke these rules. Keep it simpler and closer to the facts."
        facts = "\n".join(f"- {f}" for f in evidence_facts(ev))
        example = "\n".join(f"- {f}" for f in EXAMPLE_FACTS)
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": "Facts:\n" + example},
                {"role": "assistant", "content": EXAMPLE_ANSWER},
                {"role": "user", "content": "Facts:\n" + facts}]
        # return_dict=True gives the same dict (input_ids + attention_mask) in old and new
        # transformers versions; newer versions return a dict even without it.
        enc = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                           return_dict=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model.generate(**enc)
        n_prompt = enc["input_ids"].shape[1]
        return self.tok.decode(out[0, n_prompt:], skip_special_tokens=True).strip()


def trim_to_sentence(text):
    """Cut a reply that ran out of tokens back to its last full sentence."""
    text = text.strip()
    if text.endswith((".", "!", "?")):
        return text
    cut = max(text.rfind(". "), text.rfind("! "), text.rfind("? "))
    return text[:cut + 1] if cut > 0 else text


def narrate(ev, llm):
    """LLM first (2 tries), template if the guardrails reject both."""
    attempts = []
    if llm is not None:
        for strict in (False, True):
            raw = llm(ev, strict=strict)
            text = trim_to_sentence(raw)
            problems = check_text(text, ev)
            attempts.append({"text": text, "raw": raw, "problems": problems})
            if not problems:
                return text, "llm", attempts
    text = template_writer(ev)
    problems = check_text(text, ev)
    if problems:   # should never happen; fail loudly so it gets fixed
        raise RuntimeError(f"template failed guardrails: {problems}")
    return text, "template", attempts


# ─────────────────────────────────────────────
# Step 6: explanation checks
# ─────────────────────────────────────────────
def syllables(word):
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups) - (1 if w.endswith("e") and len(groups) > 1 else 0)
    return max(1, n)


def flesch(text):
    sents = max(1, len(re.findall(r"[.!?]+", text)))
    words = re.findall(r"[A-Za-z]+", text)
    if not words:
        return None
    syl = sum(syllables(w) for w in words)
    return 206.835 - 1.015 * len(words) / sents - 84.6 * syl / len(words)


def deletion_test(scorer, x, m, base_prob, starts, drops, rng):
    """Hide the top-3 windows together vs 3 random windows together (log-odds drop)."""
    logit = lambda q: float(np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6))))
    med = float(np.median(x))

    def hide(ks):
        xo, mo = x.copy(), m.copy()
        for k in ks:
            s = starts[k]
            xo[s:s + WINDOW_DAYS] = med
            mo[s:s + WINDOW_DAYS] = 0
        return float(scorer(xo[None], mo[None])[0])

    top = list(np.argsort(-drops)[:TOP_PERIODS])
    rand = list(rng.choice(len(starts), size=TOP_PERIODS, replace=False))
    return logit(base_prob) - logit(hide(top)), logit(base_prob) - logit(hide(rand))


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Explain the LSTM's theft flags in plain English")
    ap.add_argument("--run-dir", default=str(DEFAULT_RUN))
    ap.add_argument("--source", choices=["raw", "imputed"], default=None,
                    help="data the run was trained on (default: read from the run folder path)")
    ap.add_argument("--data", default=None)
    ap.add_argument("--top-n", type=int, default=20, help="explain the N most confident flagged test customers")
    ap.add_argument("--customer-id", default=None)
    ap.add_argument("--no-llm", action="store_true", help="use the template writer only")
    ap.add_argument("--llm-id", default=LLM_ID)
    ap.add_argument("--mode", choices=["auto", "ensemble", "lstm"], default="auto",
                    help="auto: use the ensemble if 14_gbm_ensemble.py was run for this LSTM run")
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    v2 = load_v2()
    run_dir = Path(args.run_dir)
    source = args.source or ("imputed" if "imputed" in run_dir.parts else "raw")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.data:
        path = Path(args.data)
    else:
        path = v2.find_file("data/data.csv", "data.csv") if source == "raw" else v2.find_file("imputed_data.csv")
    if path is None:
        raise SystemExit("[ERROR] data file not found; pass --data")
    print(f"Loading {path} ({source}) ...")
    ids, X, M, y, cols = (v2.load_raw if source == "raw" else v2.load_imputed)(path)
    dates = pd.to_datetime(cols)
    pos = {c: i for i, c in enumerate(ids)}

    splits = json.loads((run_dir / "splits.json").read_text(encoding="utf-8"))
    train_idx = [pos[c] for c in splits["train"]]
    scorer = Scorer(v2, run_dir, device)
    base = normal_baseline(X, M, y, train_idx)

    ens_dir = run_dir / "ensemble"
    use_ens = args.mode == "ensemble" or (args.mode == "auto" and (ens_dir / "ensemble_config.json").exists())
    fm = None
    if use_ens:
        if not (ens_dir / "ensemble_config.json").exists():
            raise SystemExit(f"[ERROR] no ensemble in {ens_dir}; run 14_gbm_ensemble.py first or use --mode lstm.")
        print("Mode: ensemble (LSTM + LightGBM)")
        fm = FeatureModel(ens_dir, X, M, y, cols, train_idx)
        threshold, w = float(fm.cfg["threshold"]), float(fm.cfg["lstm_weight"])
        preds = pd.read_csv(ens_dir / "test_predictions.csv", dtype={"CONS_NO": str})
    else:
        print("Mode: LSTM only")
        threshold, w = scorer.threshold, 1.0
        preds = pd.read_csv(run_dir / "test_predictions.csv", dtype={"CONS_NO": str})
    if args.customer_id:
        chosen = [args.customer_id]
    else:
        flagged = preds[preds.theft_probability >= threshold]
        chosen = flagged.sort_values("theft_probability", ascending=False).CONS_NO.head(args.top_n).tolist()
    print(f"Explaining {len(chosen)} customer(s); threshold {threshold:.3f}")

    llm = None
    if not args.no_llm:
        try:
            llm = LLMWriter(args.llm_id, device)
        except Exception as e:   # no internet / no transformers: still produce faithful output
            print(f"[WARNING] could not load {args.llm_id} ({type(e).__name__}: {e}). Using the template writer.")

    out_dir = run_dir / "explanations"
    (out_dir / "evidence").mkdir(parents=True, exist_ok=True)
    for old in (out_dir / "evidence").glob("*.json"):   # start clean so files match this run
        old.unlink()
    rng = np.random.RandomState(SEED)
    records, del_top, del_rand, fdel_top, fdel_rand = [], [], [], [], []
    for cid in chosen:
        if cid not in pos:
            print(f"  {cid}: not in the data file, skipped")
            continue
        i = pos[cid]
        x, m = X[i].astype(np.float32), M[i].astype(np.float32)
        p_lstm = float(scorer(x[None], m[None])[0])
        periods, starts, drops = occlusion_periods(scorer, x, m, p_lstm, dates)
        ens = None
        prob = p_lstm
        if fm is not None:
            p_gbm, feats, contrib = fm.score(x, m)
            toward_theft, toward_normal, order = fm.evidence(feats, contrib)
            prob = w * p_lstm + (1 - w) * p_gbm
            ens = {"lstm_probability": p_lstm, "gbm_probability": p_gbm,
                   "toward_theft": toward_theft, "toward_normal": toward_normal}
            ft, fr = feature_deletion_test(fm, feats, order, rng)
            fdel_top.append(ft)
            fdel_rand.append(fr)
        ev = build_evidence(cid, prob, threshold, readable_features(x, m), base, periods, ens)
        (out_dir / "evidence" / f"{cid}.json").write_text(json.dumps(ev, indent=2), encoding="utf-8")
        text, writer, attempts = narrate(ev, llm)
        dt, dr = deletion_test(scorer, x, m, p_lstm, starts, drops, rng)
        del_top.append(dt)
        del_rand.append(dr)
        records.append({"CONS_NO": cid, "true_label": int(y[i]), "theft_probability": ev["theft_probability"],
                        "decision": ev["decision"], "writer": writer, "explanation": text,
                        "llm_attempts": attempts, "flesch_reading_ease": flesch(text),
                        "deletion_drop_top": dt, "deletion_drop_random": dr})
        print(f"  {cid}  p={prob:.3f}  writer={writer}")

    if not records:
        raise SystemExit("Nothing explained.")
    with open(out_dir / "explanations.jsonl", "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    with open(out_dir / "explanations.md", "w", encoding="utf-8") as f:
        f.write(f"# Theft flags, explained ({'LSTM + LightGBM ensemble' if fm else 'LSTM'})\n\n")
        for rec in records:
            f.write(f"## {rec['CONS_NO']}  (p = {rec['theft_probability']}, true label {rec['true_label']}, "
                    f"written by {rec['writer']})\n\n{rec['explanation']}\n\n")

    n_llm_tried = sum(1 for r_ in records if r_["llm_attempts"])
    summary = {
        "customers_explained": len(records),
        "precision_of_explained_flags": float(np.mean([r_["true_label"] for r_ in records])),
        "llm_used": llm is not None,
        "llm_first_try_pass_rate": (float(np.mean([not r_["llm_attempts"][0]["problems"] for r_ in records
                                                   if r_["llm_attempts"]])) if n_llm_tried else None),
        "final_text_from_llm": int(sum(r_["writer"] == "llm" for r_ in records)),
        "final_text_from_template": int(sum(r_["writer"] == "template" for r_ in records)),
        "all_final_texts_pass_guardrails": True,
        "mean_flesch_reading_ease": float(np.mean([r_["flesch_reading_ease"] for r_ in records])),
        "deletion_test_mean_logodds_drop_top_periods": float(np.mean(del_top)),
        "deletion_test_mean_logodds_drop_random_periods": float(np.mean(del_rand)),
        "deletion_test_top_beats_random_frac": float(np.mean(np.array(del_top) > np.array(del_rand))),
        "mode": "ensemble" if fm else "lstm",
        "llm_problem_counts": dict(collections.Counter(
            re.sub(r":.*", "", pr) for r_ in records for att in r_["llm_attempts"] for pr in att["problems"]
        ).most_common()),
    }
    if fm is not None:
        summary["feature_deletion_mean_logodds_drop_top3"] = float(np.mean(fdel_top))
        summary["feature_deletion_mean_logodds_drop_random3"] = float(np.mean(fdel_rand))
        summary["feature_deletion_top_beats_random_frac"] = float(np.mean(np.array(fdel_top) > np.array(fdel_rand)))
    (out_dir / "explanation_eval.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + json.dumps(summary, indent=2))
    print(f"\nSaved to {out_dir}")

    if not args.no_push:
        try:
            from repo_sync import push_run
        except ImportError:
            print("[repo_sync] repo_sync.py not found, skipping GitHub upload.")
        else:
            push_run(__file__, "hybrid-explanations",
                     [out_dir / "explanation_eval.json", out_dir / "explanations.md", out_dir / "explanations.jsonl"],
                     message=f"Hybrid explanations: {len(records)} customers", allow_weights=False)


if __name__ == "__main__":
    main()
