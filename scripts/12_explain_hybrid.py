"""
Hybrid pipeline, Steps 3-6: evidence extractor + LLM narrator + guardrails
=========================================================================
The LSTM (from 11_train_lstm_detector_v2.py) has already made every decision.
This script never changes a decision or a probability. For each customer the
LSTM flagged, it:

  3. Builds an EVIDENCE JSON:
       * decision + probability, copied from the LSTM
       * "suspicious periods": 4-week windows found by occlusion. Each window is
         replaced by that customer's own typical level and the LSTM is re-run;
         the bigger the drop in theft probability, the more that window drove
         the decision.
       * how the customer's usage compares with NORMAL customers from the
         training split (percentile ranks of simple, readable features).
  4. Asks a small LLM (TinyLlama-1.1B-Chat) to write a plain-English summary
     from the evidence JSON ONLY.
  5. Guardrails on the text:
       * every number in the text must appear in the evidence JSON
       * the text must not contradict the decision (e.g. call a flagged customer "normal")
     If the LLM fails twice, a fixed template writes the summary instead, so the
     output is always faithful. Each result records which writer was used.
  6. Explanation checks (written to explanation_eval.json):
       * deletion test: does hiding the top periods lower the theft probability
         more than hiding random periods? (are the periods really what the LSTM used)
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
"""

import argparse
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
            "theft_probability_if_period_looked_normal": float(occ[k]),
            "average daily use in this period (kWh)": float(seg.mean()),
            OVERALL_KEY: float(x.mean()),
            "missing readings in this period (days)": int(m[s:s + window].sum()),
        })
    return periods, starts, drops


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


def build_evidence(cid, prob, threshold, feats, base, periods):
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
        "comparison_with_normal_customers": comparisons[:5],
    }
    return r(ev)


# ─────────────────────────────────────────────
# Step 5: guardrails
# ─────────────────────────────────────────────
NUM_RE = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")


def numbers_in(obj):
    out = set()
    if isinstance(obj, dict):
        for v in obj.values():
            out |= numbers_in(v)
    elif isinstance(obj, list):
        for v in obj:
            out |= numbers_in(v)
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        out.add(float(obj))
        if isinstance(obj, float) and 0 <= obj <= 1:      # allow probability as a percent
            out.add(round(obj * 100, 1))
            out.add(float(round(obj * 100)))
    elif isinstance(obj, str):
        for tok in NUM_RE.findall(obj):                    # dates, IDs
            for part in re.split(r"[-/]", tok.replace(",", "")):
                if part:
                    out.add(float(part))
    return out


def check_text(text, evidence):
    """Return a list of problems (empty list = passes)."""
    problems = []
    allowed = numbers_in(evidence) | {float(TOP_PERIODS), 1.0, 2.0, 3.0, 90.0, float(WINDOW_DAYS)}
    for date in re.findall(r"\d{4}-\d{2}-\d{2}", text):
        if date not in json.dumps(evidence):
            problems.append(f"date not in evidence: {date}")
    text_no_dates = re.sub(r"\d{4}-\d{2}-\d{2}", " ", text)
    for tok in NUM_RE.findall(text_no_dates):
        v = float(tok.replace(",", ""))
        if not any(abs(v - a) <= max(0.011, 0.005 * abs(a)) for a in allowed):
            problems.append(f"number not in evidence: {tok}")
    low = text.lower()
    flagged = evidence["decision"].startswith("flagged")
    if flagged and re.search(r"\b(not (a )?(theft|suspicious|flagged)|no (evidence of )?theft|is normal)\b", low):
        problems.append("text contradicts the LSTM decision")
    if not flagged and re.search(r"\b(is|was) (committing|stealing)\b", low):
        problems.append("text contradicts the LSTM decision")
    if len(text.split()) < 25:
        problems.append("too short")
    return problems


# ─────────────────────────────────────────────
# Step 4: writers
# ─────────────────────────────────────────────
def template_writer(ev):
    pct = round(ev["theft_probability"] * 100)
    lines = [f"Customer {ev['customer_id']} was {ev['decision']} by the LSTM detector, "
             f"with an estimated theft probability of {pct}%."]
    if ev["suspicious_periods"]:
        p = ev["suspicious_periods"][0]
        lines.append(
            f"The period that most influenced this decision was {p['start_date']} to {p['end_date']}, "
            f"when average use was {p['average daily use in this period (kWh)']} kWh per day compared with "
            f"this customer's usual {p[OVERALL_KEY]} kWh.")
    unusual = [c for c in ev["comparison_with_normal_customers"] if c["unusual"]][:3]
    for c in unusual:
        direction = "higher" if c["percentile_among_normal_customers"] >= 50 else "lower"
        lines.append(f"For \"{c['feature']}\" this customer has {c['customer_value']}, {direction} than most "
                     f"normal customers (typical value {c['typical_normal_customer_value']}).")
    lines.append("This is a statistical warning sign, not proof of theft; an on-site inspection should confirm it.")
    return " ".join(lines)


class LLMWriter:
    def __init__(self, model_id, device):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"Loading {model_id} ...")
        self.tok = AutoTokenizer.from_pretrained(model_id)
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
        self.device = device

    def __call__(self, ev, strict=False):
        rules = ("Use ONLY facts and numbers from the JSON. Do not invent numbers. Do not change the "
                 "decision. Write 3-5 short sentences for a utility inspector with no technical background. "
                 "Mention the most suspicious period and the most unusual comparison. End by saying an "
                 "inspection is needed to confirm.")
        if strict:
            rules += " Copy numbers exactly as written in the JSON. Do not do any arithmetic."
        msgs = [{"role": "system", "content": "You explain the output of an electricity theft detector. " + rules},
                {"role": "user", "content": "Evidence JSON:\n" + json.dumps(ev, indent=1) + "\n\nExplanation:"}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=220, do_sample=False, repetition_penalty=1.1,
                                      pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def narrate(ev, llm):
    """LLM first (2 tries), template if the guardrails reject both."""
    attempts = []
    if llm is not None:
        for strict in (False, True):
            text = llm(ev, strict=strict)
            problems = check_text(text, ev)
            attempts.append({"text": text, "problems": problems})
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

    preds = pd.read_csv(run_dir / "test_predictions.csv", dtype={"CONS_NO": str})
    if args.customer_id:
        chosen = [args.customer_id]
    else:
        flagged = preds[preds.theft_probability >= scorer.threshold]
        chosen = flagged.sort_values("theft_probability", ascending=False).CONS_NO.head(args.top_n).tolist()
    print(f"Explaining {len(chosen)} customer(s); threshold {scorer.threshold:.3f}")

    llm = None
    if not args.no_llm:
        try:
            llm = LLMWriter(args.llm_id, device)
        except Exception as e:   # no internet / no transformers: still produce faithful output
            print(f"[WARNING] could not load {args.llm_id} ({type(e).__name__}: {e}). Using the template writer.")

    out_dir = run_dir / "explanations"
    (out_dir / "evidence").mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(SEED)
    records, del_top, del_rand = [], [], []
    for cid in chosen:
        if cid not in pos:
            print(f"  {cid}: not in the data file, skipped")
            continue
        i = pos[cid]
        x, m = X[i].astype(np.float32), M[i].astype(np.float32)
        prob = float(scorer(x[None], m[None])[0])
        periods, starts, drops = occlusion_periods(scorer, x, m, prob, dates)
        ev = build_evidence(cid, prob, scorer.threshold, readable_features(x, m), base, periods)
        (out_dir / "evidence" / f"{cid}.json").write_text(json.dumps(ev, indent=2), encoding="utf-8")
        text, writer, attempts = narrate(ev, llm)
        dt, dr = deletion_test(scorer, x, m, prob, starts, drops, rng)
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
        f.write("# LSTM theft flags, explained\n\n")
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
    }
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
