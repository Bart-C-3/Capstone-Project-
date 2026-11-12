"""
Hybrid pipeline, Step 1 (v2): LSTM theft detector with the collapse fixed
=========================================================================
The LSTM makes every theft/normal decision. A later step uses an LLM only to
turn the LSTM's evidence into plain English.

Why v1 (10_train_lstm_detector.py) predicted "theft" for everyone
  1. Column-wise Min-Max to [-1, 1] on raw kWh. SGCC has extreme outliers (one
     meter reads ~515,000 kWh in a day), so 99% of all scaled readings end up
     between -1.00 and -0.95. The network sees an almost constant input.
  2. Checkpoints were picked by validation F1. On a 50/50 set, "everyone is a
     thief" already scores F1 = 0.667, so the collapsed model looked best.

What v2 changes (all switchable, so the ablation table can show each effect)
  --scaling log_global    log1p(kWh), then one mean/std fitted on the train rows.
                          Tames the outliers but keeps each customer's usage level.
            log_customer  log1p(kWh), then z-score each customer against their
                          OWN mean/std (level passed in separately).
                          (minmax_col = the original v1 scaling, kept as baseline)
  --select auc            pick the checkpoint by validation ROC-AUC (or mcc).
                          Neither rewards an all-theft predictor.
  --arch cnnlstm          small weekly CNN-LSTM: days are folded into weeks
                          (148 weeks x 7 days), Conv1d+BatchNorm -> BiLSTM -> pooling.
                          BatchNorm re-stretches the squashed Min-Max input, which
                          is why this model learns even with the v1 scaling.
                          (paper = the original Dense(4096)->LSTM(100)->LSTM(1024))
  --source raw            leak-free data option (see below).

Data sources
  imputed  ./imputed_data.csv from 02_preprocess.py (same as v1).
           WARNING: 02_preprocess.py drops the normal customers with the MOST
           missing days, so the kept normals have ~2 missing days on average and
           the thieves ~325. The missing-day count alone separates the classes
           (ROC-AUC ~0.96), so a model can score well by spotting imputed gaps
           instead of theft. Results on this source are optimistic.
  raw      ./data/data.csv from 01_download_data.py. Normal customers are sampled
           at RANDOM (seed 42) down to 50/50, gaps are filled by linear
           interpolation, and a 0/1 "was missing" channel is given to the model.
           Missing days stay a (weak, real) signal, but not a shortcut.

Outputs: ./data/lstm-hybrid-v2/<run name>/
  best_model.pt, model_config.json, splits.json, test_predictions.csv,
  val_predictions.csv, training_log.csv, test_results.json
  and with --ablation: ./data/lstm-hybrid-v2/ablation_results.csv

Usage (VS Code Run button or terminal; keep next to imputed_data.csv / data/)
    python 11_train_lstm_detector_v2.py                       recommended config
    python 11_train_lstm_detector_v2.py --source imputed      v1 data file
    python 11_train_lstm_detector_v2.py --ablation            full comparison table
    python 11_train_lstm_detector_v2.py --arch paper --scaling minmax_col --lr 1e-3 --select f1
                                                              = the original v1 run
    add --no-push to skip the GitHub upload, --allow-cpu if there is no GPU.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_ROOT = SCRIPT_DIR / "data" / "lstm-hybrid-v2"
CONS_COL, LABEL_COL = "CONS_NO", "FLAG"
COUNT_COLS = {"n_rf_imputed", "n_prophet_imputed"}

SEED = 42
BATCH_SIZE = 64
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0
PUSH_MODEL_WEIGHTS = False   # the repo is public; weights stay local
# --protocol paper: True feeds the model the paper's Min-Max[-1,1] values exactly
# (unstable for the CNN-LSTM); False maps the same rows back to kWh and uses log scaling.
PAPER_KEEP_MINMAX = False

# Recommended defaults (the v1 originals are in the ablation list below)
# Chosen by VALIDATION AUC in the ablation (cnnlstm + minmax_col: val AUC 0.818).
DEFAULTS = dict(source="raw", arch="cnnlstm", scaling="minmax_col",
                lr=1e-3, select="auc", epochs=40, patience=8)

# Each row changes one thing at a time, starting from the v1 original.
ABLATION = [
    ("A_v1_original",        dict(arch="paper",   scaling="minmax_col",   lr=1e-3, select="f1",  epochs=20, patience=5)),
    ("B_paper_logscale",     dict(arch="paper",   scaling="log_customer", lr=1e-3, select="auc", epochs=20, patience=5)),
    ("C_paper_logscale_lr4", dict(arch="paper",   scaling="log_customer", lr=1e-4, select="auc", epochs=20, patience=5)),
    ("D_cnnlstm_minmax",     dict(arch="cnnlstm", scaling="minmax_col",   lr=1e-3, select="auc", epochs=40, patience=8)),
    ("E_cnnlstm_logscale",   dict(arch="cnnlstm", scaling="log_customer", lr=1e-3, select="auc", epochs=40, patience=8)),
    ("F_cnnlstm_logglobal",  dict(arch="cnnlstm", scaling="log_global",   lr=1e-3, select="auc", epochs=40, patience=8)),
    ("G_paper_logglobal_lr4", dict(arch="paper",  scaling="log_global",   lr=1e-4, select="auc", epochs=20, patience=5)),
]


# ─────────────────────────────────────────────
# Device / files
# ─────────────────────────────────────────────
def get_device(allow_cpu):
    print(f"Python : {sys.executable}")
    print(f"PyTorch: {torch.__version__}  (CUDA build: {torch.version.cuda})")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    if allow_cpu:
        print("[WARNING] No GPU found, training on CPU (the 'paper' model is slow on CPU).")
        return torch.device("cpu")
    raise SystemExit(
        "[ERROR] PyTorch cannot see your NVIDIA GPU. Reinstall the CUDA build in this venv:\n"
        "    py -m pip uninstall torch -y\n"
        "    py -m pip install torch --index-url https://download.pytorch.org/whl/cu124\n"
        "or run with --allow-cpu.")


def find_file(*names):
    for base in (SCRIPT_DIR, Path.cwd()):
        for name in names:
            p = base / name
            if p.exists():
                return p
    return None


def sorted_date_cols(columns):
    cand = [c for c in columns if c not in {CONS_COL, LABEL_COL} | COUNT_COLS]
    parsed = pd.to_datetime(cand, errors="coerce")
    return [c for p, c in sorted((p, c) for p, c in zip(parsed, cand) if pd.notna(p))]


# ─────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────
def load_imputed(path):
    df = pd.read_csv(path)
    cols = sorted_date_cols(df.columns)
    X = np.nan_to_num(df[cols].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mask = np.zeros_like(X)            # imputed file has no per-day missing info
    return df[CONS_COL].astype(str).to_numpy(), X, mask, df[LABEL_COL].to_numpy(np.int64), cols


def load_raw(path):
    df = pd.read_csv(path).drop_duplicates()
    df = df.drop_duplicates(subset=[CONS_COL]).reset_index(drop=True)
    rng = np.random.RandomState(SEED)
    theft = np.flatnonzero(df[LABEL_COL].to_numpy() == 1)
    normal = np.flatnonzero(df[LABEL_COL].to_numpy() == 0)
    keep = np.sort(np.concatenate([theft, rng.choice(normal, size=len(theft), replace=False)]))
    df = df.iloc[keep].reset_index(drop=True)

    cols = sorted_date_cols(df.columns)
    V = df[cols].to_numpy(np.float64)
    V[V < 0] = np.nan
    mask = np.isnan(V).astype(np.float32)
    filled = pd.DataFrame(V.T).interpolate(limit_direction="both").T.fillna(0.0)
    X = filled.to_numpy(np.float32)
    return df[CONS_COL].astype(str).to_numpy(), X, mask, df[LABEL_COL].to_numpy(np.int64), cols


# ─────────────────────────────────────────────
# Evaluation protocols
# ─────────────────────────────────────────────
def _read_raw_full(path):
    df = pd.read_csv(path).drop_duplicates()
    df = df.drop_duplicates(subset=[CONS_COL]).reset_index(drop=True)
    cols = sorted_date_cols(df.columns)
    V = df[cols].to_numpy(np.float64)
    V[V < 0] = np.nan
    return df[CONS_COL].astype(str).to_numpy(), V, df[LABEL_COL].to_numpy(np.int64), cols


def _split(y):
    idx = np.arange(len(y))
    i_trv, i_te = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
    i_tr, i_va = train_test_split(i_trv, test_size=0.2, stratify=y[i_trv], random_state=SEED)
    return i_tr, i_va, i_te


def build_paper_protocol(path):
    """Elshennawy et al. 2025, Algorithm 1, step by step:
    1. fill missing values with the (column) mode
    2. Min-Max scale to [-1, 1] using ALL customers
    3. oversample theft to 50/50 (paper: LoRAS; here: SMOTE, which also makes
       synthetic thieves as convex combinations of neighbouring real thieves)
    4. THEN split 80/20 train/test and 80/20 train/val.
    Step 3 before step 4 puts synthetic thieves, built from training thieves,
    into the test set. Rows whose ID starts with 'SYN_' are synthetic."""
    from imblearn.over_sampling import SMOTE
    ids, V, y, cols = _read_raw_full(path)
    print("  paper protocol: mode fill -> Min-Max[-1,1] on all rows -> SMOTE on all rows -> split")
    modes = pd.DataFrame(V).mode(axis=0, dropna=True).iloc[0].to_numpy()
    modes = np.where(np.isnan(modes), 0.0, modes)
    V = np.where(np.isnan(V), modes[None, :], V)
    lo, hi = V.min(0), V.max(0)
    rng = np.where(hi - lo == 0, 1.0, hi - lo)
    X = (2.0 * (V - lo) / rng - 1.0).astype(np.float32)
    Xr, yr = SMOTE(random_state=SEED).fit_resample(X, y)
    n_syn = len(yr) - len(y)
    ids = np.concatenate([ids, np.array([f"SYN_{i}" for i in range(n_syn)])])
    mask = np.zeros_like(Xr, dtype=np.float32)
    if PAPER_KEEP_MINMAX:
        return ids, Xr.astype(np.float32), mask, yr.astype(np.int64), cols, _split(yr), "none"
    # Same synthetic customers, mapped back to kWh so the model gets log scaling.
    # Min-Max over all rows squashes ~99% of SGCC values to about -1, which made
    # training unstable (BatchNorm statistics break); see README.
    Xk = ((Xr + 1.0) / 2.0 * rng + lo).astype(np.float32)
    return ids, Xk, mask, yr.astype(np.int64), cols, _split(yr), "log_global"


def build_full_protocol(path):
    """Leak-free, all 42,372 customers: split the REAL customers first, then
    SMOTE the training rows only. Validation and test stay real and imbalanced
    (~8.5% theft), like a real utility. Rows whose ID starts with 'SYN_' are synthetic."""
    from imblearn.over_sampling import SMOTE
    ids, V, y, cols = _read_raw_full(path)
    print("  full protocol: all customers, split first, SMOTE on training rows only")
    mask = np.isnan(V).astype(np.float32)
    X = pd.DataFrame(V.T).interpolate(limit_direction="both").T.fillna(0.0).to_numpy(np.float32)
    i_tr, i_va, i_te = _split(y)
    Z = np.concatenate([X[i_tr], mask[i_tr]], axis=1)
    Zr, yr = SMOTE(random_state=SEED).fit_resample(Z, y[i_tr])
    syn = Zr[len(i_tr):]
    n_syn = len(syn)
    T = X.shape[1]
    X = np.concatenate([X, syn[:, :T].astype(np.float32)])
    mask = np.concatenate([mask, (syn[:, T:] > 0.5).astype(np.float32)])
    y = np.concatenate([y, yr[len(i_tr):]]).astype(np.int64)
    ids = np.concatenate([ids, np.array([f"SYN_{i}" for i in range(n_syn)])])
    i_tr = np.concatenate([i_tr, np.arange(len(ids) - n_syn, len(ids))])
    return ids, X, mask, y, cols, (i_tr, i_va, i_te), None


def subset_report(run_dir, protocol):
    """Score the saved test predictions on the subsets that matter."""
    tp = pd.read_csv(run_dir / "test_predictions.csv", dtype={"CONS_NO": str})
    thr = float(json.loads((run_dir / "model_config.json").read_text())["threshold"])
    syn = tp.CONS_NO.str.startswith("SYN_").to_numpy()
    yt, pt = tp.true_label.to_numpy(), tp.theft_probability.to_numpy()
    out = {"n_test": int(len(tp)), "n_test_synthetic": int(syn.sum())}

    def add(name, keep):
        m = metrics_at(yt[keep], pt[keep], thr)
        m["n"] = int(keep.sum())
        m["n_theft"] = int(yt[keep].sum())
        out[name] = m

    add("all_test_rows", np.ones(len(tp), bool))
    real = ~syn
    add("real_customers_only", real)
    rng = np.random.RandomState(SEED)
    th = np.flatnonzero(real & (yt == 1))
    no = np.flatnonzero(real & (yt == 0))
    bal = np.zeros(len(tp), bool)
    bal[th] = True
    bal[rng.choice(no, size=min(len(th), len(no)), replace=False)] = True
    add("real_customers_balanced_50_50", bal)
    if protocol == "paper":
        add("synthetic_thieves_plus_real_normals", syn | (yt == 0))
    (run_dir / "subset_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nTest subsets (threshold {thr:.3f} from validation):")
    print(f"  {'subset':<38}{'n':>7}{'theft':>7}{'acc':>7}{'F1':>7}{'AUC':>7}{'MCC':>7}")
    for k, v in out.items():
        if isinstance(v, dict):
            print(f"  {k:<38}{v['n']:>7}{v['n_theft']:>7}{v['accuracy']:>7.3f}{v['f1']:>7.3f}"
                  f"{v['auc']:>7.3f}{v['mcc']:>7.3f}")
    return out


class Scaler:
    """minmax_col: v1 column-wise [-1, 1] (fit on train).
    log_global: log1p(kWh), then one mean/std from the train rows (keeps each customer's level).
    log_customer: log1p(kWh), then z-score each customer on their own mean/std."""

    def __init__(self, kind):
        self.kind = kind
        self.state = {}

    def fit(self, X):
        if self.kind == "none":
            self.state = {}
        elif self.kind == "minmax_col":
            lo, hi = X.min(0), X.max(0)
            rng = hi - lo
            rng[rng == 0] = 1.0
            self.state = {"min": lo, "range": rng}
        elif self.kind == "log_global":
            L = np.log1p(np.clip(X, 0, None))
            self.state = {"mu": float(L.mean()), "sd": float(L.std() + 1e-6)}
        else:
            lm = np.log1p(np.clip(X, 0, None)).mean(1)
            self.state = {"level_mu": float(lm.mean()), "level_sd": float(lm.std() + 1e-6)}
        return self

    def transform(self, X):
        """Return (series (N, T), customer-level scalars (N, 2))."""
        if self.kind == "none":   # data already scaled (paper protocol)
            return np.asarray(X, np.float32), np.zeros((len(X), 2), np.float32)
        if self.kind == "minmax_col":
            S = 2.0 * (X - self.state["min"]) / self.state["range"] - 1.0
            return S.astype(np.float32), np.zeros((len(X), 2), np.float32)
        if self.kind == "log_global":
            S = (np.log1p(np.clip(X, 0, None)) - self.state["mu"]) / self.state["sd"]
            return S.astype(np.float32), np.zeros((len(X), 2), np.float32)
        L = np.log1p(np.clip(X, 0, None))
        mu, sd = L.mean(1, keepdims=True), L.std(1, keepdims=True)
        S = np.clip((L - mu) / (sd + 1e-3), -10, 10)
        # per-customer z-scoring removes the overall level, so pass it back in separately
        lvl = (mu[:, 0] - self.state["level_mu"]) / self.state["level_sd"]
        spread = sd[:, 0]
        return S.astype(np.float32), np.stack([lvl, spread], 1).astype(np.float32)

    def to_json(self):
        return {"kind": self.kind,
                **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.state.items()}}


# ─────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────
class PaperLSTM(nn.Module):
    """v1 architecture: Dense(4096) -> (256x16) -> LSTM(100) -> LSTM(1024) -> Dense(100) -> 2."""

    def __init__(self, n_days, dense=4096, l1=100, l2=1024, fc=100, dropout=0.3):
        super().__init__()
        self.pre = nn.Sequential(nn.Linear(n_days, dense), nn.ReLU())
        self.steps, self.feat = dense // 16, 16
        self.lstm1 = nn.LSTM(16, l1, batch_first=True)
        self.lstm2 = nn.LSTM(l1, l2, batch_first=True)
        self.head = nn.Sequential(nn.Linear(l2, fc), nn.ReLU(), nn.Dropout(dropout), nn.Linear(fc, 2))

    def forward(self, x, mask, scalars):
        h = self.pre(x).view(x.size(0), self.steps, self.feat)
        h, _ = self.lstm1(h)
        _, (h_n, _) = self.lstm2(h)
        return self.head(h_n[-1])


class WeeklyCNNLSTM(nn.Module):
    """Days folded into weeks (148 x 7). Each week is one LSTM time step, so the
    evidence step can point at specific weeks/dates."""

    def __init__(self, n_days, use_mask=True, conv=32, hidden=64, fc=64, dropout=0.3):
        super().__init__()
        self.pad = (-n_days) % 7
        self.weeks = (n_days + self.pad) // 7
        self.use_mask = use_mask
        in_ch = 14 if use_mask else 7
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, conv, 3, padding=1), nn.BatchNorm1d(conv), nn.ReLU(),
            nn.Conv1d(conv, conv, 3, padding=1), nn.BatchNorm1d(conv), nn.ReLU())
        self.lstm = nn.LSTM(conv, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(4 * hidden + 2, fc), nn.ReLU(), nn.Dropout(dropout), nn.Linear(fc, 2))

    def forward(self, x, mask, scalars):
        B = x.size(0)
        if self.pad:   # pad at the start so the last week ends on the last real day
            x = nn.functional.pad(x, (self.pad, 0))
            mask = nn.functional.pad(mask, (self.pad, 0), value=1.0)
        h = x.view(B, self.weeks, 7)
        if self.use_mask:
            h = torch.cat([h, mask.view(B, self.weeks, 7)], dim=2)
        h = self.conv(h.transpose(1, 2)).transpose(1, 2)        # (B, weeks, conv)
        h, _ = self.lstm(h)                                      # (B, weeks, 2*hidden)
        pooled = torch.cat([h.mean(1), h.max(1).values, scalars], dim=1)
        return self.head(pooled)


def build_model(arch, n_days, use_mask):
    return PaperLSTM(n_days) if arch == "paper" else WeeklyCNNLSTM(n_days, use_mask=use_mask)


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
def metrics_at(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    div = lambda a, b: float(a / b) if b else 0.0
    return {
        "threshold": float(thr),
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": div(tn, tn + fp),
        "mcc": float(matthews_corrcoef(y, pred)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def tune_threshold(y, p):
    """Threshold with the best validation MCC (validation only, never test)."""
    grid = np.unique(np.quantile(p, np.linspace(0.02, 0.98, 97)))
    scores = [matthews_corrcoef(y, (p >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(scores))])


def selection_score(m, select):
    return {"auc": m["auc"], "mcc": m["mcc"], "f1": m["f1"]}[select]


# ─────────────────────────────────────────────
# Training one configuration
# ─────────────────────────────────────────────
def make_loader(S, M, C, y, shuffle):
    ds = TensorDataset(torch.from_numpy(S), torch.from_numpy(M), torch.from_numpy(C), torch.from_numpy(y))
    return DataLoader(ds, batch_size=BATCH_SIZE if shuffle else 256, shuffle=shuffle,
                      num_workers=0, pin_memory=False)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = []
    for xb, mb, cb, _ in loader:
        out.append(torch.softmax(model(xb.to(device), mb.to(device), cb.to(device)), 1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def train_one(name, cfg, data, device, out_dir):
    ids, X, mask, y, cols, splits = data
    i_tr, i_va, i_te = splits
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    scaler = Scaler(cfg["scaling"]).fit(X[i_tr])
    parts = {}
    for k, ii in (("train", i_tr), ("val", i_va), ("test", i_te)):
        S, C = scaler.transform(X[ii])
        parts[k] = (S, mask[ii].astype(np.float32), C, y[ii])
    loaders = {k: make_loader(*v, shuffle=(k == "train")) for k, v in parts.items()}
    eval_train = make_loader(*parts["train"], shuffle=False)

    use_mask = bool(mask.any())
    model = build_model(cfg["arch"], X.shape[1], use_mask).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    loss_fn = nn.CrossEntropyLoss()

    print(f"\n=== {name} ===  {cfg}  params={n_params:,}")
    print(f"{'Ep':>3} {'TrLoss':>7} {'TrAUC':>6} {'VaLoss':>7} {'VaAUC':>6} {'VaF1':>6} {'VaMCC':>6} {'Pred%theft':>10} {'s':>5}")
    best, best_ep, bad, log = -np.inf, 0, 0, []
    for ep in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for xb, mb, cb, yb in loaders["train"]:
            xb, mb, cb, yb = xb.to(device), mb.to(device), cb.to(device), yb.to(device)
            loss = loss_fn(model(xb, mb, cb), yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            opt.step()
            tot += loss.item() * len(yb)
            n += len(yb)
        tr_p = predict(model, eval_train, device)
        va_p = predict(model, loaders["val"], device)
        y_va = parts["val"][3]
        va_loss = float(nn.functional.binary_cross_entropy(
            torch.tensor(va_p).clamp(1e-6, 1 - 1e-6), torch.tensor(y_va, dtype=torch.float32)))
        va_m = metrics_at(y_va, va_p, 0.5)
        tr_auc = roc_auc_score(parts["train"][3], tr_p)
        score = selection_score(va_m, cfg["select"])
        sched.step(score)
        print(f"{ep:>3} {tot / n:>7.4f} {tr_auc:>6.3f} {va_loss:>7.4f} {va_m['auc']:>6.3f} "
              f"{va_m['f1']:>6.3f} {va_m['mcc']:>6.3f} {100 * (va_p >= 0.5).mean():>9.1f}% {time.time() - t0:>5.1f}")
        log.append({"epoch": ep, "train_loss": tot / n, "train_auc": tr_auc, "val_loss": va_loss,
                    "val_auc": va_m["auc"], "val_f1": va_m["f1"], "val_mcc": va_m["mcc"],
                    "val_pred_theft_frac": float((va_p >= 0.5).mean()), "lr": opt.param_groups[0]["lr"]})
        if score > best + 1e-4:
            best, best_ep, bad = score, ep, 0
            torch.save(model.state_dict(), out_dir / "best_model.pt")
        else:
            bad += 1
            if bad >= cfg["patience"]:
                print(f"Early stop (no val {cfg['select']} gain for {cfg['patience']} epochs)")
                break
    pd.DataFrame(log).to_csv(out_dir / "training_log.csv", index=False)

    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device, weights_only=True))
    va_p = predict(model, loaders["val"], device)
    te_p = predict(model, loaders["test"], device)
    thr = tune_threshold(parts["val"][3], va_p)
    res = {"default_threshold": metrics_at(parts["test"][3], te_p, 0.5),
           "tuned_threshold": metrics_at(parts["test"][3], te_p, thr),
           "val_at_tuned": metrics_at(parts["val"][3], va_p, thr),
           "best_epoch": best_ep, "selected_by": cfg["select"], "config": cfg}

    with open(out_dir / "test_results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    with open(out_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump({"arch": cfg["arch"], "n_days": int(X.shape[1]), "use_mask": use_mask,
                   "scaler": scaler.to_json(), "threshold": thr, "date_columns": cols,
                   "best_epoch": best_ep, "config": cfg}, f, indent=2)
    with open(out_dir / "splits.json", "w", encoding="utf-8") as f:
        json.dump({k: ids[ii].tolist() for k, ii in (("train", i_tr), ("val", i_va), ("test", i_te))}, f)
    for k, p in (("val", va_p), ("test", te_p)):
        ii = {"val": i_va, "test": i_te}[k]
        pd.DataFrame({"CONS_NO": ids[ii], "true_label": y[ii], "theft_probability": p,
                      "predicted_label": (p >= thr).astype(int)}).to_csv(out_dir / f"{k}_predictions.csv", index=False)

    t = res["tuned_threshold"]
    print(f"TEST  acc={t['accuracy']:.3f} f1={t['f1']:.3f} auc={t['auc']:.3f} mcc={t['mcc']:.3f} "
          f"(thr={thr:.3f} from val)  cm={t['confusion_matrix']}")
    return res


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Train the v2 LSTM theft detector")
    ap.add_argument("--source", choices=["raw", "imputed"], default=DEFAULTS["source"])
    ap.add_argument("--protocol", choices=["balanced", "paper", "full"], default="balanced",
                    help="balanced: 50/50 real customers (default). paper: reproduce the paper "
                         "(oversample BEFORE splitting, leaky). full: all customers, SMOTE on train only")
    ap.add_argument("--data", default=None, help="path to data.csv (raw) or imputed_data.csv")
    ap.add_argument("--arch", choices=["cnnlstm", "paper"], default=DEFAULTS["arch"])
    ap.add_argument("--scaling", choices=["log_global", "log_customer", "minmax_col", "none"], default=DEFAULTS["scaling"])
    ap.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    ap.add_argument("--select", choices=["auc", "mcc", "f1"], default=DEFAULTS["select"])
    ap.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    ap.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    ap.add_argument("--ablation", action="store_true", help="run every config in ABLATION")
    ap.add_argument("--only", nargs="*", default=None, help="with --ablation: run only these names")
    ap.add_argument("--max-epochs", type=int, default=None, help="cap epochs for every ablation row")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    device = get_device(args.allow_cpu)
    if args.data:
        path = Path(args.data)
    elif args.source == "raw":
        path = find_file("data/data.csv", "data.csv")
    else:
        path = find_file("imputed_data.csv")
    if path is None or not path.exists():
        raise SystemExit(f"[ERROR] data file for --source {args.source} not found "
                         f"(raw: run 01_download_data.py; imputed: run 02_preprocess.py), or pass --data.")
    print(f"\nLoading {path} (source={args.source}, protocol={args.protocol}) ...")
    if args.protocol != "balanced":
        if args.source != "raw":
            raise SystemExit("[ERROR] --protocol paper/full need --source raw (data/data.csv).")
        builder = build_paper_protocol if args.protocol == "paper" else build_full_protocol
        ids, X, mask, y, cols, (i_tr, i_va, i_te), forced_scaling = builder(path)
        if forced_scaling:
            args.scaling = forced_scaling
    else:
        ids, X, mask, y, cols = (load_raw if args.source == "raw" else load_imputed)(path)
        i_tr, i_va, i_te = _split(y)
    print(f"  rows={len(y)} (synthetic={int(sum(i.startswith('SYN_') for i in ids))})  days={len(cols)}"
          f"  theft={int(y.sum())}  normal={int((y == 0).sum())}"
          f"  missing-day channel={'yes' if mask.any() else 'no'}")
    print(f"  split train={len(i_tr)} val={len(i_va)} test={len(i_te)} (stratified, seed {SEED})")
    data = (ids, X, mask, y, cols, (i_tr, i_va, i_te))

    root = OUT_ROOT / args.source if args.protocol == "balanced" else OUT_ROOT / f"protocol_{args.protocol}"
    pushed = []
    if args.ablation:
        rows = []
        for name, cfg in ABLATION:
            if args.only and name not in args.only:
                continue
            cfg = dict(cfg)
            if args.max_epochs:
                cfg["epochs"] = min(cfg["epochs"], args.max_epochs)
            res = train_one(name, cfg, data, device, root / name)
            t = res["tuned_threshold"]
            rows.append({"run": name, "source": args.source, **{k: cfg[k] for k in ("arch", "scaling", "lr", "select")},
                         "epochs_run": cfg["epochs"], "best_epoch": res["best_epoch"],
                         "test_acc": t["accuracy"], "test_f1": t["f1"], "test_precision": t["precision"],
                         "test_recall": t["recall"], "test_specificity": t["specificity"],
                         "test_mcc": t["mcc"], "test_auc": t["auc"], "threshold": t["threshold"]})
        table = pd.DataFrame(rows)
        table_path = root / "ablation_results.csv"
        if table_path.exists() and args.only:
            old = pd.read_csv(table_path)
            table = pd.concat([old[~old.run.isin(table.run)], table]).sort_values("run")
        table.to_csv(table_path, index=False)
        print("\nAblation (test set, threshold tuned on validation):")
        print(table.round(3).to_string(index=False))
        pushed = [table_path] + [root / r["run"] / f for r in rows
                                 for f in ("test_results.json", "training_log.csv")]
        msg = f"LSTM v2 ablation on {args.source} data"
        run_label = f"lstm-hybrid-v2-{args.source}-ablation"
    else:
        cfg = {k: getattr(args, k) for k in ("arch", "scaling", "lr", "select", "epochs", "patience")}
        name = args.run_name or f"{cfg['arch']}_{cfg['scaling']}"
        res = train_one(name, cfg, data, device, root / name)
        run_dir = root / name
        pushed = [run_dir / f for f in ("test_results.json", "model_config.json", "training_log.csv",
                                        "test_predictions.csv", "val_predictions.csv", "splits.json")]
        if args.protocol != "balanced":
            subset_report(run_dir, args.protocol)
            pushed.append(run_dir / "subset_results.json")
        if PUSH_MODEL_WEIGHTS:
            pushed.append(run_dir / "best_model.pt")
        t = res["tuned_threshold"]
        msg = f"LSTM v2 {name} ({args.source}, {args.protocol}): test F1={t['f1']:.3f} AUC={t['auc']:.3f} MCC={t['mcc']:.3f}"
        run_label = (f"lstm-hybrid-v2-{args.source}-{name}" if args.protocol == "balanced"
                     else f"lstm-hybrid-v2-protocol-{args.protocol}-{name}")

    if not args.no_push:
        try:
            from repo_sync import push_run
        except ImportError:
            print("\n[repo_sync] repo_sync.py not found next to this script, skipping GitHub upload.")
        else:
            push_run(__file__, run_label, [p for p in pushed if Path(p).exists()],
                     message=msg, allow_weights=PUSH_MODEL_WEIGHTS)


if __name__ == "__main__":
    main()
