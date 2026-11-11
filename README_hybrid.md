# Hybrid electricity theft detection: LSTM decides, LLM explains

New scripts (put them in the project folder next to `repo_sync.py`, `imputed_data.csv` and `data/data.csv`):

| Script | Step | What it does |
|---|---|---|
| `11_train_lstm_detector_v2.py` | 1–2 | Trains the LSTM detector, saves model, config, splits, predictions. `--ablation` runs the comparison table. |
| `12_explain_hybrid.py` | 3–6 | For each flagged customer: evidence JSON (occlusion periods + comparison with normal customers) → TinyLlama writes the text → guardrails check every number/date → template fallback. Writes `explanation_eval.json`. |
| `13_run_hybrid_pipeline.py` | 7 | Runs 11 then 12 with one command. |

All three keep the project rules: `--no-push` flag, `repo_sync.push_run(...)` at the end, weights never pushed, Windows-safe data loaders (`num_workers=0`, `pin_memory=False`).

## Quick start (Windows, VS Code terminal, venv active)

```
py 13_run_hybrid_pipeline.py                 # train (GPU) + explain top 20 flags with TinyLlama
py 11_train_lstm_detector_v2.py --ablation   # full comparison table, incl. the paper model
py 12_explain_hybrid.py --no-llm             # explanations without downloading TinyLlama
```
`12_` needs `transformers` for TinyLlama (`py -m pip install transformers accelerate`). Without it, or offline, it falls back to the template writer and says so.

## Why v1 collapsed (verified on the real SGCC data)

1. **Scaling.** Column-wise Min-Max to [-1, 1] on raw kWh: some meters read up to ~515,000 kWh/day, so 99.3% of all scaled values are between -1.00 and -0.95 (median -0.999). The paper model (Dense → LSTM, no normalisation layer) sees a near-constant input.
2. **Checkpoint selection by F1.** On a 50/50 set, predicting "theft" for everyone already gives F1 = 0.667, the exact v1 result. v2 selects by validation AUC and tunes the threshold on validation MCC.

## Data leak in `02_preprocess.py`

`undersample()` drops the normal customers with the **most** missing days. After undersampling, normals average ~2 missing days and thieves ~325. The missing-day count alone separates the classes with ROC-AUC **0.963** (on the full raw data it is only 0.569).

- For the LLM classifier this matters a lot: `03_build_dataset.py` puts `n_prophet_imputed`, `frac_total_imputed`, etc. into the JSONL input, so TinyLlama can partly "detect theft" by counting imputed days.
- For the LSTM it mattered little in our test (AUC 0.830 on the leaky set vs 0.822 on the leak-free set). That test used linear interpolation in place of your RF/Prophet imputation; with RF/Prophet the gap could be different.
- `11_` therefore defaults to `--source raw`: it reads `data/data.csv`, picks normal customers **at random** down to 50/50, interpolates gaps, and gives the model a 0/1 "missing" channel.

## Detector results (leak-free raw data, 7,230 customers, 50/50, test = 1,446 real customers)

Threshold tuned on validation only. Single seed (42), CPU run, so differences of ±0.02 may be noise.

| Run | Model | Scaling | Val AUC | Test acc | Test F1 | Test AUC | Test MCC |
|---|---|---|---|---|---|---|---|
| D (default) | weekly CNN-LSTM | Min-Max (v1) | 0.818 | 0.735 | 0.729 | 0.822 | 0.471 |
| E | weekly CNN-LSTM | log1p + per-customer z | 0.792 | 0.709 | 0.694 | 0.780 | 0.420 |
| F | weekly CNN-LSTM | log1p + global z | 0.807 | 0.720 | 0.692 | 0.796 | 0.447 |
| D on imputed stand-in | weekly CNN-LSTM | Min-Max (v1) | 0.815 | 0.749 | 0.734 | 0.830 | 0.501 |
| A (v1 original), 1 epoch | paper Dense-LSTM | Min-Max (v1), LR 1e-3 | 0.634 | train loss 0.6935 = chance, predicts 0% theft | | | |
| G, 1 epoch only | paper Dense-LSTM | log1p + global z, LR 1e-4 | 0.652 | 0.613 | 0.649 | 0.686 | 0.232 |

For comparison, v1: acc 0.500, F1 0.667, specificity 0.0 (all-theft). TinyLlama (user runs): F1 ≈ 0.71–0.725 on the leaky JSONL features.

The weekly CNN-LSTM folds the 1,034 days into 148 weeks × 7 days → Conv1d + BatchNorm → BiLSTM(64) → mean/max pooling → 2 logits (~71k parameters, a few seconds per epoch even on CPU). BatchNorm re-stretches the squashed Min-Max input, which is why it learns with the v1 scaling too.

## Explanation results (`12_`, run D, 40 most confident flags, template writer)

- All 40 final texts pass the guardrails (every number and date is in the evidence JSON; no contradiction of the LSTM decision).
- Deletion test: hiding the 3 top periods lowers the theft log-odds by 1.90 on average vs 0.17 for 3 random periods; top beats random for 40/40 customers. So the periods named in the text are what the LSTM actually used.
- Readability: Flesch reading ease ≈ 63 ("plain English").
- TinyLlama itself was **not** tested (Hugging Face is blocked in the sandbox where this was built). The guardrail checker was tested on hand-written good and bad texts: it catches invented numbers, invented dates and contradictions.

## Not done / open

- Paper-model rows were run for 1 epoch only (~20 min/epoch on the sandbox CPU). Full 20-epoch runs need the GPU: ``py 11_train_lstm_detector_v2.py --ablation --only A_v1_original B_paper_logscale C_paper_logscale_lr4 G_paper_logglobal_lr4`.
- The paper's evaluation protocol (split vs. oversampling order) is still unchecked; its site was blocked from the sandbox. Paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12000372/
- Several seeds (e.g. 5) would give error bars for the poster.
- Customer IDs: `12_` reads `CONS_NO` straight from the data file, so the "Customer unknown" issue of the old explainers does not apply here.
