"""
Hybrid pipeline, Step 7: run everything with one command
========================================================
  1. 11_train_lstm_detector_v2.py   LSTM makes the theft decisions
  2. 12_explain_hybrid.py           evidence JSON + LLM narration + guardrails

Usage
    python 13_run_hybrid_pipeline.py                      train, then explain top 20 flags
    python 13_run_hybrid_pipeline.py --skip-train         reuse an existing run folder
    python 13_run_hybrid_pipeline.py --no-llm --no-push   fast, offline
Any other options are passed to the training script, e.g. --scaling log_global.
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def run(cmd):
    print("\n>>> " + " ".join(cmd), flush=True)
    code = subprocess.call(cmd, cwd=SCRIPT_DIR)
    if code != 0:
        raise SystemExit(f"[ERROR] step failed with exit code {code}")


def main():
    ap = argparse.ArgumentParser(description="Train the LSTM detector, then explain its flags")
    ap.add_argument("--source", choices=["raw", "imputed"], default="raw")
    ap.add_argument("--run-name", default="pipeline")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    args, train_extra = ap.parse_known_args()

    py = sys.executable
    common = (["--no-push"] if args.no_push else [])
    if not args.skip_train:
        run([py, "11_train_lstm_detector_v2.py", "--source", args.source, "--run-name", args.run_name]
            + (["--allow-cpu"] if args.allow_cpu else []) + common + train_extra)
    run_dir = SCRIPT_DIR / "data" / "lstm-hybrid-v2" / args.source / args.run_name
    run([py, "12_explain_hybrid.py", "--run-dir", str(run_dir), "--source", args.source,
         "--top-n", str(args.top_n)] + (["--no-llm"] if args.no_llm else []) + common)
    print(f"\nDone. Decisions: {run_dir / 'test_predictions.csv'}")
    print(f"      Explanations: {run_dir / 'explanations' / 'explanations.md'}")


if __name__ == "__main__":
    main()
