"""
LightGBM on engineered features + CNN-LSTM ensemble (fair 50/50 raw split)
==========================================================================
Trains LightGBM on readable per-customer features, using the same customers and
the same train/val/test split as 11_train_lstm_detector_v2.py (raw source, seed 42).
Then it averages its theft probability with a finished LSTM run's probabilities.
The threshold is tuned on validation only.

Needs: py -m pip install lightgbm
Usage: python 14_gbm_ensemble.py [--lstm-run data/lstm-hybrid-v2/raw/cnnlstm_minmax_col] [--no-push]
Output: data/lstm-hybrid-v2/raw/ensemble_results.json
"""
import argparse, json, importlib.util
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
SCRIPT_DIR = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--lstm-run", default=str(SCRIPT_DIR / "data" / "lstm-hybrid-v2" / "raw" / "cnnlstm_minmax_col"))
ap.add_argument("--data", default=None)
ap.add_argument("--no-push", action="store_true")
args = ap.parse_args()
from sklearn.metrics import roc_auc_score
spec=importlib.util.spec_from_file_location("v2",SCRIPT_DIR/"11_train_lstm_detector_v2.py");v2=importlib.util.module_from_spec(spec);spec.loader.exec_module(v2)
path=Path(args.data) if args.data else v2.find_file("data/data.csv","data.csv")
ids,X,M,y,cols=v2.load_raw(path)
i_tr,i_va,i_te=v2._split(y)
dates=pd.to_datetime(cols)
def feats(x,m):
    L=np.log1p(np.clip(x,0,None)); f={}
    f["mean"]=x.mean(); f["std"]=x.std(); f["cv"]=x.std()/(x.mean()+1e-6); f["max"]=x.max()
    for q in (5,25,50,75,95): f[f"p{q}"]=np.percentile(x,q)
    z=(x<=0.01)&(m==0); f["zero_frac"]=z.mean(); f["miss"]=m.sum()
    run=b=0
    for v in z: run=run+1 if v else 0; b=max(b,run)
    f["zero_streak"]=b
    mon=pd.Series(L).groupby(dates.to_period("M")).mean().to_numpy()
    for i,v in enumerate(mon): f[f"m{i}"]=v-L.mean()
    q=np.array_split(L,12)
    for i,c in enumerate(q): f[f"q{i}"]=c.mean()-L.mean()
    d=np.diff(L); f["max_drop"]=d.min(); f["max_rise"]=d.max(); f["ac1"]=np.nan_to_num(np.corrcoef(L[:-1],L[1:])[0,1]) if L.std()>0 else 0
    f["ac7"]=np.nan_to_num(np.corrcoef(L[:-7],L[7:])[0,1]) if L.std()>0 else 0
    f["trend"]=np.polyfit(np.arange(len(L)),L,1)[0]
    wk=pd.Series(L).groupby(dates.dayofweek).mean().to_numpy(); 
    for i,v in enumerate(wk): f[f"dow{i}"]=v-L.mean()
    return f
F=pd.DataFrame([feats(X[i],M[i]) for i in range(len(y))]).fillna(0)
p={"objective":"binary","learning_rate":0.03,"num_leaves":31,"min_child_samples":20,"feature_fraction":0.7,"bagging_fraction":0.8,"bagging_freq":1,"verbose":-1,"seed":42}
m=lgb.train(p,lgb.Dataset(F.iloc[i_tr],y[i_tr]),3000,valid_sets=[lgb.Dataset(F.iloc[i_va],y[i_va])],callbacks=[lgb.early_stopping(100,verbose=False)])
pv,pt=m.predict(F.iloc[i_va]),m.predict(F.iloc[i_te])
run=args.lstm_run
lv=pd.read_csv(run+"/val_predictions.csv",dtype={"CONS_NO":str}).set_index("CONS_NO").theft_probability
lt=pd.read_csv(run+"/test_predictions.csv",dtype={"CONS_NO":str}).set_index("CONS_NO").theft_probability
lv,lt=lv.loc[ids[i_va]].to_numpy(),lt.loc[ids[i_te]].to_numpy()
def rep(name,v,t):
    thr=v2.tune_threshold(y[i_va],v); m_=v2.metrics_at(y[i_te],t,thr)
    print(f"{name:<22} valAUC={roc_auc_score(y[i_va],v):.3f}  test acc={m_['accuracy']:.3f} F1={m_['f1']:.3f} AUC={m_['auc']:.3f} MCC={m_['mcc']:.3f}")
    return m_
out={"lightgbm":rep("LightGBM features",pv,pt),"lstm":rep("CNN-LSTM (run D)",lv,lt),"ensemble":rep("Average of both",(pv+lv)/2,(pt+lt)/2)}
out_path=SCRIPT_DIR/"data"/"lstm-hybrid-v2"/"raw"/"ensemble_results.json"
json.dump(out,open(out_path,"w"),indent=1)
print("Saved", out_path)
if not args.no_push:
    try:
        from repo_sync import push_run
        push_run(__file__,"lstm-hybrid-v2-ensemble",[out_path],message=f"Ensemble: test F1={out['ensemble']['f1']:.3f} AUC={out['ensemble']['auc']:.3f}",allow_weights=False)
    except ImportError:
        print("[repo_sync] repo_sync.py not found, skipping GitHub upload.")
