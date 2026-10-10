"""Leakage-controlled early-origin one-step forecasting audit.

This is an ADDITIONAL evaluation, not an early-to-late multi-step rollout.
All training targets must be observed no later than the selected cutoff.
Requires WM793b CSVs acquired with ``make data``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments'))
from run_all import event_forest_predict, evaluate  # noqa: E402


def prepare_pairs(data_dir: Path) -> pd.DataFrame:
    medians = []
    for name, symbol in [('WM793b_outer.csv','R'), ('WM793b_inhibited.csv','I'), ('WM793b_necrotic.csv','N')]:
        frame = pd.read_csv(data_dir / name)[['Day','InitialCondition','Radius']].dropna()
        series = frame.groupby(['InitialCondition','Day'])['Radius'].median().rename(symbol)
        medians.append(series)
    wide = pd.concat(medians, axis=1).dropna().reset_index().sort_values(['InitialCondition','Day'])
    records = []
    for ic, g in wide.groupby('InitialCondition'):
        a = g[['Day','R','I','N']].to_numpy(float)
        for start, end in zip(a[:-1], a[1:]):
            records.append((ic, start[0], end[0]-start[0], *start[1:], *end[1:]))
    pairs = pd.DataFrame(records, columns=['ic','day','dt','R','I','N','R2','I2','N2'])
    if pairs.empty:
        raise ValueError('No complete consecutive transitions found')
    pairs['fI'] = np.divide(pairs.I, pairs.R, out=np.zeros(len(pairs)), where=pairs.R>0)
    pairs['fN'] = np.divide(pairs.N, pairs.I, out=np.zeros(len(pairs)), where=pairs.I>0)
    pairs['fI2'] = np.divide(pairs.I2, pairs.R2, out=np.zeros(len(pairs)), where=pairs.R2>0)
    pairs['fN2'] = np.divide(pairs.N2, pairs.I2, out=np.zeros(len(pairs)), where=pairs.I2>0)
    return pairs


def split_at_cutoff(pairs: pd.DataFrame, cutoff: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_day = pairs.day + pairs.dt
    train = pairs.loc[target_day <= cutoff].copy()
    # Only one-step predictions whose origin is observed at the cutoff.
    test = pairs.loc[np.isclose(pairs.day, cutoff) & (target_day > cutoff)].copy()
    assert train.empty or (train.day + train.dt <= cutoff).all()
    assert test.empty or (test.day >= cutoff).all()
    return train, test


def audit(pairs: pd.DataFrame, cutoffs: list[float], seed: int = 7) -> tuple[pd.DataFrame,pd.DataFrame]:
    summaries, cases = [], []
    for cutoff in cutoffs:
        train, test = split_at_cutoff(pairs, cutoff)
        if len(train) < 5 or test.empty:
            summaries.append({'cutoff':cutoff,'model':'NOT_EVALUATED','n_train':len(train),'n_test':len(test),'reason':'Insufficient training transitions or no observed next-step target'})
            continue
        truth = test[['R2','I2','N2']].to_numpy()
        models = {
            'Persistence': test[['R','I','N']].to_numpy(),
            'CMC-EventForest': event_forest_predict(train,test,seed=seed),
            'Matched raw-radii forest': event_forest_predict(train,test,seed=seed,mode='raw_radii'),
        }
        for name, pred in models.items():
            summaries.append({'cutoff':cutoff,'model':name,'n_train':len(train),'n_test':len(test), **evaluate(truth,pred)})
            for j, (_, case) in enumerate(test.iterrows()):
                cases.append({'cutoff':cutoff,'model':name,'ic':int(case.ic),'origin_day':float(case.day),'target_day':float(case["day"] + case["dt"]), **{f'true_{k}':float(truth[j,i]) for i,k in enumerate('RIN')}, **{f'pred_{k}':float(pred[j,i]) for i,k in enumerate('RIN')}})
    return pd.DataFrame(summaries), pd.DataFrame(cases)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results')
    parser.add_argument('--cutoffs',type=float,nargs='+',default=[6,9,12,15,18])
    args=parser.parse_args()
    pairs=prepare_pairs(args.data_dir)
    summary,cases=audit(pairs,args.cutoffs)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    summary.to_csv(args.output_dir/'early_origin_one_step_metrics.csv',index=False)
    cases.to_csv(args.output_dir/'early_origin_one_step_cases.csv',index=False)
    print(summary.to_string(index=False))

if __name__=='__main__':
    main()
