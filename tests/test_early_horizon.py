import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from early_horizon_audit import split_at_cutoff, audit


def sample():
    rows=[]
    for ic in [1,2]:
        for day in [0,3,6,9,12]:
            rows.append(dict(ic=ic,day=day,dt=3,R=100+day+ic,I=60+day,N=20+day,R2=103+day+ic,I2=63+day,N2=23+day,fI=.6,fN=.3,fI2=.6,fN2=.3))
    return pd.DataFrame(rows)


def test_cutoff_never_uses_future_targets():
    train,test=split_at_cutoff(sample(),9)
    assert (train.day+train.dt<=9).all()
    assert (test.day==9).all()
    assert len(test)==2


def test_insufficient_training_is_explicit():
    summary,cases=audit(sample(),[0])
    assert summary.iloc[0]['model']=='NOT_EVALUATED'
    assert cases.empty
