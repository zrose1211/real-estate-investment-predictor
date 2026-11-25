import pandas as pd
import pytest
from pipeline import prepare,scenario

def test_scenario():
    r=scenario(300000,200000,30000,10000,.06)
    assert r['profit']==42000
    assert r['break_even_resale']==pytest.approx(240000/.94)
    assert r['roi']==pytest.approx(.175)

def test_invalid():
    with pytest.raises(ValueError):
        scenario(1,1,1,1,1)

def test_temporal_alignment():
    d=pd.DataFrame(dict(city=['A']*4,year=[2012]*4,month=[1,2,3,5],median=[100,110,120,140],sales=[1]*4,listings=[2]*4,inventory=[3]*4))
    x=prepare(d)
    assert len(x)==1
    assert x.iloc[0]['median']==110
    assert x.iloc[0].lag1==100
    assert x.iloc[0].target==120
