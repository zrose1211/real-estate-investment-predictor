"""Evaluate one-month-ahead metro ZHVI forecasts on the newest available CSV."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

URL = 'https://files.zillowstatic.com/research/public_csvs/zhvi/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv'

def prepare(raw):
    dates=[c for c in raw.columns if pd.notna(pd.to_datetime(c,format='%Y-%m-%d',errors='coerce'))]
    if not dates or 'RegionID' not in raw: raise ValueError('Expected Zillow wide-format metro ZHVI CSV')
    raw=raw.loc[raw.RegionType.astype(str).str.lower().eq('msa')] if 'RegionType' in raw else raw
    d=raw.melt(id_vars=['RegionID'],value_vars=dates,var_name='date',value_name='value')
    d['date']=pd.to_datetime(d.date).dt.to_period('M')
    d['value']=pd.to_numeric(d.value,errors='coerce')
    d=d.sort_values(['RegionID','date'])
    if d.duplicated(['RegionID','date']).any(): raise ValueError('Duplicate metro-month')
    g=d.groupby('RegionID')
    for k in (1,3,12):
        lag=g.value.shift(k).where(g.date.shift(k)==d.date-k)
        d[f'growth{k}']=d.value/lag-1
    d['target']=g.value.shift(-1).where(g.date.shift(-1)==d.date+1)
    d['target_date']=d.date+1
    d['target_growth']=d.target/d.value-1
    return d.replace([np.inf,-np.inf],np.nan).dropna(subset=['value','target','growth1','growth3','growth12','target_growth'])

def run(source,out):
    d=prepare(pd.read_csv(source))
    if d.empty: raise ValueError('No valid metro-month observations')
    end=d.target_date.max()
    test_start=end-11
    val_start=test_start-12
    train=d[d.target_date<val_start]; val=d[(d.target_date>=val_start)&(d.target_date<test_start)]; test=d[d.target_date>=test_start]
    if min(map(len,(train,val,test)))==0: raise ValueError('Insufficient chronological coverage')
    features=['growth1','growth3','growth12']
    models={'ridge':make_pipeline(StandardScaler(),Ridge(alpha=10)), 'gradient_boosting':HistGradientBoostingRegressor(max_iter=100,random_state=42)}
    scores={'persistence':float(mean_absolute_error(val.target,val.value))}
    for n,m in models.items():
        m.fit(train[features],train.target_growth)
        scores[n]=float(mean_absolute_error(val.target,val.value*(1+m.predict(val[features]))))
    winner=min(scores,key=scores.get)
    dev=pd.concat([train,val])
    pred=test.value.to_numpy() if winner=='persistence' else test.value*(1+models[winner].fit(dev[features],dev.target_growth).predict(test[features]))
    metrics=dict(source=str(source),latest_target_month=str(end),test_start=str(test_start),validation_start=str(val_start),selected=winner,validation_mae=scores,test_mae=float(mean_absolute_error(test.target,pred)),baseline_mae=float(mean_absolute_error(test.target,test.value)),test_rows=len(test),metros=int(test.RegionID.nunique()))
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    p=test[['RegionID','target_date','target','value']].copy();p['prediction']=pred;p.to_csv(out/'predictions.csv',index=False)
    import matplotlib.pyplot as plt
    monthly=p.assign(model_error=(p.target-p.prediction).abs(),baseline_error=(p.target-p.value).abs()).groupby('target_date')[['model_error','baseline_error']].mean()
    monthly.index=monthly.index.to_timestamp()
    ax=monthly.plot(figsize=(9,4),title='Monthly holdout mean absolute error')
    ax.set_ylabel('Error (USD)');ax.set_xlabel('Target month')
    ax.figure.tight_layout();ax.figure.savefig(out/'forecast_error.png',dpi=150);plt.close(ax.figure)
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('csv',nargs='?',default=URL,help='Zillow metro ZHVI CSV path or URL');p.add_argument('--out',default='reports_zhvi');a=p.parse_args();run(a.csv,a.out)
