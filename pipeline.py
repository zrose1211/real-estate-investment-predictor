"""Forecast next-month city median sale price; never use future inputs."""
import json
from pathlib import Path
from importlib.resources import files
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def prepare(raw):
    d = raw.copy().sort_values(['city','year','month'])
    d['period'] = pd.to_datetime(dict(year=d.year,month=d.month,day=1)).dt.to_period('M')
    if d.duplicated(['city','period']).any():
        raise ValueError('Duplicate city-month')
    g = d.groupby('city',sort=False)
    d['target'] = g['median'].shift(-1)
    d['target_period'] = g['period'].shift(-1)
    d['lag1'] = g['median'].shift(1)
    d['lag_period'] = g['period'].shift(1)
    d = d[(d.target_period == d.period+1) & (d.lag_period == d.period-1)].copy()
    d['growth'] = d['median']/d['lag1']-1
    d['season_sin'] = np.sin(2*np.pi*d.month/12)
    d['season_cos'] = np.cos(2*np.pi*d.month/12)
    return d.replace([np.inf,-np.inf],np.nan).dropna(subset=['target','median','lag1','growth','sales','listings','inventory'])


def scenario(resale, purchase, repairs, holding, sale_fee=.06):
    if min(resale,purchase,repairs,holding)<0 or not 0<=sale_fee<1:
        raise ValueError('Invalid costs or fee')
    cost=purchase+repairs+holding
    profit=resale*(1-sale_fee)-cost
    return {'profit':profit,'break_even_resale':cost/(1-sale_fee),'roi':profit/cost if cost else None}


def run():
    raw=pd.read_csv(files('plotnine').joinpath('data/txhousing.csv'))
    d=prepare(raw)
    nums=['median','lag1','growth','sales','listings','inventory','season_sin','season_cos']
    cols=['city']+nums
    train=d[d.target_period<'2013-01']; val=d[(d.target_period>='2013-01')&(d.target_period<'2014-01')]; test=d[d.target_period>='2014-01']
    assert train.target_period.max()<val.target_period.min()<test.target_period.min()
    def build(model):
        pre=ColumnTransformer([('city',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['city']),('num',StandardScaler(),nums)])
        return make_pipeline(pre,model)
    models={'ridge':Ridge(alpha=10),'random_forest':RandomForestRegressor(n_estimators=120,min_samples_leaf=8,random_state=42,n_jobs=-1),'gradient_boosting':HistGradientBoostingRegressor(max_iter=150,l2_regularization=1,random_state=42)}
    scores={'persistence':mean_absolute_error(val.target,val['median'])}
    for name, model in models.items():
        pipe=build(model).fit(train[cols],train.target)
        scores[name]=mean_absolute_error(val.target,pipe.predict(val[cols]))
    best=min(models,key=scores.get)
    dev=pd.concat([train,val]); pipe=build(models[best]).fit(dev[cols],dev.target)
    pred=pipe.predict(test[cols]); base=test['median']
    results={'rows':{'train':len(train),'validation':len(val),'test':len(test)},'cities':int(d.city.nunique()),'validation_mae':scores,'selected_ml_model':best,'test_mae':mean_absolute_error(test.target,pred),'baseline_test_mae':mean_absolute_error(test.target,base),'test_rmse':mean_squared_error(test.target,pred)**.5,'baseline_test_rmse':mean_squared_error(test.target,base)**.5}
    out=Path('reports');out.mkdir(exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(results,indent=2)+'\n')
    p=test[['city','target_period','target']].copy();p['prediction']=pred;p['baseline']=base;p.to_csv(out/'predictions.csv',index=False)
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,5));ax.scatter(test.target,pred,s=9,alpha=.25);low=min(test.target.min(),pred.min());high=max(test.target.max(),pred.max());ax.plot([low,high],[low,high],'k--');ax.set(xlabel='Actual next-month median ($)',ylabel='Predicted median ($)',title='Held-out Texas housing market forecasts');fig.tight_layout();fig.savefig(out/'forecast.png',dpi=150);plt.close(fig)
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    run()
