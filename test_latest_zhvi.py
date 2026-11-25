import pandas as pd
from latest_zhvi import prepare, run

def sample():
    dates=pd.date_range('2019-01-01',periods=72,freq='MS')
    rows=[]
    for i in range(3):
        row={'RegionID':i,'RegionType':'msa'}
        row.update({d.strftime('%Y-%m-%d'):100000+i*10000+j*500 for j,d in enumerate(dates)})
        rows.append(row)
    return pd.DataFrame(rows)

def test_target_alignment():
    d=prepare(sample())
    assert (d.target_date==d.date+1).all()
    assert (d.target-d.value==500).all()

def test_refresh_outputs(tmp_path):
    source=tmp_path/'zhvi.csv'
    sample().to_csv(source,index=False)
    run(source,tmp_path/'out')
    assert (tmp_path/'out'/'metrics.json').exists()
    assert (tmp_path/'out'/'forecast_error.png').exists()
