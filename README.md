# Real Estate Market Forecasting & Investment Scenarios

[Python](https://www.python.org/) · [Automated checks](https://github.com/zrose1211/real-estate-investment-predictor/actions) · [Source files](https://github.com/zrose1211/real-estate-investment-predictor/tree/main)

A historical market-level forecasting case study and a separate property investment scenario calculator. This project is designed around business analytics, not a claim of property-specific appraisal or realized investment profits.

## Current-data version: Zillow metro home values

The [latest_zhvi.py](latest_zhvi.py) pipeline uses the latest available Zillow Home Value Index (ZHVI) data, with 12 calendar months for validation and the latest 12 for holdout. It compares ridge and gradient boosting with persistence. Download the metro-level, all-homes, smoothed seasonally adjusted ZHVI CSV from [Zillow Research](https://www.zillow.com/research/data/).

```bash
pip install -r requirements.txt
python latest_zhvi.py  # download current Zillow CSV
# or use a local snapshot:
python latest_zhvi.py path/to/Metro_zhvi.csv
```

The pipeline records the actual final observation month and exports metrics, predictions and an error comparison chart to `reports_zhvi/`. The November 25, 2026 evaluation completed successfully. ZHVI measures typical home values, not median transaction prices; the prior Texas benchmark below remains a distinct historical experiment.

## Current Zillow results (through October 2026)

| Metric | Ridge regression | Persistence |
| --- | ---: | ---: |
| Holdout MAE | **$337.03** | $891.09 |

**62.2% lower MAE** across 10,728 metro-month observations in 894 metros, November 2025–October 2026. Ridge was selected on the preceding validation period. The target is next-month ZHVI, a smoothed home-value index, not an individual transaction price. The current-vintage series may be revised; the result is not a real-time vintage backtest.

[Metrics JSON](reports_current/metrics.json) · [Evaluation run and downloadable predictions/chart](https://github.com/zrose1211/real-estate-investment-predictor/actions/runs/45696921691)

## Historical benchmark results

| Candidate | 2013 validation MAE |
|---|---:|
| Persistence (current median) | $10,243 |
| Ridge regression | $9,959 |
| Random forest | $10,157 |
| Histogram gradient boosting | $9,989 |

Ridge was selected using validation data only, then retrained on the combined training and validation sets.

| 2014–2015 held-out result | Ridge | Persistence |
|---|---:|---:|
| MAE | $10,074 | $11,517 |
| RMSE | $13,826 | $16,415 |

The selected model reduced test MAE by **12.5%**. The experiment covers 46 Texas markets and 865 held-out city-month observations after filtering. See [complete metrics](reports/metrics.json).

## Data and methodology

- Source: the historical [txhousing dataset](https://ggplot2.tidyverse.org/reference/txhousing.html), distributed in plotnine. Monthly city-level data from 2000–2015; no private company data.
- Forecast the next month's median sales price using current median, previous median, price change, sales, listings, inventory, seasonality and city.
- Exclude missing observations and nonconsecutive months. No future-month inputs.
- Split by **target date**: before 2013 (5,583 training rows), 2013 (548 validation rows), 2014 onward (865 test rows).
- Fit preprocessing on training data only; select by validation MAE and evaluate on the later test period.
- Baseline: predict that next month's median equals the current month's median.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python pipeline.py
```

The pipeline generates metrics, held-out predictions and an actual-versus-predicted scatter plot in `reports/`.

## Investment scenario calculator

```python
from pipeline import scenario
scenario(resale=300000, purchase=200000, repairs=30000, holding=10000, sale_fee=.06)
```

Illustrative inputs yield $42,000 estimated profit, 17.5% return on modeled costs, and approximately $255,319 break-even resale. These are hypothetical assumptions, not actual investment results. Include financing, taxes and other expenses in the cost inputs as appropriate.

## Limitations

City medians reflect transaction mix and are not individual property valuations. Historical data end in 2015 and may have reporting delays or revisions; the experiment assumes current-month inputs are available at prediction time. Complete-case filtering may bias coverage. A single historical holdout does not guarantee future performance. The scenario calculator requires an independently supported property resale estimate.

## Automated evaluation

The [Portfolio checks workflow](https://github.com/zrose1211/real-estate-investment-predictor/actions/workflows/checks.yml) runs the test suite on GitHub Actions. Select **Run workflow** to regenerate the evaluation outputs; download the `evaluation-results` artifact from a successful run. The refresh uses external data sources; availability and results depend on the downloaded snapshot. The workflow does not overwrite previously published historical metrics.
