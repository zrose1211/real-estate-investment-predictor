# Real Estate Market Forecasting & Investment Scenarios

A historical market-level forecasting case study and a separate property investment scenario calculator. This project is designed around business analytics, not a claim of property-specific appraisal or realized investment profits.

## Results

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

## Portfolio disclosure

This project was prepared with AI development assistance. Review and reproduce the results, understand the methodology, and make personal contributions before representing the project as independent work.
