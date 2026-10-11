# Nifty 200 benchmark report

Stocks requested: 200 · evaluated: 190 · skipped: 10 · errors: 0
Held-out window: last 126 sessions per stock (never seen by training or model selection).

## Horizon: 1 trading day(s)

Stocks: 190 · selected model counts: {'drift': 0, 'naive': 173, 'random_forest': 6, 'ridge': 7, 'xgboost': 4}

**Full system (the model selected for each stock), held-out window**

* Median MAPE: 1.33% (mean 1.37%)
* Median MAE improvement vs naive: 0.00%
* Where the pipeline's pick differed from naive: it beat naive on MAE in 3 stocks and lost in 14 (two-sided sign test p = 0.013; more losses than wins means the selection did not generalise)

Where a non-naive model was selected (17 stocks): median MAE vs naive -1.20%, beat naive in 3/17 (p = 0.013), directional accuracy 49.6% (n=2113).

**Each algorithm used on every stock (held-out window)**

Median MAE vs naive: negative = worse than naive. Sign test is two-sided on wins vs losses against naive; a small p with more losses than wins means the model is *significantly worse* than naive.

| Algorithm | Median MAPE % | Median MAE vs naive % | Wins vs naive | Losses vs naive | Sign-test p (two-sided) | Directional acc. |
|---|---|---|---|---|---|---|
| drift | 1.33 | -0.35 | 37 | 153 | 0.000 | 48.7% |
| naive | 1.33 | 0.00 | 0 | 0 | n/a | n/a |
| random_forest | 1.34 | -0.67 | 60 | 130 | 0.000 | 49.9% |
| ridge | 1.35 | -1.54 | 35 | 155 | 0.000 | 50.2% |
| xgboost | 1.36 | -1.73 | 44 | 146 | 0.000 | 50.1% |

Validation (walk-forward, before the held-out window) for comparison:

| Algorithm | Median MAE vs naive % | Stocks beating naive |
|---|---|---|
| drift | -0.38 | 52/190 |
| naive | 0.00 | 0/190 |
| random_forest | -1.04 | 31/190 |
| ridge | -1.76 | 23/190 |
| xgboost | -2.53 | 21/190 |

## Horizon: 5 trading day(s)

Stocks: 190 · selected model counts: {'drift': 12, 'naive': 149, 'random_forest': 12, 'ridge': 8, 'xgboost': 9}

**Full system (the model selected for each stock), held-out window**

* Median MAPE: 2.93% (mean 3.09%)
* Median MAE improvement vs naive: 0.00%
* Where the pipeline's pick differed from naive: it beat naive on MAE in 9 stocks and lost in 32 (two-sided sign test p = 0.000; more losses than wins means the selection did not generalise)

Where a non-naive model was selected (41 stocks): median MAE vs naive -2.76%, beat naive in 9/41 (p = 0.000), directional accuracy 49.7% (n=4953).

**Each algorithm used on every stock (held-out window)**

Median MAE vs naive: negative = worse than naive. Sign test is two-sided on wins vs losses against naive; a small p with more losses than wins means the model is *significantly worse* than naive.

| Algorithm | Median MAPE % | Median MAE vs naive % | Wins vs naive | Losses vs naive | Sign-test p (two-sided) | Directional acc. |
|---|---|---|---|---|---|---|
| drift | 2.93 | -1.76 | 38 | 152 | 0.000 | 47.1% |
| naive | 2.88 | 0.00 | 0 | 0 | n/a | n/a |
| random_forest | 3.01 | -3.14 | 39 | 151 | 0.000 | 49.2% |
| ridge | 3.00 | -3.10 | 41 | 149 | 0.000 | 49.4% |
| xgboost | 3.07 | -5.02 | 32 | 158 | 0.000 | 49.8% |

Validation (walk-forward, before the held-out window) for comparison:

| Algorithm | Median MAE vs naive % | Stocks beating naive |
|---|---|---|
| drift | -1.53 | 48/190 |
| naive | 0.00 | 0/190 |
| random_forest | -4.47 | 32/190 |
| ridge | -3.71 | 21/190 |
| xgboost | -7.30 | 20/190 |
