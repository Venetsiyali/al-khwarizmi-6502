"""Statistics for the revised paper (UID 6502): SS, NSE, KGE, RMSE,
block-bootstrap CIs, Diebold-Mariano (HLN, lag h-1) and Clark-West tests."""
import numpy as np, pandas as pd
from scipy import stats

d = pd.read_csv("predictions.csv")
y, Q, C = d.observed_anomaly.values, d.observed_Q.values, d.climatology.values
n = len(y)
MODELS = ["ARIMA", "LSTM", "XGBoost", "STL_ARIMA", "STL_ARIMA_LSTM", "Proposed"]
rng = np.random.default_rng(42)

def lrv(x, L):  # long-run variance, rectangular window
    x = x - x.mean()
    g = [np.mean(x[k:] * x[:n - k]) for k in range(L + 1)]
    return g[0] + 2 * sum(g[1:])

def dm_hln(e1, e0, h):  # two-sided, e0 = benchmark errors
    dd = e0**2 - e1**2
    s = dd.mean() / np.sqrt(lrv(dd, h - 1) / n)
    t = s * np.sqrt((n + 1 - 2*h + h*(h - 1)/n) / n)
    return t, 2 * (1 - stats.t.cdf(abs(t), n - 1))

def clark_west(f, h):  # one-sided, climatology (zero anomaly) nested
    dd = y**2 - (y - f)**2 + f**2
    t = dd.mean() / np.sqrt(lrv(dd, h - 1) / n)
    return t, 1 - stats.norm.cdf(t)

def kge(o, s):
    r = np.corrcoef(o, s)[0, 1]
    return 1 - np.sqrt((r-1)**2 + (s.std()/o.std()-1)**2 + (s.mean()/o.mean()-1)**2)

B = 12
idx = [np.concatenate([np.arange(s, s + B) for s in rng.integers(0, n - B + 1, -(-n // B))])[:n]
       for _ in range(2000)]

rows = []
for h in (1, 3, 6):
    for m in MODELS:
        f = d[f"{m}_anom_h{h}"].values; e = y - f; q = C + f
        ss = 1 - np.sum(e**2) / np.sum(y**2)
        lo, hi = np.percentile([1 - np.sum(e[i]**2) / np.sum(y[i]**2) for i in idx], [2.5, 97.5])
        rows.append(dict(h=h, model=m, SS=ss, SS_lo=lo, SS_hi=hi,
                         NSE=1 - np.sum((Q-q)**2) / np.sum((Q-Q.mean())**2),
                         KGE=kge(Q, q), RMSE=np.sqrt(np.mean((Q-q)**2)),
                         p_DM_vs_clim=dm_hln(e, y, h)[1], p_CW_vs_clim=clark_west(f, h)[1]))
out = pd.DataFrame(rows).round(3)
print(out.to_string(index=False))
out.to_csv("revision_statistics.csv", index=False)
