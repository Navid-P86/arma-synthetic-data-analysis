import statsmodels.api as sm
from statsmodels.tsa.stattools import kpss
import pandas as pd

def perform_kpss_test(series):
    """Performs the KPSS test for stationarity."""
    indices = ['Test Statistic', 'p-value', 'Lags Used', 'Critical Value (10%)', 'Critical Value (5%)', 'Critical Value (2.5%)', 'Critical Value (1%)']
    kpss_stat, p_value, lags, crit = kpss(series)
    out = pd.Series([kpss_stat, p_value, lags] + list(crit.values()), index=indices)
    return out

def fit_arma_model(series, order=(2, 2)):
    """Fits an ARMA model to the series."""
    model = sm.tsa.ARIMA(series, order=(order[0], 0, order[1]))
    results = model.fit()
    return results