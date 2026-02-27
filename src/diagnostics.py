import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import jarque_bera

def run_residual_diagnostics(residuals):
    """Runs Ljung-Box and Jarque-Bera tests on model residuals."""
    lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
    jb_stat, jb_p = jarque_bera(residuals)
    
    return {
        'ljung_box_p_value': lb_test['lb_pvalue'].values[0],
        'jarque_bera_p_value': jb_p
    }