import numpy as np
import pandas as pd
import statsmodels.api as sm

def generate_arma_sample(ar_list, ma_list, n_samples=10000, sigma=2.0, random_state=42):
    """Generates a synthetic ARMA sample using the exact AR and MA coefficient lists."""
    np.random.seed(random_state)
    
    # We pass the lists directly to statsmodels
    y = sm.tsa.arma_generate_sample(ar=ar_list, ma=ma_list, nsample=n_samples, scale=sigma)
    return pd.Series(y, index=range(n_samples))