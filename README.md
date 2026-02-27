# ARMA Process Simulation & Statistical Diagnostics

## Project Overview
This project demonstrates the mathematical simulation of a synthetic **ARMA(2,2)** time series and the rigorous statistical testing required to validate model assumptions. It covers the transition from theoretical equations to simulation, stationarity testing, and residual diagnostics.

## Repository Structure
```text
├── notebooks/                
│   └── 01_arma_simulation_and_diagnostics.ipynb  
├── src/                        
│   ├── data_generation.py    # Generates ARMA samples based on lag coefficients
│   ├── models.py             # Stationarity (KPSS) and ARMA model fitting
│   ├── diagnostics.py        # Statistical tests (Ljung-Box, Jarque-Bera)
│   ├── ts_utils.py           # Custom ACF/PACF utilities
│   └── visualizations.py     # Time series plotting
├── README.md
├── requirements.txt            
└── .gitignore
```
# Statistical Rigor
## The project validates the simulated model through:

* KPSS Test: To verify trend stationarity.

* Ljung-Box Test: To ensure no remaining autocorrelation in residuals (White Noise check).

* Jarque-Bera Test: To test the normality of residuals.