# Portfolio Optimization

Takes list of tickers, downloads price data, calculates portfolio weights based on optimization strategies and computes cumulative returns for the given dataset. Saves weights to csv file.

Uses the general-purpose optimization function (solver) in SciPy’s optimization module that finds the minimum of an objective function. 

`from scipy.optimize import minimize`
`result = minimize(fun, x0, args=(), method='SLSQP', bounds=None, constraints=None)`

Where:
- `fun`: Objective function to minimize.
- `x0`: Initial guess for the optimization.
- `args=()`: Additional arguments passed to the objective function.
- `method`: Optimization algorithm (e.g., SLSQP, L-BFGS-B, COBYLA).
- `bounds`: Bounds for variables (e.g., weight constraints in portfolios).
- `constraints`: Constraints (e.g., sum of weights = 1 for portfolio optimization)

Files:
- *config.json*: contains necessary params for code execution
- *tickers_list.csv*: contains tickers for consideration and download
- *bl_views.csv*: Black-Litterman absolute views dictionary

Usage:
<bash>
- pip install requirements.txt
- python3 portfolio_optimization.py

Improvements (IN PROGRESS)
- Supports a Risk-Free Asset (risk_free_rate). 
    - If risk_free_rate > 0, the portfolio includes cash or a treasury-like asset.
- Supports Dynamic Constraints (target_risk, leverage_limit)
    - Ensures leverage control (e.g., leverage_limit=1.5 allows up to 150% exposure).
    - Ensures risk targeting (e.g., target_risk=0.10 constrains volatility to 10%).
- Handles Long-Only & Short-Selling Portfolios
    - Long-only: All weights between [0,1].
    - Long-short: Allows negative weights when allow_short_selling=True.
- Ensures Numerical Stability
    - Regularizes cov_matrix to prevent computational errors.


## Optimization Strategies

### Equal Weights Portfolio (EWP)
- Goal: Determine equal weights across assets (Baseline)
- Approach: For a given number of assets assign equal weights to each one 

### Momentum Based Portfolio (MOM)
- Goal: Determine weights based on past performance
- Approach: Given a time window allocate weights considering performance contribution

### Volatility Portfolio (VOL)
- Goal: Allocates more weight to lower-volatility assets (inverse volatility strategy).

### Mean-Variance Optimization (MVO)
- Goal: Maximize return for a given level of risk OR minimize risk for a given level of return (Markowitz Modern Portfolio Theory).
- Approach: Uses expected returns, variances (risk), and covariances (correlations) of asset returns to determine the optimal asset allocation.

### Black-Litterman Optimization (BLO)
- Goals: combines Modern Portfolio Theory (MPT) with investor views to create a more stable and intuitive asset allocation.
- Approach: Uses market equilibrium returns as a starting point, blends in investor views and levels of confidence in a controlled way to produce more stable and diversified portfolios.

### Risk Parity Portfolio (RPP)
- Goal: Allocate assets so that each contributes an equal proportion of total portfolio risk, rather than equal capital allocation.
- Approach: Assigns higher weights to lower-volatility assets and lower weights to higher-volatility assets, ensuring that each asset's risk contribution is the same.

### Maximum Diversification Portfolio (MDP)
- Goal: Maximize the diversification ratio, which is the ratio of weighted asset volatilities to total portfolio volatility.
- Approach: Uses volatilities and the covariance matrix (not just correlation) to construct a portfolio that achieves the highest diversification benefit by spreading risk across less correlated assets.

### Minimum Variance Portfolio (MVP)
- Goal: aims to minimize overall portfolio risk (volatility) regardless of expected returns.
- Approach: Uses the covariance matrix to find the most stable allocation. It does not consider expected returns—only risk (volatility).



# ToDos
- Allocation and Rebalancing Engines
- Dash App
- Refactoring for efficiency since functions perform same computation (calculate returns, cov_matrix, etc)
- Some magic numbers still present (delta, min_periods)
- Error Handling not implemented throughout code
- Improvements (leverage_ratio, target_risk and risk_free_asset) not implemented on all functions



