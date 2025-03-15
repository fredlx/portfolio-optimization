import pandas as pd
import numpy as np
import json
import yfinance as yf
import ccxt

from scipy.optimize import minimize

from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt.black_litterman import BlackLittermanModel, market_implied_prior_returns, market_implied_risk_aversion


def load_config(path):
    "Load required config file"
    with open(path) as config_file:
        data = json.load(config_file)
    return data

def load_data():
    config = load_config("config.json")
    return config


def get_tickers(filename):
    return pd.read_csv(filename)["tickers"].to_list()

def download_data(tickers, start_date=None, period=None, intervals="1d", price_col='Close', drop_last_candle=True):
    """Downloads ohlcv data from yfinance"""
    if period == None:
        data = yf.download(tickers, start=start_date, interval=intervals)
    else:
        data = yf.download(tickers, period=period, interval=intervals)
    
    data = data[price_col].dropna()
    
    if drop_last_candle:
        data = data[:-1]
    return data

def save_data(data, file_path, file_name):
    data.to_csv(file_path + file_name)

def load_csv(file_path, file_name):
    return pd.read_csv(file_path + file_name, parse_dates=True, index_col=0)

def split_data(data, start_date=None, end_date=None):
    """Splits the data into train and test sets based on start_date and end_date"""

    # Convert start_date and end_date to datetime if they are strings
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)

    # Ensure that start_date and end_date exist in the dataset index
    if start_date is not None and start_date not in data.index:
        raise ValueError(f"Start date {start_date} is not in the dataset index.")
    if end_date is not None and end_date not in data.index:
        raise ValueError(f"End date {end_date} is not in the dataset index.")

    # Ensure start_date is before end_date
    if start_date and end_date and start_date >= end_date:
        raise ValueError(f"Start date ({start_date}) must be earlier than end date ({end_date}).")

    # Split data using slicing (handles None values automatically)
    train_data = data[start_date:end_date]
    test_data = data[end_date:][1:]  # exclude end_date
    return train_data, test_data
        
def calculate_number_assets(data):
    return len(data.columns)

def calculate_returns(data):
    return data.pct_change().dropna()

def calculate_rolling_returns(returns, lookback):
    #returns = calculate_returns(data)
    return returns.rolling(lookback).mean().dropna()

def calculate_volatility(returns):
    return returns.std().values

def calculate_rolling_volatility(returns, lookback, min_periods=10):
    return returns.rolling(window=lookback, min_periods=min_periods).std()

def calculate_covariance_matrix(returns):
    cov_matrix = returns.cov().values
    cov_matrix += np.eye(len(cov_matrix)) * 1e-6  # Regularize for numerical stability
    return cov_matrix

def calculate_correlation_matrix(returns):
    return returns.corr().values

def calculate_portfolio_volatility(weights, covariance_matrix):
    return np.sqrt(weights.T @ covariance_matrix @ weights)  # matrix multiplication


def calculate_initial_guess(covariance_matrix):
    """Start with inverse-volatility weighted guess for better convergence"""
    inv_vol = 1 / np.sqrt(np.diag(covariance_matrix))
    init_guess = inv_vol / np.sum(inv_vol)
    return init_guess
    
def calculate_portfolio_returns(data, weights):
    returns = calculate_returns(data)
    portfolio_returns = returns @ weights
    return portfolio_returns
    
def portfolio_cumulative_returns(data, weights, start_date=None, end_date=None):
    """
    Computes cumulative returns for each optimization strategy
    Use start_date, end_date = None, None for entire dataset
    """
    df = data[start_date:end_date] if start_date or end_date else data.copy()  # None handled by default in Pandas slicing
    returns = calculate_returns(df)
    weights = np.array(weights) # Ensure np.array
    
    portfolio_returns = returns @ weights  # Matrix multiplication

    # Compute cumulative returns
    cumulative_returns = (1 + portfolio_returns).cumprod()
    return cumulative_returns


### Optimization Strategies ###

# 1:: Minimum Variance Portfolio (MVP): Low-risk portfolio
def minimum_variance_portfolio(data, allow_short_selling=False):
    """Computes the Minimum Variance Portfolio weights with stability improvements"""
    
    # Compute returns and covariance matrix
    returns = calculate_returns(data)
    cov_matrix = calculate_covariance_matrix(returns)
    num_assets = calculate_number_assets(data)
    
    # Use inverse volatility for better initial guess
    init_guess = calculate_initial_guess(cov_matrix)

    # Set constraints: weights must sum to 1
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

    # Set bounds for weights (long-only or allowing short-selling)
    bounds = [(-1, 1) if allow_short_selling else (0, 1) for _ in range(num_assets)]

    # Define objective function: portfolio volatility
    def portfolio_volatility(weights):
        return np.sqrt(weights.T @ cov_matrix @ weights)

    # Optimize for minimum variance
    result = minimize(
        portfolio_volatility, 
        init_guess, 
        bounds=bounds, 
        constraints=constraints
        )

    return result.x


# 2:: Risk Parity Portfolio (RPP): Equalized risk contributions
def risk_parity_portfolio(data, allow_short_selling=False):
    """Computes Risk Parity Portfolio weights given asset price data."""

    def risk_contributions(weights, cov_matrix):
        """Calculate risk contributions of each asset to total portfolio risk."""
        #portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights)  # not needed
        marginal_risk = cov_matrix @ weights  # Marginal risk of each asset
        risk_contributions = weights * marginal_risk  # Risk contributions of each asset
        return risk_contributions

    def risk_parity_objective(weights, cov_matrix):
        """Objective function: Minimize differences in risk contributions."""
        risk_contribs = risk_contributions(weights, cov_matrix)
        return np.sum((risk_contribs - risk_contribs.mean()) ** 2)  # Minimize variance of risk contributions

    # Compute asset returns and covariance matrix
    returns = calculate_returns(data)
    cov_matrix = calculate_covariance_matrix(returns)
    num_assets = calculate_number_assets(data)
    
    # Start with inverse-volatility weighted guess for better convergence
    init_guess = calculate_initial_guess(cov_matrix)

    # Define bounds
    bounds = [(-1, 1) if allow_short_selling else (0, 1) for _ in range(num_assets)]

    # Ensure portfolio weights sum to 1
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

    # Solve optimization problem
    result = minimize(
        fun=risk_parity_objective,
        x0=init_guess,
        args=(cov_matrix,),
        method='SLSQP',
        constraints=constraints,
        bounds=bounds
    )
    
    if not result.success:
        raise ValueError("Optimization failed. Try adjusting constraints or data.")

    return result.x  # Optimal weights


# 3:: Maximum Diversification Portfolio (MDP): Diversified across uncorrelated assets
def max_diversification_portfolio(data, allow_short_selling=False, leverage_limit=None, target_risk=None):
    """
    Computes the Maximum Diversification Portfolio (MDP) weights with optional leverage and risk constraints.
    
    Parameters:
        - data (pd.DataFrame): Asset price data (columns are assets, rows are time).
        - allow_short_selling (bool): If True, allows short selling (weights can be negative).
        - leverage_limit (float or None): If set, ensures that the sum of absolute weights does not exceed this limit.
        - target_risk (float or None): If set, constrains portfolio volatility to this value.
    
    Returns:
        - NumPy array of optimal portfolio weights.
    """
    
    # Compute returns and covariance matrix
    returns = calculate_returns(data)
    volatilities = calculate_volatility(returns)
    cov_matrix = calculate_covariance_matrix(returns)
    num_assets = calculate_number_assets(data)

    # Use inverse-volatility weights for a better initial guess    
    init_guess = calculate_initial_guess(cov_matrix)

    # Define bounds: (-1,1) for long-short, (0,1) for long-only
    bounds = [(-1, 1) if allow_short_selling else (0, 1) for _ in range(num_assets)]

    # Ensure portfolio weights sum to 1
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    
    # Add leverage constraint if specified
    if leverage_limit is not None:
        constraints.append({'type': 'ineq', 'fun': lambda w: leverage_limit - np.sum(np.abs(w))})

    # Add target risk constraint if specified
    if target_risk is not None:
        constraints.append({'type': 'eq', 'fun': lambda w: np.sqrt(w.T @ cov_matrix @ w) - target_risk})

    def diversification_ratio(weights, *args):
        """Objective function: Maximize Diversification Ratio (minimize negative ratio)."""
        volatilities, cov_matrix = args
        portfolio_vol = calculate_portfolio_volatility(weights, cov_matrix)
        weighted_vols = weights @ volatilities
        return -weighted_vols / portfolio_vol  # Negative for maximization

    # Solve optimization problem
    result = minimize(
        fun=diversification_ratio, 
        x0=init_guess, 
        args=(volatilities, cov_matrix),  
        bounds=bounds, 
        constraints=constraints,
        method='SLSQP'
        )
    
    if not result.success:
        raise ValueError("Optimization failed. Try adjusting constraints or data.")

    return result.x  # Optimal weights

### 4. MVO: Mean-Variance Optimal Portfolio
# Find the highest return for a given level of risk (Uses Markowitz Modern Portfolio Theory)
def mean_variance_optimization(data, target_return=0.002, allow_short_selling=False):
    """
    Computes the Mean-Variance Optimal Portfolio for a given target return (daily).
    If target_return is outside [min, max] it will raise an error
    """
    
    # Compute returns, mean returns, and covariance matrix
    returns = calculate_returns(data)
    cov_matrix = calculate_covariance_matrix(returns)
    num_assets = calculate_number_assets(data)

    # Check mean return range to avoid optimization error
    mean_returns = returns.mean().values.reshape(-1, 1)  # Reshape to column vector
    min_return = np.min(mean_returns)
    max_return = np.max(mean_returns)
    if (target_return < min_return) or (target_return > max_return):
        raise ValueError(f"Target daily return {target_return:.2%} is outside the achievable range [{min_return:.2%}, {max_return:.2%}].")
    
    # Use inverse-volatility weights for a better initial guess
    #inv_vol = 1 / np.sqrt(np.diag(cov_matrix))
    #init_guess = inv_vol / np.sum(inv_vol)
    init_guess = calculate_initial_guess(cov_matrix)

    # Set constraints: Weights must sum to 1, and expected return must match target return
    deviation_allowed = 0.01 # 0.01 for 1%
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Fully invested
        {'type': 'ineq', 'fun': lambda w: w @ mean_returns - target_return + deviation_allowed}  # Target return constraint, allow small deviation 1% ineq not eq
    ]

    # Define bounds: Allow short selling or enforce long-only constraints
    bounds = [(-1, 1) if allow_short_selling else (0, 1) for _ in range(num_assets)]

    # Define objective function: Minimize portfolio volatility (risk)
    def portfolio_volatility(weights):
        return np.sqrt(weights.T @ cov_matrix @ weights)

    # Solve optimization problem
    result = minimize(portfolio_volatility, init_guess, bounds=bounds, constraints=constraints)

    if not result.success:
        raise ValueError("Optimization failed. Try adjusting the target return or check data validity.")

    return result.x  # Optimal portfolio weights


### 5:: Black-Litterman Optimization (Investors views)
def get_market_caps_crypto(tickers):
    """"Returns market caps for given list of crypto tickers. Source binance"""
    exchange = ccxt.binance()  # Use Binance exchange
    binance_tickers = [x.split("-")[0] + "/USDT" for x in tickers]

    market_caps = {}
    for ticker in binance_tickers:
        symbol = ticker.replace("/", "")  # Convert format to "BTCUSDT"
        ticker_info = exchange.fetch_ticker(symbol)

        price = ticker_info.get("last", None)  # Use .get() to avoid KeyError
        volume = ticker_info.get("quoteVolume", None)

        if price is not None and volume is not None:
            market_caps[ticker.replace("/", "-")[:-1]] = price * volume
        else:
            print(f"Skipping {ticker}: Missing data from Binance")
            
    return market_caps  # dict

def get_views():
    # #viewdict = {"BTC-USD": 0.20, "ETH-USD": -0.30, "XRP-USD": 0.30}  # "BTC will rise by 20%"
    return pd.read_csv("bl_views.csv", index_col="tickers").to_dict()["views"]

def black_litterman_optimization_default(data, viewdict, allow_short_selling=False):
    """Runs BL basic model with absolute views, prior set to equal and omega default"""
    S = risk_models.sample_cov(data)
    S += np.eye(len(S)) * 1e-6
    bl = BlackLittermanModel(S, pi="equal", absolute_views=viewdict, omega="default")
    rets = bl.bl_returns()

    if allow_short_selling:
        ef = EfficientFrontier(rets, S, weight_bounds=(-1, 1))  
    else:
        ef = EfficientFrontier(rets, S)
    bl_weights = ef.max_sharpe()
    
    return np.array(list(bl_weights.values()))

def black_litterman_optimization(data, viewdict, market_caps=None, market_prices=None, allow_short_selling=False):
    """Runs BL model with absolute views, risk-aversion (delta) from market_prices, market caps for prior"""
    
    # Compute covariance matrix
    S = risk_models.sample_cov(data)
    S += np.eye(len(S)) * 1e-6  # Add a small diagonal value for numerical stability
    
    # Compute market-implied risk aversion (delta) if market prices are available
    if market_prices is not None:
        market_prices_series = pd.Series(market_prices)
        delta = market_implied_risk_aversion(market_prices_series)
        delta = 2.5 if delta <= 0 else delta
    else:
        delta = 2.5  # Default risk aversion coefficient if no prices are provided: 2.5
        
    # Compute market-implied prior returns
    if market_caps is not None:
        prior = market_implied_prior_returns(market_caps, delta, S)
    else:
        prior = "equal"  # Default to equal-weighted priors if no market caps provided
    
    # Run Black-Litterman model
    bl = BlackLittermanModel(
        S, 
        pi=prior,
        absolute_views=viewdict, 
        omega="default"   # uncertainty level = "default", proportional to the covariance of the assets
        )
    rets = bl.bl_returns()

    if allow_short_selling:
        ef = EfficientFrontier(rets, S, weight_bounds=(-1, 1))  
    else:
        ef = EfficientFrontier(rets, S)
    
    bl_weights = ef.max_sharpe()  # Find maximum Sharpe ratio portfolio, ordered dict
    return np.array([bl_weights[key] for key in sorted(bl_weights)])



### Volatility Portfolio
#This ensures that the portfolio is risk-balanced. it allocates more capital to less risky assets.

def volatility_portfolio(data, lookback=30):
    """Rolling volatility past x days"""
    returns = calculate_returns(data)
    volatilities = calculate_rolling_volatility(returns, lookback, min_periods=10)
    weights = 1 / volatilities                          # Inverse volatility weighting
    weights = weights.div(weights.sum(axis=1), axis=0)  # Normalize weights to sum to 1
    return weights.iloc[-1].to_numpy()

### Equal Weights Portfolio

def equal_weighted_portfolio(data):
    """Computes an equal-weighted portfolio."""
    num_assets = calculate_number_assets(data)
    return np.ones(num_assets) / num_assets


### Momentum 

# Momentum for top x assets with equal weights
"""def momentum_portfolio(data, lookback=30, top_n=3):
    #Computes a momentum-based portfolio using past returns.
    returns = calculate_returns(data)
    past_returns = returns.rolling(lookback).mean().dropna()
    
    top_assets = past_returns.iloc[-1].nlargest(top_n).index
    weights = np.where(data.columns.isin(top_assets), 1 / top_n, 0)
    return weights"""

# Customized Momentum that allocates weights according to rolling mean returns
def momentum_portfolio(data, lookback=30):
    """Computes a momentum-based portfolio according to rolling mean returns."""
    returns = calculate_returns(data)
    rolling_returns = returns.rolling(window=lookback).mean().dropna()
    
    # if objective is to assign weights according to past performance
    last_returns = rolling_returns.iloc[-1]
    weights = last_returns / np.sum(last_returns)  # np.sum to ensure sum is exactly 1
    
    if np.sum(weights) != 1.0:
        weights.iloc[-1] = 1 - np.sum(weights[:-1])
    return weights.to_numpy()


def main():
    
    # Get config data
    production = eval(load_data()['production'])
    file_path = load_data()['file_path']
    file_name = load_data()['file_name']
    start_date = None if load_data()['start_date'] == "None" else load_data()['start_date']
    end_date = None if load_data()['end_date'] == "None" else load_data()['end_date']
    shorts = eval(load_data()['allow_short_selling'])
    lookback = load_data()['lookback']
    leverage_limit=load_data()['leverage_limit']
    target_risk=load_data()['target_risk']
    target_return = load_data()['target_return']
    
    # Get list of tickers from file
    tickers = get_tickers(filename="tickers_list.csv")
    
    # Get price data (download for production, load csv for development)
    if production:
        data = download_data(
            tickers, 
            start_date=None, 
            period='10y', 
            intervals="1d", 
            price_col='Close', 
            drop_last_candle=True
            )
    else:
        data = load_csv(file_path, file_name)
    
    # Split data for some basic backtesting
    train_data, test_data = split_data(data, start_date=start_date, end_date=end_date)
    
    # Calculate optimal weights
    ewp_w = equal_weighted_portfolio(data=train_data)
    mom_w = momentum_portfolio(data=train_data, lookback=lookback)
    mvp_w = minimum_variance_portfolio(data=train_data, allow_short_selling=shorts)
    rpp_w = risk_parity_portfolio(data=train_data, allow_short_selling=shorts)
    mdp_w = max_diversification_portfolio(data=train_data, allow_short_selling=shorts, leverage_limit=None, target_risk=None)
    mvo_w = mean_variance_optimization(data=train_data, target_return=target_return, allow_short_selling=shorts)
    
    viewdict = get_views()
    blo_w = black_litterman_optimization(data=train_data, viewdict=viewdict, market_caps=None, market_prices=None, allow_short_selling=shorts)
    
    # Calculate Cumulative Returns for both train and test datasets and Create df
    df_train, df_test = pd.DataFrame(), pd.DataFrame()
    cols = ["ewp", "mom","mvp", "rpp", "mdp", "mvo", "blo"]
    weights_all = [ewp_w, mom_w, mvp_w, rpp_w, mdp_w, mvo_w, blo_w]
    for w,c in zip(weights_all, cols):
        temp_train = pd.DataFrame()
        temp_train[f"{c}_rets_train"] = calculate_portfolio_returns(data=train_data, weights=w)
        temp_train[f"{c}_cum_rets_train"] = portfolio_cumulative_returns(data=train_data, weights=w)
        df_train = pd.concat([df_train, temp_train], axis=1)
        
        temp_test = pd.DataFrame()
        temp_test[f"{c}_rets_test"] = calculate_portfolio_returns(data=test_data, weights=w)
        temp_test[f"{c}_cum_rets_test"] = portfolio_cumulative_returns(data=test_data, weights=w)
        df_test = pd.concat([df_test, temp_test], axis=1)
        
    # Final df
    df = pd.DataFrame()
    df["strategy_name"] = cols
    df["weights"] = weights_all
    df["daily_mean_rets_train"] = df_train[[x for x in df_train.iloc[-1].index if "cum_rets" not in x]].mean().to_list()
    df["daily_mean_rets_test"] = df_test[[x for x in df_test.iloc[-1].index if "cum_rets" not in x]].mean().to_list()
    df["cumulative_rets_train"] = df_train[[x for x in df_train.iloc[-1].index if "cum_rets" in x]].iloc[-1].to_list()
    df["cumulative_rets_test"] = df_test[[x for x in df_test.iloc[-1].index if "cum_rets" in x]].iloc[-1].to_list()
    
    # save to csv
    df.to_csv("df_final.csv")
    df_train.to_csv("df_train.csv")
    df_test.to_csv("df_test.csv")


if __name__ == '__main__':
    
    main()
    
    