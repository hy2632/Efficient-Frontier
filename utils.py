import numpy as np
import pandas as pd
import datetime
from matplotlib import rcParams
import matplotlib.pyplot as plt
from pandas_datareader import data
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint



def randomWeightGen(n):
    w = np.random.random(n)
    return w / w.sum()


def rateOfReturn(asset: np.array):
    return asset[1:] / asset[:-1] - 1


def get_assets_data(
    start_date,
    end_date,
    tickers: list,
):
    assets = []
    for ticker in tqdm(tickers):
        assets.append(
            data.get_data_yahoo(ticker, start_date,
                                end_date)["Close"].to_numpy())

    assert len(assets) == len(tickers)
    assets_daily_return = np.array([rateOfReturn(asset)
                                    for asset in assets]).squeeze()
    return assets_daily_return

def get_covariance_matrix(assets_daily_return):
    """
        assets_daily_return: (days, n_assets)
    """

    Sigma = np.cov(assets_daily_return, ddof=0)
    return Sigma


def get_imputed_rf(start_date, end_date):
    """
        Compared to stocks data, risk-free daily data usually have missing values.
        Use this method to impute.
    """

    AAPL = data.get_data_yahoo("AAPL", start_date,
                                end_date)
    rf = data.get_data_yahoo("^TNX", start_date,
                                end_date)
    missing_dates = list(set(AAPL.index) - set(rf.index))

    for missing_date in missing_dates:
        # Roll back till last date with value:
        shift = 1
        while(True):
            try:
                line = rf.loc[missing_date - datetime.timedelta(days=shift)]
            except:
                shift += 1
                continue
            break
        
        df_temp = pd.DataFrame(
        data = {
            "Date": [missing_date],
            "High": line["High"],
            "Low": line["Low"],
            "Open": line["Open"],
            "Close": line["Close"],
            "Volume": line["Volume"],
            "Adj Close": line["Adj Close"],
            }
        ).set_index("Date")
        rf = rf.append(df_temp)

    return rf.sort_index()


def MonteCarlo(n, Sigma, R, times):
    """
        Parameters:
        ---------
        n: n_assets

        Sigma: Covariance matrix of shape (n, n)

        R: Expected annual return (mean over the time period) of assets in the pool, of shape (n, )
        
        times: times of simulation

        return:
        ---------
        arr_mu: array of daily expected return of the simulated portfolio, the y coordinates

        arr_volatility: array of daily volatility (standard deviation) of the portfolio, the x coordinates
        
        arr_w: array of the weight vector "w" at each point
        
    """

    arr_volatility = []
    arr_mu = []
    arr_w = []

    for i in tqdm(range(times)):
        w = randomWeightGen(n)
        arr_w.append(w)

        arr_volatility.append(np.sqrt(np.dot(w.T, np.dot(Sigma, w))))
        arr_mu.append(np.dot(R.T, w))

    arr_volatility = np.array(arr_volatility)
    arr_mu = np.array(arr_mu)
    arr_w = np.array(arr_w).reshape(times, -1)
    return arr_mu, arr_volatility, arr_w


def analyticalSolver(n, Sigma, R, arr_mu):
    """
        Parameters:
        ---------
        n: n_assets

        Sigma: Covariance matrix of shape (n, n)

        R: Expected annual return (mean over the time period) of assets in the pool, of shape (n, )

        arr_mu: np.array of "mu" (expected daily return of the portfolio), the y coordinates
        
        return:
        ---------
        arr_volatility: array of daily volatility (standard deviation) of the portfolio, the x coordinates
        
        arr_w: array of the weight vector "w" at each point
    """

    # The matrix on the left
    mat1 = np.vstack([
        np.hstack([2 * Sigma, -np.expand_dims(R, axis=1), -np.ones((n, 1))]),
        np.hstack([R, [0], [0]]),
        np.hstack([np.ones(n), [0], [0]])
    ])

    arr_volatility = []
    arr_w = []

    for mu in tqdm(arr_mu):
        vec2 = np.array([0] * n + [mu] + [1])
        w_lambda = np.linalg.solve(mat1, vec2)
        w = w_lambda[:n]
        arr_w.append(w)
        volatility = np.sqrt(np.dot(w, np.dot(Sigma, w)))
        arr_volatility.append(volatility)

    arr_volatility = np.array([arr_volatility]).squeeze()
    arr_w = np.array(arr_w)

    return arr_volatility, arr_w


def optimizerSolver(n, Sigma, R, arr_mu):
    """
        Solving for the efficient frontier using optimizer scipy.optimize.minimize().

        Parameters:
        ---------
        n: n_assets

        Sigma: Covariance matrix of shape (n, n)

        R: Expected daily return (mean over the time period) of assets in the pool, of shape (n, )

        arr_mu: np.array of "mu" (expected daily return of the portfolio), the y coordinates
        
        return:
        ---------
        arr_volatility: array of daily volatility (standard deviation) of the portfolio, the x coordinates
        
        arr_w: array of the weight vector "w" at each point
    """
    def fun(x):
        w = np.array(x)
        return w.T.dot(Sigma.dot(w))

    arr_w = np.array([])
    arr_volatility = np.array([])

    for mu in tqdm(arr_mu):

        con1 = lambda x: x.sum() - 1
        nlc1 = NonlinearConstraint(con1, 0, 0)

        con2 = lambda x: np.dot(R, x) - mu
        nlc2 = NonlinearConstraint(con2, 0, 0)

        bounds = [(0, None)] * n

        result = minimize(
            fun=fun,
            x0=np.array([1 / n] * n),
            constraints=(nlc1, nlc2),
            bounds=bounds,
        )
        w = result.x
        arr_w = np.append(arr_w, w)
        volatility = np.sqrt(result.fun)
        arr_volatility = np.append(arr_volatility, volatility)

    arr_w = arr_w.reshape(len(arr_mu), -1)

    return arr_volatility, arr_w

def tangencySolver(n, Sigma, R, rf, arr_mu):
    """
    Solving for the tangency portfolio / CML analytically by allowing weight on risk-free asset

    $$\lambda = \frac{\mu - r_f}{(R - r_f {\bf1})^T\Sigma^{-1}(R - r_f {\bf1})}$$

    $$w = \lambda\Sigma^{-1} (R - r_f {\bf1})$$
    """

    ones = np.ones(n)
    vec1 = R - rf * ones
    Sigma_inv = np.linalg.inv(Sigma)

    arr_w = np.array([])
    arr_volatility = np.array([])

    for mu in arr_mu:
        _lambda = (mu - rf) / (vec1.T.dot(Sigma_inv).dot(vec1))
        w = _lambda * Sigma_inv.dot(vec1)
        arr_w = np.append(arr_w, w)
        volatility = np.sqrt(w.T.dot(Sigma).dot(w))
        arr_volatility = np.append(arr_volatility, volatility)
    
    arr_w = arr_w.reshape(len(arr_mu), -1)

    return arr_volatility, arr_w
    
