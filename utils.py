import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
import pdb


def randomWeightGen(n=4):
    w = np.random.random(n)
    return w / w.sum()


def rateOfReturn(asset: np.array):
    return asset[1:] / asset[:-1] - 1


# def portfolio(w, assets):
#     """
#         Parameters:
#         ---------
#             assets.shape == (days, n_assets)
#             w.shape == (n_assets)

#         return:
#         ---------
#             portfolio.shape ==(days,)
#     """
#     return np.dot(assets, w)


def get_assets_data(
    start_date="20190101",
    end_date="20191231",
    *tickers,
):

    assets = []
    for ticker in tickers:
        assets.append(
            data.get_data_yahoo(ticker, start_date,
                                end_date)["Close"].to_numpy())
    
    assert len(assets) == len(tickers)

    # Annual Risk free rate (US 10yr treasury bond, mean over the time period)
    Rf_mean = data.get_data_yahoo("^TNX", start_date,
                                  end_date)["Adj Close"].mean() / 100
    assets_daily_return = np.array([rateOfReturn(asset)
                                    for asset in assets]).squeeze()

    return Rf_mean, assets_daily_return


def get_covariance_matrix(assets_daily_return):
    """
        assets_daily_return: (days, n_assets)
    """

    Sigma = np.cov(assets_daily_return.T, ddof=0)
    return Sigma


def MonteCarlo(n, Sigma, R, times=50000):
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
        
        Target: min $w^T \Sigma w$

        Constraints: $w_i > 0, 1^Tw=1, R^Tw=mu$



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
            x0=np.array([1/n] * n),
            constraints=(nlc1, nlc2),
            bounds=bounds,
        )
        w = result.x
        arr_w = np.append(arr_w, w)
        volatility = np.sqrt(result.fun)
        arr_volatility = np.append(arr_volatility, volatility)

    arr_w = arr_w.reshape(len(arr_mu), -1)

    return arr_volatility, arr_w