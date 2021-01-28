import numpy as np
from pandas_datareader import data
from tqdm import tqdm


def randomWeightGen(n=4):
    w = np.random.random(n)
    return w / w.sum()


def rateOfReturn(asset: np.array):
    return asset[1:] / asset[:-1] - 1


def portfolio(w, assets):
    """
        Parameters:
        ---------
            assets.shape == (days, n_assets)
            w.shape == (n_assets)

        return:
        ---------
            portfolio.shape ==(days,)
    """
    return np.dot(assets, w)


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

    # Annual Risk free rate (US 10yr treasury bond, mean over the time period)
    Rf_mean = data.get_data_yahoo("^TNX", start_date,
                                  end_date)["Adj Close"].mean() / 100
    assets_daily_return = np.array([rateOfReturn(asset)
                                    for asset in assets]).squeeze()

    # Make assets to an array of (days, n_assets)
    assets = np.array(assets).squeeze()

    return assets, Rf_mean, assets_daily_return


def get_covariance_matrix(assets_daily_return):
    """
        assets_daily_return: (days, n_assets)
    """

    Sigma = np.cov(assets_daily_return.T, ddof=0)
    return Sigma


def MonteCarlo(assets):
    """
        Parameters:
        ---------
            assets.shape == (days, n_assets)
        
        return:
        ---------
            MC_axis_X, MC_axis_y: 30000 randomly generated portfolios for plotting
        
    """

    MC_axis_X = []
    MC_axis_y = []

    for i in tqdm(range(50000)):
        asset = portfolio(randomWeightGen(assets.shape[1]), assets)
        MC_axis_X.append(np.std(rateOfReturn(asset)) * np.sqrt(253))
        MC_axis_y.append(np.mean(rateOfReturn(asset)) * 253)

    return MC_axis_X, MC_axis_y


def solve(n, Sigma, R):
    """
        Parameters:
        ---------
        n: n_assets

        Sigma: Covariance matrix of shape (n, n)

        R: Expected annual return (mean over the time period) of assets in the pool, of shape (n, )
        
        return:
        ---------
        arr_mu: array of "mu" (expected annual return of the portfolio) ranging from 0 to 0.4, the y coordinates

        arr_volatility: array of annual volatility (standard deviation) of the portfolio, the x coordinates
        
        arr_w: array of the weight vector "w" at each point
    """

    # The matrix on the left
    mat1 = np.vstack([
        np.hstack([2 * Sigma, -np.expand_dims(R, axis=1), -np.ones((n, 1))]),
        np.hstack([R, [0], [0]]),
        np.hstack([np.ones(n), [0], [0]])
    ])

    arr_mu = np.linspace(0, 0.3, 2000)
    arr_volatility = []
    arr_w = []

    for mu in arr_mu:
        vec2 = np.array([0] * n + [mu] + [1])
        w_lambda = np.linalg.solve(mat1, vec2)
        w = w_lambda[:n]
        arr_w.append(w)
        volatility = np.sqrt(np.dot(w, np.dot(Sigma, w))) * np.sqrt(253)
        arr_volatility.append(volatility)

    arr_volatility = np.array([arr_volatility]).squeeze()

    return arr_mu, arr_volatility, arr_w