from utils import *


class EfficientFrontier():
    def __init__(
        self,
        tickers: list,
        start_date: str,
        end_date: str,
        simulation_times=50000,
        solve_granularity=5000,
    ):

        self.tickers = tickers
        self.n_assets = len(tickers)

        print(f"===Fetching assets data===")
        self.Rf_mean, self.assets_daily_return = get_assets_data(
            start_date, end_date, tickers)
        print("Successfully fetched assets data!\n")

        self.Sigma = get_covariance_matrix(self.assets_daily_return)
        # R: mean daily return
        self.R = np.mean(self.assets_daily_return, axis=0)

        # Monte Carlo
        print(f"===Running {simulation_times} Monte Carlo simulations===")
        self.arr_mu_MC, self.arr_volatility_MC, self.arr_w_MC = MonteCarlo(
            self.n_assets, self.Sigma, self.R, simulation_times)
        print("Success!\n")

        # set the mu array for solving, based on the range of mu from MC
        self.arr_mu_solve = np.linspace(self.arr_mu_MC.min(),
                                        self.arr_mu_MC.max(),
                                        solve_granularity)

        # Analytical solution
        print("===Deriving Analytical solution of efficient frontier===")
        self.arr_volatility_analytical, self.arr_w_analytical = analyticalSolver(
            self.n_assets, self.Sigma, self.R, self.arr_mu_solve)
        print("Success!\n")

        # Solution using scipy.optimize.minimize()
        print("===Get efficient frontier using optimizer===")
        self.arr_volatility_optimizer, self.arr_w_optimizer = optimizerSolver(
            self.n_assets, self.Sigma, self.R, self.arr_mu_solve)
        print("Success!\n")

    def printParams(self, ):
        print(f"Asset tickers: {self.tickers}")
        print("Covariance Matrix (Sigma): ")
        print(self.Sigma)
        print("Expected annual return: %.4f" % self.R)

    def plot(self, ):
        plt.style.use('ggplot')
        plt.figure(figsize=(10, 6))
        plt.grid(True)

        # plt.xlim(0, self.arr_volatility_MC.max() * np.sqrt(253) * 1.25)
        # plt.ylim(0, self.arr_mu_MC.max() * 253 * 1.25)

        # Remember to Annualize
        # Plot the asset pool
        arr_asset_var = self.Sigma.diagonal()
        for i, ticker in list(enumerate(self.tickers)):
            plt.plot(np.sqrt(arr_asset_var[i]) * np.sqrt(253),
                     self.R[i] * 253,
                     label=ticker,
                     marker="x")

        # MC, Analytical and Optimizer results
        plt.scatter(self.arr_volatility_MC * np.sqrt(253),
                    self.arr_mu_MC * 253,
                    color="grey",
                    label="Monte Carlo")

        plt.plot(self.arr_volatility_analytical * np.sqrt(253),
                 self.arr_mu_solve * 253,
                 color="Red",
                 label="Analytical")

        plt.plot(self.arr_volatility_optimizer * np.sqrt(253),
                 self.arr_mu_solve * 253,
                 color="Green",
                 label="Optimizer")

        plt.legend()
        plt.show()


if __name__ == "__main__":
    instance = EfficientFrontier(
        [
            "AAPL",
            "XOM",
            "PFE",
            "WMT",
            "BA",
        ],
        "20090101",
        "20200101",
        50000,
        500,
    )
    instance.plot()