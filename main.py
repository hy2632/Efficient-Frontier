from utils import *


class EfficientFrontier():
    def __init__(
        self,
        tickers: list,
        start_date: str,
        end_date: str,
        simulation_times=50000,
        solve_granularity=5000,
        analytical_solution=True,
        use_optimizer=False,
        tangency_line=True,
    ):

        self.tickers = tickers
        self.n_assets = len(tickers)
        self.analytical_solution = analytical_solution
        self.use_optimizer = use_optimizer
        self.tangency_line = tangency_line
        self.start_date = start_date
        self.end_date = end_date

        print(f"\n\n===Fetching assets data===")
        self.assets_daily_return = get_assets_data(start_date, end_date,
                                                   tickers)
        self.rf = np.mean(
            get_imputed_rf(start_date, end_date)["Adj Close"] / 100 / 253)
        print("Successfully fetched assets data!\n")

        self.Sigma = get_covariance_matrix(self.assets_daily_return)
        # R: mean daily return of assets in the pool, (n_assets,)
        self.R = np.mean(self.assets_daily_return, axis=1)

        # Monte Carlo
        print(f"===Running {simulation_times} Monte Carlo simulations===")
        self.arr_mu_MC, self.arr_volatility_MC, self.arr_w_MC = MonteCarlo(
            self.n_assets, self.Sigma, self.R, simulation_times)
        print("Success!\n")

        # set the mu array for solving, based on the range of mu from MC
        self.arr_mu_solve = np.linspace(self.arr_mu_MC.min(),
                                        self.arr_mu_MC.max() * 2,
                                        solve_granularity)

        # Analytical solution of efficient frontier (without risk-free asset)
        if self.analytical_solution:
            print("===Deriving Analytical solution of efficient frontier===")
            self.arr_volatility_analytical, self.arr_w_analytical = analyticalSolver(
                self.n_assets, self.Sigma, self.R, self.arr_mu_solve)
            print("Success!\n")

        # Solution using scipy.optimize.minimize()
        if self.use_optimizer:
            print("===Get efficient frontier using optimizer===")
            self.arr_volatility_optimizer, self.arr_w_optimizer = optimizerSolver(
                self.n_assets, self.Sigma, self.R, self.arr_mu_solve)
            print("Success!\n")

        if self.tangency_line:
            print(
                "===Plot the tangency line (efficient frontier with risk-free asset)==="
            )
            self.arr_volatility_tangency, self.arr_w_tangency = tangencySolver(
                self.n_assets, self.Sigma, self.R, self.rf, self.arr_mu_solve)

            self.optimal_SR = (self.arr_mu_solve[100] - self.arr_mu_solve[10]
                               ) / (self.arr_volatility_tangency[100] -
                                    self.arr_volatility_tangency[10])
            print(f"Success! Optimal sharpe ratio: {self.optimal_SR}\n")

    def get_optimal_SR(self, ):
        if self.tangency_line:
            return self.optimal_SR
        else:
            print("Did not solve the tangency line!")
            return None

    def plot(self, figsize=(12, 8)):

        # Settings for plotting
        plt.style.use('ggplot')
        plt.figure(figsize=figsize)
        plt.grid(True)
        plt.xlim(left=0)
        plt.ylabel("Expected Annual Return % (E(R))")
        plt.xlabel("Annual Volatility (Standard Deviation σ)")
        plt.set_cmap("rainbow")
        rcParams['font.family'] = 'Arial'
        font = {
            'family': 'Arial',
            'color': 'k',
            'weight': 'bold',
            'size': 22,
        }
        plt.title("E(R) - σ: " + self.start_date + " - " + self.end_date,
                  fontdict=font)

        # Annualize by 253 trading days per year
        arr_asset_var = self.Sigma.diagonal()
        for i, ticker in list(enumerate(self.tickers)):
            plt.plot(np.sqrt(arr_asset_var[i]) * np.sqrt(253),
                     self.R[i] * 253 * 100,
                     label=ticker,
                     marker="o",
                     mec="black")

        # MC, Analytical and Optimizer results
        plt.scatter(self.arr_volatility_MC * np.sqrt(253),
                    self.arr_mu_MC * 253 * 100,
                    color="grey",
                    label="Monte Carlo simulations")

        # If solved analytically:
        if self.analytical_solution:
            plt.plot(self.arr_volatility_analytical * np.sqrt(253),
                     self.arr_mu_solve * 253 * 100,
                     color="k",
                     label="Analytical (without risk-free asset)")

        # If used optimizer, plot the curve
        if self.use_optimizer:
            plt.plot(self.arr_volatility_optimizer * np.sqrt(253),
                     self.arr_mu_solve * 253 * 100,
                     color="Green",
                     label="Optimizer (without risk-free asset)")

        # If plot tangency line
        if self.tangency_line:
            plt.plot(self.arr_volatility_tangency * np.sqrt(253),
                     self.arr_mu_solve * 253 * 100,
                     color="c",
                     label="Tangency line (with risk-free asset)")

        plt.legend(loc="upper right")
        plt.show()


if __name__ == "__main__":
    instance = EfficientFrontier(
        tickers=["AAPL", "XOM", "PFE", "F", "WMT", "BA", "TSLA", "AMD"],
        start_date="20200101",
        end_date="20210130",
        simulation_times=5000,
        solve_granularity=1000,
        analytical_solution=True,
        use_optimizer=True,
        tangency_line=True,
    )
    instance.plot(figsize=None)
