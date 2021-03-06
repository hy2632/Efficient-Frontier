﻿# Efficient-Frontier

## References
* [Wikipedia - Modern portfolio theory - Mathematical Model](https://en.wikipedia.org/wiki/Modern_portfolio_theory)

* [马科维茨投资模型](https://mp.weixin.qq.com/s/0RlN0UTh3slFgErd8XRGhQ)

* [Foundations of Financial Engineering (Columbia University, Fall 2016)](http://www.columbia.edu/~mh2078/FoundationsFE.html#:~:text=It%20is%20a%20core%20course,and%20financial%20problems%20and%20products.)

## Brief

Analysis on the modern portfolio theory.

* Efficient frontier with / without risk free asset
* Optimal sharpe ratio / portfolio

## Example of usage

```python
instance = EfficientFrontier(
    symbols=["AAPL", "XOM", "PFE", "F", "WMT", "BA", "TSLA", "AMD"],
    start_date="20200101",
    end_date="20210130", 
    simulation_times=5000, 
    solve_granularity=1000, 
    analytical_solution=True,
    use_optimizer=True,
    tangency_line=True,
)
instance.plot((12,8))
```

![](Misc/img/figure.png)

## Content

- Monte Carlo simulation of portfolios consisting of assets in the designated pool.
- Solve for the curve of the efficient frontier **without risk-free asset** analytically (a constrained optimization problem) using lagrange multiplier. This solution did not take into consideration the constraint that all weights are greater than 0, a.k.a. it allows short and leverage.
  - ![](Misc/img/render1.png)

<!-- Rendering provided by http://www.sciweavers.org/free-online-latex-equation-editor -->
<!-- $$
\begin{bmatrix}2\Sigma &-R & -{\bf1}\\ R^T &0 & 0 \\ {\bf1}^T &0 &0 \end{bmatrix} 
* \begin{bmatrix}w\\\lambda_1\\\lambda_2\end{bmatrix} 
= \begin{bmatrix}0\\\mu \\ 1\end{bmatrix}
$$ -->

- (2021/1/30 Update) `optimizerSolver()` to solve for the efficient frontier using scipy optimizer. Added the constraint of no short. Due to the limit of the optimizer, this solution performs bad for low `mu` part.

- (2021/2/1 Update) `tangencySolver()` which solves for the "with risk-free asset" case. The solution can be proved to be the tangency line of the efficient frontier curve in the "no risk-free asset" setting.

- Plotting the figure of all above.

### 2021/2/1: Frontier with risk-free asset —— the tangency line

See http://yaohuacn.com/2021/02/02/Modern-Portfolio-Theory-Mean-variance-analysis-Efficient-Frontier/
