# behavioral-portfolio
Optimization code for the paper:
    
A. Cinfrignini, D. Petturiti and B. Vantaggi. Behavioral dynamic portfolio selection with S-shaped utility and epsilon-contaminations. **European Journal of Operational Research**, Accepted, 2025.


# Requirements
The code has been tested on Python 3.10 with the following libraries:
* **matplotlib** 3.7.1
* **numpy** 1.23.5
* **pandas** 1.5.3
* **pyomo** 6.6.1
* **scipy** 1.10.1
* **yfinance** 0.2.18

Reference to the library **pyomo** is here: http://www.pyomo.org/

Reference to the library **yfinance** is here: https://pypi.org/project/yfinance/

The code necessitates of the **bonmin** solver that can be downloaded here: https://www.coin-or.org/Bonmin/

The **bonmin** solver should be located in a folder and the path to that folder should be inserted in the variable **optimizer_path** in the top of file **behavioral_portfolio.py**

# File inventory
**behavioral_portfolio.py**: Optimization code that reduces the dynamic portfolio selection in a binomial
market into the maximization of the function of the variable eta:
_f(eta, T, u, d, V0, r, p, epsilon_p, epsilon_m, alpha_p, alpha_m, lamb)_
where:
* _eta_: optimization variable
* _T_: time horizon in periods
* _u_: up factor for the binomial market
* _d_: down factor for the binomial market
* _V0_: initial endowment (either positive or negative)
* _r_: risk-free interest rate per period
* _p_: real-world probability parameter for the binomial market
* _epsilon_p_: ambiguity parameter for gains
* _epsilon_m_: ambiguity parameter for losses
* _alpha_p_: constant relative risk aversion for gains
* _alpha_m_: constant relative risk aversion for losses
* _lamb_: scale parameter for losses

**IMPORTANT: eta is assumed to range in [[V0]+, +infinity)**

**calibration_GOOG.ipynb**: Calibration of parameters on market data for the stock GOOG downloading data from Yahoo! Finance through the library yfinance

**exhaustive_search.py**: Maximizes the following function with respect to the variable eta using an exhausitive search in a partition of a subinterval of [[V0]+, +infinity):
_f(eta, T, u, d, V0, r, p, epsilon_p, epsilon_m, alpha_p, alpha_m, lamb)_

**golden_section_search.py**: Maximizes the following function with respect to the variable eta using the Golden-section search algorithm:
_f(eta, T, u, d, V0, r, p, epsilon_p, epsilon_m, alpha_p, alpha_m, lamb)_

**graph_f.py**: Plots the graph of the following function with respect to the variable eta:
_f(eta, T, u, d, V0, r, p, epsilon_p, epsilon_m, alpha_p, alpha_m, lamb)_
