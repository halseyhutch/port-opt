import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt
import streamlit as st
import yfinance as yf
from sklearn.covariance import LedoitWolf

# ---------------------------------------------
# Helper functions
# ---------------------------------------------


def compute_portfolio_stats(weights, mu, cov):
    ret = np.dot(weights, mu)
    vol = np.sqrt(np.dot(weights.T, cov).dot(weights))
    return ret, vol


def mean_variance_objective(weights, mu, cov, lam):
    ret, vol = compute_portfolio_stats(weights, mu, cov)
    # maximize return - lambda*risk, flipped since scipy minimize is used
    return -(ret - lam * vol)


def max_mu_objective(weights, mu, cov, lam):
    ret, _ = compute_portfolio_stats(weights, mu, cov)
    return -ret


def vol_constraint_factory(Sigma, target_vol):
    def cons_fun(w):
        var = float(w.dot(Sigma).dot(w))
        # require var <= target_vol^2 -> cons_fun(w) >= 0
        return target_vol**2 - var
    return cons_fun


OBJECTIVES = {
    'Mean-Variance': mean_variance_objective,
    'Target Vol': max_mu_objective,
}


@st.cache_data
def safe_download(tickers, **kwargs):
    for attempt in range(3):
        try:
            return yf.download(tickers, **kwargs)
        except Exception as e:
            time.sleep(0.5)


def min_variance_for_target_return(mu, Sigma, target_return):
    n = len(mu)

    def portfolio_variance(w):
        return w.T @ Sigma @ w

    # Constraints
    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w: w @ mu - target_return}
    ]

    # Bounds for long-only
    bounds = opt.Bounds(np.zeros(n), np.ones(n))

    # Initial guess
    w0 = np.ones(n) / n

    result = opt.minimize(
        portfolio_variance, w0,
        method="SLSQP",
        constraints=cons,
        bounds=bounds
    )

    if not result.success:
        return None, None

    w_opt = result.x
    var_opt = portfolio_variance(w_opt)
    return w_opt, var_opt


def efficient_frontier(mu, Sigma, num_points=40):
    mu_min, mu_max = mu.min(), mu.max()
    target_returns = np.linspace(mu_min, mu_max, num_points)

    vols = []
    rets = []
    weights = []

    for tr in target_returns:
        w, var = min_variance_for_target_return(mu, Sigma, tr)
        if w is None:
            vols.append(np.nan)
            rets.append(np.nan)
            weights.append([np.nan]*len(mu))
        else:
            vols.append(np.sqrt(var))
            rets.append(tr)
            weights.append(w)

    return np.array(vols), np.array(rets), np.array(weights), target_returns


def return_at_target_vol(target_vol, vols, rets):
    # Requires vols sorted
    order = np.argsort(vols)
    vols_sorted = vols[order]
    rets_sorted = rets[order]
    return np.interp(target_vol, vols_sorted, rets_sorted)


# ---------------------------------------------
# Report
# ---------------------------------------------

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
st.title("Mean-Variance Portfolio Optimizer")

tab_data, tab_results, tab_frontier = st.tabs(
    ["Inputs", "Optimization Results", "Efficient Frontier"])

with tab_data:
    st.header("Asset List")

    # Default asset list
    default_assets = [
        {"Asset": "US Equity", "Ticker": "SPY", "lb": 0.0, "ub": 1.0},
        {"Asset": "Developed Intl Equity", "Ticker": "VEA", "lb": 0.0, "ub": 1.0},
        {"Asset": "Emerging Intl Equity", "Ticker": "EEM", "lb": 0.0, "ub": 1.0},
        {"Asset": "TIPS", "Ticker": "TIP", "lb": 0.0, "ub": 1.0},
        {"Asset": "US IG Bonds", "Ticker": "AGG", "lb": 0.0, "ub": 1.0},
        {"Asset": "Intl Bonds", "Ticker": "BNDX", "lb": 0.0, "ub": 1.0},
        {"Asset": "REITs", "Ticker": "VNQ", "lb": 0.0, "ub": 1.0},
        {"Asset": "Gold", "Ticker": "GLD", "lb": 0.0, "ub": 1.0},
        {"Asset": "Industrial Metals", "Ticker": "XME", "lb": 0.0, "ub": 1.0},
        {"Asset": "Oil", "Ticker": "USO", "lb": 0.0, "ub": 1.0},
        {"Asset": "BTC", "Ticker": "BTC-USD", "lb": 0.0, "ub": 1.0},
    ]

    # Dynamic table for asset info
    df_assets = st.data_editor(
        pd.DataFrame(default_assets),
        num_rows="dynamic",
        key="asset_editor",
        hide_index=True,
        column_config={
            "Asset": "Asset",
            "Ticker": "Ticker",
            "lb": st.column_config.NumberColumn(
                "Min Alloc",
                format="percent"
            ),
            'ub': st.column_config.NumberColumn(
                "Max Alloc",
                format="percent"
            ),
        },
    )

    tickers = df_assets["Ticker"].tolist()
    upper_bounds = df_assets['ub'].astype(float).values
    lower_bounds = df_assets['lb'].astype(float).values
    n = len(df_assets)

    st.divider()
    st.subheader("Estimate Returns & Covariance")

    start_date, end_date = st.date_input('Date Range', (datetime.datetime.now(
    ) - datetime.timedelta(365 * 5), datetime.datetime.now()))
    freq = st.selectbox("Return frequency", ["Weekly", "Monthly"], index=0)

    interval_map = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
    interval = interval_map[freq]

    price_df = safe_download(tickers, start=start_date,
                             end=end_date, interval=interval)["Close"]

    if price_df.empty:
        st.sidebar.error(
            "Price fetch returned no data — check tickers or try again later.")
    else:
        # align dataframe
        price_df = price_df.reindex(tickers, axis=1)
        # if single ticker, make it a DataFrame
        if isinstance(price_df, pd.Series):
            price_df = price_df.to_frame()
        price_df = price_df[~price_df.index.duplicated(keep='first')]

        # Resample if needed
        if freq == 'Weekly':
            price_df = price_df.resample('W-FRI').last()
        elif freq == 'Monthly':
            price_df = price_df.resample('M').last()

        # compute log returns
        returns = np.log(price_df / price_df.shift(1)).dropna()
        # annualization factor
        if freq == 'Daily':
            ann_factor = 252
        elif freq == 'Weekly':
            ann_factor = 52
        else:
            ann_factor = 12

        mu_est = returns.mean() * ann_factor
        Sigma_est = LedoitWolf().fit(returns).covariance_ * \
            ann_factor  # returns.cov() * ann_factor

        sigma_est = pd.Series(np.sqrt(np.diag(Sigma_est)), index=tickers)
        Dinv = np.diag(1 / np.sqrt(np.diag(Sigma_est)))
        rho_est = Dinv @ Sigma_est @ Dinv

        st.write("**Estimated Expected Returns (μ):**")
        mu_df = st.data_editor(
            pd.DataFrame({"Return": mu_est, "Stdev": sigma_est}),
            column_config={k: st.column_config.NumberColumn(
                k, format='percent') for k in ['Return', 'Stdev']},
        )

        st.write("**Estimated Correlations (ρ):**")
        rho_df = st.data_editor(
            pd.DataFrame(rho_est, index=tickers, columns=tickers),
            column_config={tick: st.column_config.NumberColumn(
                tick, format='percent') for tick in tickers}
        )

        mu = mu_df['Return'].astype(float).values
        sigma = mu_df['Stdev'].astype(float).values
        rho = rho_df.astype(float).values
        Sigma = np.diag(sigma) @ rho @ np.diag(sigma)

        st.divider()

        st.write('**Sample Rolling Correlations:**')
        window = st.slider(
            "Rolling correlation window (weeklies only)", 12, 104, 26)
        target = st.selectbox("Choose asset for rolling correlation", tickers)
        compare_assets = st.multiselect(
            "Compare target asset with:",
            tickers,
            default=[t for t in tickers if t != target]
        )

        rolling_corr = returns.rolling(window).corr()
        rolling_corr_df = (
            rolling_corr
            .loc[pd.IndexSlice[:, target], compare_assets]
            .reset_index(level=1, drop=True)
            .dropna(how='all', axis=0)
            .mul(100)
        )
        st.line_chart(rolling_corr_df, y_label='Corr (%)')

with tab_results:
    st.header("Optimization")

    capital = st.number_input("Total Capital ($M)", min_value=0.0, value=10.0)
    obj_name = st.radio(
        'Objective',
        OBJECTIVES.keys()
    )
    lam = st.number_input("Lambda (risk aversion)", min_value=0.0, value=1.0)
    target_vol = st.number_input("Target Volatility", min_value=0.0, value=0.1)
    run_opt = st.button("Run Optimization")

    if run_opt:
        x0 = np.ones(n) / n
        bounds = list(zip(lower_bounds, upper_bounds))
        cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        obj_fn = OBJECTIVES[obj_name]
        if obj_name == 'Target Vol':
            cons = (cons, {'type': 'ineq',
                    'fun': vol_constraint_factory(Sigma, target_vol)})

        res = opt.minimize(obj_fn, x0, args=(mu, Sigma, lam),
                           bounds=bounds, constraints=cons)

        if res.success:
            w_opt = res.x
            port_ret, port_vol = compute_portfolio_stats(w_opt, mu, Sigma)
            # Contributions to expected return
            ret_contrib = pd.Series(w_opt, index=tickers).mul(mu, axis=0)
            # Contributions to variance: w_i * (Sigma w)_i
            marginal = Sigma.dot(w_opt)
            var_contrib = pd.Series(w_opt, index=tickers) * marginal
            risk_ctrb_pct = var_contrib / var_contrib.sum()
            risk_ctrb_abs = risk_ctrb_pct * port_vol
            result_df = pd.DataFrame({
                "Asset": df_assets["Asset"],
                "Ticker": tickers,
                "Weight": w_opt * 100,
                "Dollars": w_opt * capital,
                "Exp Ret Ctrb": ret_contrib.values * 100,
                "Risk Ctrb": risk_ctrb_abs.values * 100,
            })

            st.subheader("Optimal Allocation")
            st.metric("Expected Portfolio Return", f"{port_ret:.2%}")
            st.metric("Expected Portfolio Volatility", f"{port_vol:.2%}")

            st.dataframe(
                result_df,
                column_config={
                    'Weight': st.column_config.NumberColumn('Weight (%)', format='%.1f'),
                    'Dollars': st.column_config.NumberColumn('Dollars ($M)', format='%.2f'),
                    'Exp Ret Ctrb': st.column_config.NumberColumn('Exp Ret Ctrb (%)', format='%.1f'),
                    'Risk Ctrb': st.column_config.NumberColumn('Risk Ctrb (%)', format='%.1f'),
                },
                hide_index=True,
            )

        else:
            st.error("Optimization failed.")

with tab_frontier:
    st.header("Efficient Frontier")
    selected_assets = st.multiselect(
        "Select assets to plot", tickers, default=tickers[:2])
    run_frontier = st.button('Calculate Frontier')

    if run_frontier:
        col1, col2 = st.columns(2)

        vols, rets, all_weights, target_returns = efficient_frontier(
            mu, Sigma, num_points=60)

        with col1:
            fig1, ax1 = plt.subplots()
            ax1.plot(vols * 100, rets * 100)
            ax1.set_xlabel("Target Volatility (%)")
            ax1.set_ylabel("Expected Return (%)")
            ax1.set_title("Efficient Frontier")
            st.pyplot(fig1, width=500)

        with col2:
            fig2, ax2 = plt.subplots()
            for asset in selected_assets:
                idx = tickers.index(asset)
                ax2.plot(vols * 100, all_weights[:, idx] * 100, label=asset)

            ax2.set_xlabel("Target Volatility (%)")
            ax2.set_ylabel("Weight (%)")
            ax2.set_title("Weights Along the Efficient Frontier")
            ax2.legend()
            st.pyplot(fig2, width=500)
