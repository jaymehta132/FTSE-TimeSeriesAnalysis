# model_grid_search.py
"""
Grid-search ARMA(p,q) x GARCH(1,1) with Normal / Student-t innovations.
Saves:
 - model_grid_results.csv : metrics for each model
 - diagnostics/ : diagnostic plots for top models
Requires:
 - data/FTSE_PreprocessedData.csv
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from scipy.stats import jarque_bera, probplot
from arch import arch_model

from tqdm import tqdm

sns.set(style="whitegrid")
warnings.filterwarnings("ignore")

# -------------- Config ----------------
DATA_PATH = "data/FTSE_PreprocessedData.csv"
RET_COLS = ["Log Returns", "Returns"]  # prefer Log Returns; fallback to Returns
OUT_DIR = "model_grid_outputs"
DIAG_DIR = os.path.join(OUT_DIR, "diagnostics")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DIAG_DIR, exist_ok=True)

# Grid
AR_P = [0, 1]
MA_Q = [0, 1]
GARCH_P = 1
GARCH_Q = 1
DISTS = ["normal", "t"]  # distributions to try

LB_LAGS = [10, 20, 40]  # Ljung-Box lags to report

# -------------- Helpers ----------------
def load_returns(path, prefer_col="Log Returns"):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    # choose column robustly
    col = None
    for c in [prefer_col, "Log Returns", "Returns", "Return", "logret", "log_return"]:
        if c in df.columns:
            col = c
            break
    if col is None:
        raise ValueError(f"Couldn't find a returns column in {path}. Available columns: {df.columns.tolist()}")
    series = df[col].astype(float).dropna()
    series.index = pd.to_datetime(df.loc[series.index, "Date"])
    return series, col

def safe_loglik(res):
    """Try extracting log-likelihood from common attributes, else None."""
    for attr in ("llf", "loglikelihood", "loglik", "loglikelihood_value", "loglikelihood_"):
        if hasattr(res, attr):
            val = getattr(res, attr)
            try:
                return float(val)
            except Exception:
                pass
    # statsmodels ARIMAResults has llf attribute usually; arch results maybe loglikelihood
    if hasattr(res, "llf"):
        try:
            return float(res.llf)
        except Exception:
            pass
    # fallback
    return None

def safe_params_count(res):
    try:
        return len(res.params)
    except Exception:
        try:
            return res.params.shape[0]
        except Exception:
            return None

def extract_garch_alpha_beta(garch_res):
    """Try to extract alpha and beta from param names robustly."""
    alpha = np.nan
    beta = np.nan
    try:
        pser = garch_res.params  # pandas Series
        for name, val in pser.items():
            n = name.lower()
            if "alpha" in n and np.isnan(alpha):
                alpha = float(val)
            if "beta" in n and np.isnan(beta):
                beta = float(val)
            # sometimes arch names look like 'a[1]' 'b[1]'
            if (n.startswith("a") and n[1:].strip("[]0123456789") == "") and np.isnan(alpha):
                # ignore
                pass
        # If not found, attempt positional:
        if np.isnan(alpha) or np.isnan(beta):
            vals = np.asarray(pser, dtype=float)
            # heuristics: typically omega, alpha[1], beta[1], ... -> positions 0,1,2
            if vals.size >= 3:
                if np.isnan(alpha):
                    alpha = float(vals[1])
                if np.isnan(beta):
                    beta = float(vals[2])
    except Exception:
        pass
    return alpha, beta

# -------------- Main grid search ----------------
def run_grid():
    returns, used_col = load_returns(DATA_PATH, prefer_col=RET_COLS[0])
    print(f"Using returns column: '{used_col}' with {returns.shape[0]} observations.")

    results = []
    n = len(returns)

    # Loop grid
    for p in AR_P:
        for q in MA_Q:
            for dist in DISTS:
                model_name = f"ARMA({p},{q}) + GARCH({GARCH_P},{GARCH_Q})-{dist}"
                print(f"\nFitting {model_name} ...")
                try:
                    # 1) Fit ARMA(p,q) on returns using ARIMA (ARIMA(order=(p,0,q)))
                    #    suppress convergence warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        arima = ARIMA(returns, order=(p, 0, q))
                        arima_res = arima.fit(method_kwargs={"warn_convergence": False})
                    
                    resid = arima_res.resid.dropna()
                    # 2) Fit GARCH on residuals (mean=0 since mean removed by ARMA)
                    am = arch_model(resid, mean="Zero", vol="Garch", p=GARCH_P, q=GARCH_Q, dist=dist)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        garch_res = am.fit(disp="off")
                    
                    # standardized residuals
                    try:
                        std_resid = garch_res.std_resid.dropna()
                    except Exception:
                        # compute manually
                        cond_vol = getattr(garch_res, "conditional_volatility", None)
                        if cond_vol is None:
                            cond_vol = garch_res.conditional_volatility
                        std_resid = resid / cond_vol
                        std_resid = pd.Series(std_resid).dropna()

                    # combined loglik
                    ll_arima = safe_loglik(arima_res)
                    ll_garch = safe_loglik(garch_res)
                    combined_ll = None
                    if ll_arima is not None and ll_garch is not None:
                        combined_ll = ll_arima + ll_garch

                    # combined params count
                    k_arima = safe_params_count(arima_res) or 0
                    k_garch = safe_params_count(garch_res) or 0
                    k_total = k_arima + k_garch

                    # AIC / BIC (combined; approximate)
                    aic_combined = np.nan
                    bic_combined = np.nan
                    if combined_ll is not None and k_total>0:
                        aic_combined = -2.0 * combined_ll + 2.0 * k_total
                        bic_combined = -2.0 * combined_ll + k_total * np.log(n)

                    # Ljung-Box on standardized residuals & squared standardized residuals
                    lb_res = acorr_ljungbox(std_resid, lags=LB_LAGS, return_df=True)
                    lb_sq = acorr_ljungbox(std_resid**2, lags=LB_LAGS, return_df=True)

                    # ARCH-LM on standardized residuals (nlags=12)
                    arch_test = het_arch(std_resid, nlags=12)

                    # Jarque-Bera on standardized residuals
                    jb_stat, jb_p = jarque_bera(std_resid)[:2]

                    # extract alpha & beta
                    alpha, beta = extract_garch_alpha_beta(garch_res)
                    alpha_beta_sum = np.nan
                    try:
                        alpha_beta_sum = float(alpha) + float(beta)
                    except Exception:
                        alpha_beta_sum = np.nan

                    # store
                    results.append({
                        "model": model_name,
                        "p": p, "q": q, "dist": dist,
                        "n_obs": n,
                        "ll_arima": ll_arima, "ll_garch": ll_garch, "ll_combined": combined_ll,
                        "k_params": k_total,
                        "aic_combined": aic_combined, "bic_combined": bic_combined,
                        # Ljung-Box p-values for residuals for each lag
                        **{f"lb_res_p_lag{lag}": float(lb_res["lb_pvalue"].loc[lag]) for lag in LB_LAGS},
                        **{f"lb_sq_p_lag{lag}": float(lb_sq["lb_pvalue"].loc[lag]) for lag in LB_LAGS},
                        "arch_lm_stat": float(arch_test[0]), "arch_lm_p": float(arch_test[1]),
                        "jb_stat": float(jb_stat), "jb_p": float(jb_p),
                        "alpha": alpha, "beta": beta, "alpha_beta_sum": alpha_beta_sum,
                    })

                    # Save diagnostics for top models (by aic later) - but also save per-fit plots here
                    basefname = f"p{p}_q{q}_{dist}"
                    # Q-Q plot of standardized residuals (vs chosen dist)
                    plt.figure(figsize=(6,5))
                    if dist == "t":
                        # compare to t with df estimated in model if available
                        df_nu = None
                        try:
                            # 'nu' parameter name is common for arch t-dist
                            params = garch_res.params
                            for name, val in params.items():
                                if name.lower().startswith("nu") or "df" in name.lower():
                                    df_nu = float(val)
                                    break
                        except Exception:
                            df_nu = None
                        if df_nu is not None and df_nu > 2:
                            # use scipy's probplot with t
                            import scipy.stats as st
                            probplot(std_resid, dist=st.t(df_nu), plot=plt)
                            plt.title(f"Q-Q: {basefname} vs t(df={df_nu:.2f})")
                        else:
                            # fallback to normal qq
                            probplot(std_resid, dist="norm", plot=plt)
                            plt.title(f"Q-Q: {basefname} (fallback normal)")
                    else:
                        probplot(std_resid, dist="norm", plot=plt)
                        plt.title(f"Q-Q: {basefname} vs Normal")
                    plt.tight_layout()
                    plt.savefig(os.path.join(DIAG_DIR, f"qq_{basefname}.png"))
                    plt.close()

                    # ACF plots for residuals and squared residuals
                    try:
                        from statsmodels.graphics.tsaplots import plot_acf
                        fig, ax = plt.subplots(1,2, figsize=(12,4))
                        plot_acf(std_resid, lags=40, ax=ax[0], title=f"ACF std_resid {basefname}")
                        plot_acf(std_resid**2, lags=40, ax=ax[1], title=f"ACF std_resid^2 {basefname}")
                        plt.tight_layout()
                        plt.savefig(os.path.join(DIAG_DIR, f"acf_{basefname}.png"))
                        plt.close()
                    except Exception:
                        pass

                except Exception as e:
                    print(f" fit failed for {model_name}: {e}")
                    results.append({
                        "model": model_name,
                        "p": p, "q": q, "dist": dist,
                        "n_obs": n,
                        "ll_arima": np.nan, "ll_garch": np.nan, "ll_combined": np.nan,
                        "k_params": np.nan,
                        "aic_combined": np.nan, "bic_combined": np.nan,
                        **{f"lb_res_p_lag{lag}": np.nan for lag in LB_LAGS},
                        **{f"lb_sq_p_lag{lag}": np.nan for lag in LB_LAGS},
                        "arch_lm_stat": np.nan, "arch_lm_p": np.nan,
                        "jb_stat": np.nan, "jb_p": np.nan,
                        "alpha": np.nan, "beta": np.nan, "alpha_beta_sum": np.nan,
                    })
    # Save results
    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values(["aic_combined", "bic_combined"], ascending=True)
    df_res.to_csv(os.path.join(OUT_DIR, "model_grid_results.csv"), index=False)
    print(f"\nGrid search finished. Results saved to {os.path.join(OUT_DIR, 'model_grid_results.csv')}")
    return df_res

if __name__ == "__main__":
    df_results = run_grid()
    # show best few:
    print("\nTop models by AIC (combined):")
    print(df_results[["model", "aic_combined", "bic_combined", "alpha_beta_sum", "arch_lm_p", "jb_p"]].head(10))
