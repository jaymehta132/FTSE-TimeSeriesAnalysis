import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from scipy.stats import jarque_bera, probplot
import scipy.stats as st
from arch import arch_model

from tqdm import tqdm

sns.set(style="whitegrid")
warnings.filterwarnings("ignore")

DATA_PATH = "../data/FTSE_PreprocessedData.csv"
RET_COLS = ["Log Returns", "Returns"]  
OUT_DIR = "../results/model_grid_outputs"
DIAG_DIR = os.path.join(OUT_DIR, "diagnostics")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DIAG_DIR, exist_ok=True)

AR_P = [0, 1]
MA_Q = [0, 1]
GARCH_P = 1
GARCH_Q = 1
DISTS = ["normal", "t"]  # distributions to try

LB_LAGS = [10, 20, 40]  # Ljung-Box lags to report

def load_returns(path, prefer_col="Log Returns"):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    start, end = pd.Timestamp("2005-01-01"), pd.Timestamp("2007-12-31")
    df = df[(df["Date"] >= start) & (df["Date"] <= end)].reset_index(drop=True)

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

def log_likelihood(res):
    for attr in ("llf", "loglikelihood", "loglik", "loglikelihood_value", "loglikelihood_"):
        if hasattr(res, attr):
            val = getattr(res, attr)
            return float(val)
    if hasattr(res, "llf"):
        return float(res.llf)
    return None

def safe_params_count(res):
    return len(res.params)
    

def extract_garch_alpha_beta(garch_res):
    alpha = np.nan
    beta = np.nan
    pser = garch_res.params  
    for name, val in pser.items():
        n = name.lower()
        if "alpha" in n and np.isnan(alpha):
            alpha = float(val)
        if "beta" in n and np.isnan(beta):
            beta = float(val)
        if (n.startswith("a") and n[1:].strip("[]0123456789") == "") and np.isnan(alpha):
            pass
    if np.isnan(alpha) or np.isnan(beta):
        vals = np.asarray(pser, dtype=float)
        if vals.size >= 3:
            if np.isnan(alpha):
                alpha = float(vals[1])
            if np.isnan(beta):
                beta = float(vals[2])
    return alpha, beta

def run_grid():
    returns, used_col = load_returns(DATA_PATH, prefer_col=RET_COLS[0])
    print(f"Using returns column: '{used_col}' with {returns.shape[0]} observations.")

    results = []
    n = len(returns)

    for p in AR_P:
        for q in MA_Q:
            for dist in DISTS:
                model_name = f"ARMA({p},{q}) + GARCH({GARCH_P},{GARCH_Q})-{dist}"
                print(f"\nFitting {model_name} ...")

                # Fit ARMA(p,q) on returns
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    arima = ARIMA(returns, order=(p, 0, q))
                    arima_res = arima.fit(method_kwargs={"warn_convergence": False})
                
                resid = arima_res.resid.dropna()
                # Fit GARCH on residuals (mean=0 since mean removed by ARMA)
                am = arch_model(resid, mean="Zero", vol="Garch", p=GARCH_P, q=GARCH_Q, dist=dist)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    garch_res = am.fit(disp="off")

                try:
                    std_resid = garch_res.std_resid.dropna()
                except Exception:
                    cond_vol = getattr(garch_res, "conditional_volatility", None)
                    if cond_vol is None:
                        cond_vol = garch_res.conditional_volatility
                    std_resid = resid / cond_vol
                    std_resid = pd.Series(std_resid).dropna()

                ll_arima = log_likelihood(arima_res)
                ll_garch = log_likelihood(garch_res)
                combined_ll = None
                if ll_arima is not None and ll_garch is not None:
                    combined_ll = ll_arima + ll_garch

                k_arima = safe_params_count(arima_res) or 0
                k_garch = safe_params_count(garch_res) or 0
                k_total = k_arima + k_garch

                aic_combined = np.nan
                bic_combined = np.nan
                if combined_ll is not None and k_total>0:
                    aic_combined = -2.0 * combined_ll + 2.0 * k_total
                    bic_combined = -2.0 * combined_ll + k_total * np.log(n)

                # Ljung-Box 
                lb_res = acorr_ljungbox(std_resid, lags=LB_LAGS, return_df=True)
                lb_sq = acorr_ljungbox(std_resid**2, lags=LB_LAGS, return_df=True)

                # ARCH-LM
                arch_test = het_arch(std_resid, nlags=12)

                # Jarque-Bera
                jb_stat, jb_p = jarque_bera(std_resid)[:2]

                # extract alpha & beta
                alpha, beta = extract_garch_alpha_beta(garch_res)
                alpha_beta_sum = np.nan
                alpha_beta_sum = float(alpha) + float(beta)

                results.append({
                    "model": model_name,
                    "p": p, "q": q, "dist": dist,
                    "n_obs": n,
                    "ll_arima": ll_arima, "ll_garch": ll_garch, "ll_combined": combined_ll,
                    "k_params": k_total,
                    "aic_combined": aic_combined, "bic_combined": bic_combined,
                    **{f"lb_res_p_lag{lag}": float(lb_res["lb_pvalue"].loc[lag]) for lag in LB_LAGS},
                    **{f"lb_sq_p_lag{lag}": float(lb_sq["lb_pvalue"].loc[lag]) for lag in LB_LAGS},
                    "arch_lm_stat": float(arch_test[0]), "arch_lm_p": float(arch_test[1]),
                    "jb_stat": float(jb_stat), "jb_p": float(jb_p),
                    "alpha": alpha, "beta": beta, "alpha_beta_sum": alpha_beta_sum,
                })

                basefname = f"p{p}_q{q}_{dist}"
                plt.figure(figsize=(6,5))
                if dist == "t":
                    df_nu = None
                    params = garch_res.params
                    for name, val in params.items():
                        if name.lower().startswith("nu") or "df" in name.lower():
                            df_nu = float(val)
                            break

                    if df_nu is not None and df_nu > 2:
                        probplot(std_resid, dist=st.t(df_nu), plot=plt)
                        plt.title(f"Q-Q: {basefname} vs t(df={df_nu:.2f})")
                    else:
                        probplot(std_resid, dist="norm", plot=plt)
                        plt.title(f"Q-Q: {basefname} (fallback normal)")
                else:
                    probplot(std_resid, dist="norm", plot=plt)
                    plt.title(f"Q-Q: {basefname} vs Normal")
                plt.tight_layout()
                plt.savefig(os.path.join(DIAG_DIR, f"qq_{basefname}.png"))
                plt.close()

                fig, ax = plt.subplots(1,2, figsize=(12,4))
                plot_acf(std_resid, lags=40, ax=ax[0], title=f"ACF std_resid {basefname}")
                plot_acf(std_resid**2, lags=40, ax=ax[1], title=f"ACF std_resid^2 {basefname}")
                plt.tight_layout()
                plt.savefig(os.path.join(DIAG_DIR, f"acf_{basefname}.png"))
                plt.close()

    # Save results
    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values(["aic_combined", "bic_combined"], ascending=True)
    df_res.to_csv(os.path.join(OUT_DIR, "model_grid_results.csv"), index=False)
    print(f"\nGrid search finished. Results saved to {os.path.join(OUT_DIR, 'model_grid_results.csv')}")
    return df_res

if __name__ == "__main__":
    df_results = run_grid()
    print("\nTop models by AIC (combined):")
    print(df_results[["model", "aic_combined", "bic_combined", "alpha_beta_sum", "arch_lm_p", "jb_p"]].head(10))
