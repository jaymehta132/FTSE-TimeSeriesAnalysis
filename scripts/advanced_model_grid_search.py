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
from arch import arch_model

from tqdm import tqdm

sns.set(style="whitegrid")
warnings.filterwarnings("ignore")

DATA_PATH = "../data/FTSE_PreprocessedData.csv"
OUT_DIR = "../results/model_grid_advanced_outputs"
DIAG_DIR = os.path.join(OUT_DIR, "diagnostics")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DIAG_DIR, exist_ok=True)

START_DATE = "2005-01-01"
END_DATE   = "2007-12-31"

ARMA_ORDER = (0, 0, 1) # Mean: ARMA(0,1) (i.e. ARIMA(0,0,1))

# Volatility model search space
GARCH_PQ = [(1,1), (1,2), (2,1), (2,2)]
VOL_MODELS = [
    ("GARCH", None),        # regular GARCH(p,q)
    ("GJR", {"o": 1}),      # GJR-GARCH via o=1
    ("EGARCH", None),       # EGARCH(1,1) (we will use p=1,q=1)
    ("ARCH", None),         # ARCH(q)
]
DISTS = ["t", "skewt"]     # distributions to try

LB_LAGS = [10, 20, 40]
ARCH_NLAGS = 12

def load_returns(path, prefer_col="Log Returns"):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    start, end = pd.Timestamp(START_DATE), pd.Timestamp(END_DATE)
    df = df[(df["Date"] >= start) & (df["Date"] <= end)].reset_index(drop=True)
    col = None
    for c in [prefer_col, "Log Returns", "Returns", "Return", "logret", "log_return"]:
        if c in df.columns:
            col = c
            break
    if col is None:
        raise ValueError(f"No returns column found. Available: {df.columns.tolist()}")
    series = df[col].astype(float).dropna()
    series.index = pd.to_datetime(df.loc[series.index, "Date"])
    return series, col

def log_likelihood(res):
    for attr in ("llf", "loglikelihood", "loglik", "loglikelihood_value", "loglikelihood_"):
        if hasattr(res, attr):
            return float(getattr(res, attr))

    return None

def count_params(res):
    return len(res.params)

def extract_alpha_beta_from_garch(garch_res):
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

def run_advanced_grid():
    returns, used_col = load_returns(DATA_PATH)
    print(f"Using {used_col} with {len(returns)} obs from {START_DATE} to {END_DATE}.")

    results = []
    n = len(returns)

    print("Fitting ARMA mean (0,1)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arma = ARIMA(returns, order=ARMA_ORDER)
        arma_res = arma.fit(method_kwargs={"warn_convergence": False})
    resid = arma_res.resid.dropna()

    for vol_name, vol_opts in tqdm(VOL_MODELS, desc="Vol models"):
        if vol_name == "EGARCH":
            for dist in DISTS:
                model_label = f"ARMA(0,1)+EGARCH(1,1)-{dist}"
                print(f"\nFitting {model_label} ...")
                am = arch_model(resid, mean="Zero", vol="EGARCH", p=1, o=1, q=1, dist=dist)
                gres = am.fit(disp="off")
                results.extend(process_fit(model_label, arma_res, gres, resid, dist))
        elif vol_name == "GJR":
            for (p,q) in GARCH_PQ:
                for dist in DISTS:
                    model_label = f"ARMA(0,1)+GJR-GARCH({p},{q})-{dist}"
                    print(f"\nFitting {model_label} ...")
                    am = arch_model(resid, mean="Zero", vol="Garch", p=p, o=1, q=q, dist=dist)
                    gres = am.fit(disp="off")
                    results.extend(process_fit(model_label, arma_res, gres, resid, dist))
        elif vol_name == "GARCH":
            for (p,q) in GARCH_PQ:
                for dist in DISTS:
                    model_label = f"ARMA(0,1)+GARCH({p},{q})-{dist}"
                    print(f"\nFitting {model_label} ...")
                    am = arch_model(resid, mean="Zero", vol="Garch", p=p, q=q, dist=dist)
                    gres = am.fit(disp="off")
                    results.extend(process_fit(model_label, arma_res, gres, resid, dist))
        elif vol_name == "ARCH":
            for q in [1,2,4,8]:
                for dist in DISTS:
                    model_label = f"ARMA(0,1)+ARCH({q})-{dist}"
                    print(f"\nFitting {model_label} ...")
                    am = arch_model(resid, mean="Zero", vol="ARCH", p=q, q=0, dist=dist)
                    gres = am.fit(disp="off")
                    
                    results.extend(process_fit(model_label, arma_res, gres, resid, dist))
        else:
            pass

    df = pd.DataFrame(results)
    df = df.sort_values(["aic_combined", "bic_combined"], ascending=True)
    out_csv = os.path.join(OUT_DIR, "model_grid_advanced_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved results to {out_csv}")
    return df

def process_fit(model_label, arma_res, garch_res, resid, dist_label):
    out = []
    n = len(resid)
    if garch_res is None:
        out.append({
            "model": model_label,
            "n_obs": n,
            "ll_arima": log_likelihood(arma_res),
            "ll_garch": np.nan,
            "ll_combined": np.nan,
            "k_params": np.nan,
            "aic_combined": np.nan,
            "bic_combined": np.nan,
            **{f"lb_res_p_lag{lag}": np.nan for lag in LB_LAGS},
            **{f"lb_sq_p_lag{lag}": np.nan for lag in LB_LAGS},
            "arch_lm_stat": np.nan, "arch_lm_p": np.nan,
            "jb_stat": np.nan, "jb_p": np.nan,
            "alpha": np.nan, "beta": np.nan, "alpha_beta_sum": np.nan
        })
        return out

    try:
        std_resid = garch_res.std_resid.dropna()
    except Exception:
        cond_vol = getattr(garch_res, "conditional_volatility", None)
        if cond_vol is None:
            cond_vol = garch_res.conditional_volatility
        std_resid = (resid / cond_vol).dropna()

    ll_arima = log_likelihood(arma_res)
    ll_garch = log_likelihood(garch_res)
    combined_ll = None
    if (ll_arima is not None) and (ll_garch is not None):
        combined_ll = ll_arima + ll_garch

    k_arima = count_params(arma_res) or 0
    k_garch = count_params(garch_res) or 0
    k_total = k_arima + k_garch

    aic_combined = np.nan
    bic_combined = np.nan
    if combined_ll is not None and k_total > 0:
        aic_combined = -2.0 * combined_ll + 2.0 * k_total
        bic_combined = -2.0 * combined_ll + k_total * np.log(n)

    # Ljung-Box on std_resid & squared
    lb_res = acorr_ljungbox(std_resid, lags=LB_LAGS, return_df=True)
    lb_sq = acorr_ljungbox(std_resid**2, lags=LB_LAGS, return_df=True)


    # ARCH-LM
    arch_test = het_arch(std_resid, nlags=ARCH_NLAGS)
    arch_stat = float(arch_test[0])
    arch_p = float(arch_test[1])

    # JB
    jb_stat, jb_p = jarque_bera(std_resid)[:2]

    # alpha, beta
    alpha, beta = extract_alpha_beta_from_garch(garch_res)
    alpha_beta_sum = np.nan
    alpha_beta_sum = float(alpha) + float(beta)

    base = model_label.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "").replace("+", "")
    plt.figure(figsize=(6,5))
    if dist_label == "t":
        df_nu = None
        for nm, val in garch_res.params.items():
            if str(nm).lower().startswith("nu") or "df" in str(nm).lower():
                df_nu = float(val); break

        if df_nu is not None and df_nu > 2:
            import scipy.stats as st
            probplot(std_resid, dist=st.t(df_nu), plot=plt)
            plt.title(f"Q-Q: {base} vs t(df={df_nu:.2f})")
        else:
            probplot(std_resid, dist="norm", plot=plt)
            plt.title(f"Q-Q: {base} (fallback normal)")
    elif dist_label == "skewt":
        probplot(std_resid, dist="norm", plot=plt)
        plt.title(f"Q-Q: {base} vs Normal (skewt used)")
    else:
        probplot(std_resid, dist="norm", plot=plt)
        plt.title(f"Q-Q: {base} vs Normal")
    plt.tight_layout()
    plt.savefig(os.path.join(DIAG_DIR, f"qq_{base}.png"))
    plt.close()

    fig, ax = plt.subplots(1,2, figsize=(12,4))
    plot_acf(std_resid, lags=40, ax=ax[0], title=f"ACF_stdresid_{base}")
    plot_acf(std_resid**2, lags=40, ax=ax[1], title=f"ACF_stdresid2_{base}")
    plt.tight_layout()
    plt.savefig(os.path.join(DIAG_DIR, f"acf_{base}.png"))
    plt.close()


    record = {
        "model": model_label,
        "n_obs": n,
        "ll_arima": ll_arima,
        "ll_garch": ll_garch,
        "ll_combined": combined_ll,
        "k_params": k_total,
        "aic_combined": aic_combined,
        "bic_combined": bic_combined,
        # lb p-values
        **({f"lb_res_p_lag{lag}": float(lb_res["lb_pvalue"].loc[lag]) for lag in LB_LAGS} if lb_res is not None else {f"lb_res_p_lag{lag}": np.nan for lag in LB_LAGS}),
        **({f"lb_sq_p_lag{lag}": float(lb_sq["lb_pvalue"].loc[lag]) for lag in LB_LAGS} if lb_sq is not None else {f"lb_sq_p_lag{lag}": np.nan for lag in LB_LAGS}),
        "arch_lm_stat": arch_stat, "arch_lm_p": arch_p,
        "jb_stat": jb_stat, "jb_p": jb_p,
        "alpha": alpha, "beta": beta, "alpha_beta_sum": alpha_beta_sum
    }

    out.append(record)
    return out

if __name__ == "__main__":
    df_out = run_advanced_grid()
    print("\nTop models by AIC (combined):")
    print(df_out[["model", "aic_combined", "bic_combined", "alpha_beta_sum", "arch_lm_p", "jb_p"]].head(10))
