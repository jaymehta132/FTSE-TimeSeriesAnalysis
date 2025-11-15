import os
import logging
import json, yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import setupLogging, loadConfig, seedEverything
# from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.diagnostic import het_arch
from statsmodels.graphics.tsaplots import plot_acf
import scipy.stats as stats

logger = logging.getLogger(__name__)

def main():
    logDir = "logs"
    setupLogging(logDir=logDir, logLevel=logging.INFO)
    config = loadConfig()
    seedEverything(config.get("seed", 42))
    logger.info("Starting FTSE Time Series Analysis")

    # Load preprocessed data
    preprocessedDataPath = config["data"]["preprocessedDataPath"]
    data = pd.read_csv(preprocessedDataPath)
    logger.info(f"Loaded preprocessed data from {preprocessedDataPath} with shape {data.shape}.")
    # Set Date as datetime
    data['Date'] = pd.to_datetime(data['Date'])
    startDate = config.get("dates", {}).get("startDate")
    endDate = config.get("dates", {}).get("endDate")
    logger.info(f"Date range for analysis: {startDate} to {endDate}")
    data = data[(data['Date'] >= startDate) & (data['Date'] <= endDate)]
    logger.info(f"Data shape after filtering by date: {data.shape}")

    analysisResultsDir = config.get("results", {}).get("analysisResultsDir", "results/analysis")
    os.makedirs(analysisResultsDir, exist_ok=True)

    # Plot rolling statistics
    window_size = config.get("eda", {}).get("rollingWindowSize", 30)
    for col in ['Price', 'Returns', 'Log Returns']:
        rolling_mean = data[col].rolling(window=window_size).mean()
        rolling_std = data[col].rolling(window=window_size).std()

        plt.figure(figsize=(14, 7))
        plt.plot(data['Date'], data[col], label=f'Original {col}', color='blue', alpha=0.5)
        plt.plot(data['Date'], rolling_mean, label=f'Rolling Mean ({window_size})', color='red')
        plt.plot(data['Date'], rolling_std, label=f'Rolling Std Dev ({window_size})', color='green')
        plt.title(f'Rolling Mean and Standard Deviation of {col}')
        plt.legend()
        rollingStatsPath = os.path.join(analysisResultsDir, f'rolling_stats_{col.lower().replace(" ", "_")}.png')
        plt.savefig(rollingStatsPath)
        logger.info(f"Rolling statistics plot for {col} saved to {rollingStatsPath}.")
        # plt.show()


    series = data['Log Returns']*100  # Scale log returns
    # Fit a ARMA + GARCH model
    p = config.get("model", {}).get("arma_garch", {}).get("p")
    q = config.get("model", {}).get("arma_garch", {}).get("q")
    dist = config.get("model", {}).get("arma_garch", {}).get("dist")
    lags = config.get("model", {}).get("arma_garch", {}).get("lags")
    model = arch_model(
        series,
        vol='Garch', p=p, q=q,
        mean='AR', lags=lags, dist=dist
    )
    res = model.fit(disp='off')
    logger.info(f"Fitted ARMA({lags},0) + GARCH({p},{q}) model to log returns.")

    # Save model summary
    modelSummaryPath = os.path.join(analysisResultsDir, "arma_garch_model_summary.txt")
    with open(modelSummaryPath, "w") as f:
        f.write(res.summary().as_text())
    logger.info(f"ARMA + GARCH model summary saved to {modelSummaryPath}.")

    sns.set_style(style="whitegrid")

    # Conditional volatility plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=res.conditional_volatility, color='blue')
    plt.title('Conditional Volatility from GARCH Model')
    plt.savefig(os.path.join(analysisResultsDir, "conditional_volatility.png"))
    plt.close()
    logger.info("Generated conditional volatility plot.")

    # Std residuals plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=res.std_resid, color='red')
    plt.title('Standardized Residuals from GARCH Model')
    plt.savefig(os.path.join(analysisResultsDir, "standardized_residuals.png"))
    plt.close()
    logger.info("Generated standardized residuals plot.")

    clean_resid = res.std_resid.dropna()

    # Ljung-Box test for autocorrelation
    lb_test = acorr_ljungbox(clean_resid, lags=[10], return_df=True)
    logger.info("Performed Ljung-Box test on standardized residuals.")
    # Print Ljung-Box test results
    logger.info(f"Ljung-Box test results:\n{lb_test}")
    # Reject null hypothesis if p-value < 0.05
    if lb_test['lb_pvalue'].values[0] < 0.05:
        logger.info("Ljung-Box test indicates presence of autocorrelation in residuals.")
    else:
        logger.info("Ljung-Box test indicates no significant autocorrelation in residuals.")

    # ARCH test for remaining ARCH effects
    arch_test = het_arch(clean_resid)
    logger.info(f"ARCH test results: {arch_test}")
    # Reject null hypothesis if p-value < 0.05
    if arch_test[1] < 0.05:
        logger.info("ARCH test indicates presence of remaining ARCH effects in residuals.")
    else:
        logger.info("ARCH test indicates no significant remaining ARCH effects in residuals.")

    # Q-Q plot of residuals
    plt.figure(figsize=(8,6))
    stats.probplot(clean_resid, dist="t", sparams=(res.params['nu'],), plot=plt)
    plt.title("QQ Plot (Student-t)")
    plt.savefig(os.path.join(analysisResultsDir, "qq_plot_student_t.png"))
    plt.close() 
    logger.info("Generated QQ plot for standardized residuals.")

    # Histogram of residuals
    plt.figure(figsize=(10, 6))
    sns.histplot(clean_resid, bins=50, kde=True)
    plt.title("Histogram of Standardized Residuals")
    plt.savefig(os.path.join(analysisResultsDir, "residual_histogram.png"))
    plt.close()
    logger.info("Generated histogram for standardized residuals.")

    # ACF plots for residuals and squared residuals
    plt.figure(figsize=(10, 6))
    plot_acf(clean_resid, lags=40)
    plt.title("ACF of Standardized Residuals (Seaborn style)")
    plt.savefig(os.path.join(analysisResultsDir, "acf_std_resid.png"))
    plt.close()
    logger.info("Generated ACF plot for standardized residuals.")

    # ACF of squared residuals
    plt.figure(figsize=(10, 6))
    plot_acf(clean_resid**2, lags=40)
    plt.title("ACF of Squared Standardized Residuals (Seaborn style)")
    plt.savefig(os.path.join(analysisResultsDir, "acf_squared_std_resid.png"))
    plt.close()
    logger.info("Generated ACF plot for squared standardized residuals.")

    # Pairplot of residuals
    sns.pairplot(pd.DataFrame({"std_resid": clean_resid}))
    plt.savefig(os.path.join(analysisResultsDir, "pairplot_residuals.png"))
    plt.close()
    logger.info("Generated pairplot for standardized residuals.")



if __name__ == "__main__":
    main()