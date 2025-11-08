import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os 
import yaml
from src.utils import setupLogging, loadConfig, seedEverything
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

logger = logging.getLogger(__name__)

def main():
    logDir = "logs"
    setupLogging(logDir=logDir, logLevel=logging.INFO)
    config = loadConfig()
    seedEverything(config.get("seed", 42))
    logger.info("Exploratory Data Analysis (EDA) started.")
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


    edaResultsDir = config.get("results", {}).get("edaResultsDir", "results/eda")
    os.makedirs(edaResultsDir, exist_ok=True)
    sns.set_style("whitegrid")

    # Plot the Price and Returns
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=data, x='Date', y='Price', label='Price')
    # sns.lineplot(data=data, x='Date', y='Returns', label='Returns')
    plt.title('Price Over Time')
    plt.legend()
    plt.savefig(os.path.join(edaResultsDir, 'price_over_time.png'))
    logger.info("Price over time plot saved.")
    # plt.show()

    plt.figure(figsize=(14, 7))
    sns.lineplot(data=data, x='Date', y='Log Returns', label='Log Returns', color='orange')
    plt.title('Log Returns Over Time')
    plt.legend()
    plt.savefig(os.path.join(edaResultsDir, 'log_returns_over_time.png'))
    logger.info("Log Returns over time plot saved.")
    # plt.show()

    plt.figure(figsize=(14,7))
    sns.histplot(data['Log Returns'], bins=100, kde=True)
    plt.title('Log Returns Distribution')
    plt.savefig(os.path.join(edaResultsDir, 'log_returns_distribution.png'))
    logger.info("Log Returns distribution plot saved.")
    # plt.show()

    plt.figure(figsize=(14,7))
    sns.lineplot(data=data, x='Date', y='Returns', label='Returns', color='green')
    plt.title('Returns Over Time')
    plt.legend()
    plt.savefig(os.path.join(edaResultsDir, 'returns_over_time.png'))
    logger.info("Returns over time plot saved.")

    plt.figure(figsize=(14,7))
    sns.histplot(data['Returns'], bins=100, kde=True)
    plt.title('Returns Distribution')
    plt.savefig(os.path.join(edaResultsDir, 'returns_distribution.png'))
    logger.info("Returns distribution plot saved.")


    # Mean, Median, Std, Skewness, Kurtosis for Returns and Log Returns
    statistics = {}
    for col in ['Returns', 'Log Returns']:
        statistics[col] = {
            'Mean': float(np.mean(data[col])),
            'Median': float(np.median(data[col])),
            'Std Dev': float(np.std(data[col])),
            'Skewness': float(data[col].skew()),
            'Excess Kurtosis': float(data[col].kurtosis()),
            'Kurtosis': float(data[col].kurtosis()) + 3
        }
        logger.info(f"Statistics for {col}: {statistics[col]}")
    # Save statistics to a YAML file
    statsPath = os.path.join(edaResultsDir, 'eda_statistics.yaml')
    with open(statsPath, 'w') as file:
        yaml.dump(statistics, file)
    logger.info(f"EDA statistics saved to {statsPath}.")

    # Q-Q Plots vs Normal Distribution
    for col in ['Returns', 'Log Returns']:
        plt.figure(figsize=(8, 8))
        stats.probplot(data[col], dist="norm", plot=plt)
        plt.title(f'Q-Q Plot of {col} vs Normal Distribution')
        qqPlotPath = os.path.join(edaResultsDir, f'qq_plot_{col.lower().replace(" ", "_")}_normal.png')
        plt.savefig(qqPlotPath)
        logger.info(f"Q-Q plot of {col} vs Normal distribution saved to {qqPlotPath}.")
        # plt.show()

    # Q-Q Plots vs t-Distribution
    for col in ['Returns', 'Log Returns']:
        plt.figure(figsize=(8, 8))
        stats.probplot(data[col], dist="t", sparams=(5,), plot=plt)  # Using df=5 for t-distribution
        plt.title(f'Q-Q Plot of {col} vs t-Distribution (df=5)')
        qqPlotPath = os.path.join(edaResultsDir, f'qq_plot_{col.lower().replace(" ", "_")}_tdist.png')
        plt.savefig(qqPlotPath)
        logger.info(f"Q-Q plot of {col} vs t-Distribution saved to {qqPlotPath}.")
        # plt.show()

    # ACF PACF plots for Returns and Log Returns
    for col in ['Returns', 'Log Returns']:
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        plot_acf(data[col], ax=plt.gca(), lags=40)
        plt.title(f'ACF of {col}')
        plt.subplot(1, 2, 2)
        plot_pacf(data[col], ax=plt.gca(), lags=40, method='ywm')
        plt.title(f'PACF of {col}')
        acfPacfPath = os.path.join(edaResultsDir, f'acf_pacf_{col.lower().replace(" ", "_")}.png')
        plt.savefig(acfPacfPath)
        logger.info(f"ACF and PACF plots of {col} saved to {acfPacfPath}.")
        # plt.show()

    

    logger.info("Exploratory Data Analysis (EDA) completed.")


if __name__ == "__main__":
    main()