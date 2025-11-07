import os
import pandas
import numpy as np
import logging
from src.utils import setupLogging, loadConfig, seedEverything
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

def main():
    logDir = "logs"
    setupLogging(logDir=logDir, logLevel=logging.INFO)
    config = loadConfig()
    seedEverything(config.get("seed", 42))
    logger.info("Data preprocessing started.")

    # Load the dataset
    rawDataPath = config["data"]["rawDataPath"]
    data = pandas.read_csv(rawDataPath)
    logger.info(f"Loaded raw data from {rawDataPath} with shape {data.shape}.")

    # Print head of the dataset
    logger.info("First 5 rows of the dataset:")
    logger.info(f"\n{data.head()}")


    # Check for missing values
    missing_values = data.isnull().sum()
    logger.info("Missing values in each column:")
    logger.info(f"\n{missing_values[missing_values > 0]}")

    # Remove "Vol." column
    if "Vol." in data.columns:
        data = data.drop(columns=["Vol."])
        logger.info('"Vol." column removed from the dataset.')

    logger.info("First 5 rows of the dataset:")
    logger.info(f"\n{data.head()}")

    # Convert 'Date' column to datetime
    if 'Date' in data.columns:
        data['Date'] = pandas.to_datetime(data['Date'], format='%d/%m/%Y')
        logger.info("'Date' column converted to datetime format.")

    # Sort according to Date
    data = data.sort_values(by='Date')
    logger.info("First 5 rows of the dataset:")
    logger.info(f"\n{data.head()}")

    # Remove 'High', 'Low', 'Open', 'Change %' columns
    columns_to_remove = ['High', 'Low', 'Open', 'Change %']
    data = data.drop(columns=columns_to_remove, errors='ignore')
    logger.info(f"Removed columns: {columns_to_remove}")

    logger.info("First 5 rows of the dataset:")
    logger.info(f"\n{data.head()}")

    # Print Data Types
    logger.info("Data types of each column:")
    logger.info(f"\n{data.dtypes}")

    # Convert Price columns to float
    price_columns = ['Price']
    for col in price_columns:
        if col in data.columns:
            data[col] = data[col].str.replace(',', '').astype(float)
            logger.info(f"Converted '{col}' column to float.")
    logger.info("First 5 rows of the dataset:")
    logger.info(f"\n{data.head()}")
    # Print Data Types after conversion
    logger.info("Data types of each column after conversion:")
    logger.info(f"\n{data.dtypes}")

    # Now Calculate daily Returns as R_t = (Price_t - Price_(t-1)) / Price_(t-1)
    data['Returns'] = data['Price'].pct_change()  # Returns in percentage
    data['Log Returns'] = np.log(data['Returns'] + 1)  # Log Returns in percentage
    # Remove row with NaN Returns (first row)
    data = data.dropna(subset=['Returns'])
    logger.info("'Returns' column added to the dataset.")
    logger.info("First 5 rows of the dataset:")
    logger.info(f"\n{data.head()}")

    # Print Last 5 rows of the dataset
    logger.info("Last 5 rows of the dataset:")
    logger.info(f"\n{data.tail()}")

    # Save Dataset in data/processed/
    preprocessedDataPath = config["data"]["preprocessedDataPath"]
    data.to_csv(preprocessedDataPath, index=False)
    logger.info(f"Processed data saved to {preprocessedDataPath}")


    resultsDir = config["results"]["resultsDir"]
    os.makedirs(resultsDir, exist_ok=True)


    # Plot the Price and Returns
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=data, x='Date', y='Price', label='Price')
    # sns.lineplot(data=data, x='Date', y='Returns', label='Returns')
    plt.title('Price Over Time')
    plt.legend()
    plt.savefig(os.path.join(resultsDir, 'price_over_time.png'))
    # plt.show()

    plt.figure(figsize=(14, 7))
    sns.lineplot(data=data, x='Date', y='Log Returns', label='Log Returns', color='orange')
    plt.title('Log Returns Over Time')
    plt.legend()
    plt.savefig(os.path.join(resultsDir, 'log_returns_over_time.png'))
    # plt.show()

    plt.figure(figsize=(14,7))
    sns.histplot(data['Log Returns'], bins=100, kde=True)
    plt.title('Log Returns Distribution')
    plt.savefig(os.path.join(resultsDir, 'log_returns_distribution.png'))
    # plt.show()

    logger.info("Data preprocessing completed.")

if __name__ == "__main__":
    main()