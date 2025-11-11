"""
Forecasting script for FTSE 100 Time Series Analysis
Generates multi-step forecasts from three candidate models:
1. ARMA(0,1) + GJR-GARCH(1,1) - Skewed-t
2. ARMA(0,1) + GJR-GARCH(1,1) - Student-t
3. ARMA(0,1) + GARCH(1,1) - Student-t
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
from scipy import stats
import logging
import yaml
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

logger = logging.getLogger(__name__)


def setup_logging(log_dir="logs", log_level=logging.INFO):
    """Setup logging configuration"""
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = log_path / f"forecasting_{timestamp}.log"
    log_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    logger_root = logging.getLogger()
    logger_root.setLevel(log_level)

    if logger_root.hasHandlers():
        logger_root.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger_root.addHandler(console_handler)

    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(log_format)
    logger_root.addHandler(file_handler)


def load_config(config_path="config.yaml"):
    """Load configuration file"""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def seed_everything(seed_value=42):
    """Set random seeds for reproducibility"""
    import random
    random.seed(seed_value)
    np.random.seed(seed_value)


class ForecastModel:
    """Wrapper for ARMA-GARCH model forecasting"""

    def __init__(self, name, mean_spec, vol_spec, dist):
        self.name = name
        self.mean_spec = mean_spec  # e.g., 'ARMA(0,1)'
        self.vol_spec = vol_spec    # e.g., 'GJR-GARCH(1,1)'
        self.dist = dist            # 't' or 'skewt'
        self.model = None
        self.result = None

    def fit(self, returns):
        """Fit the model to returns data"""
        logger.info(f"Fitting {self.name}...")

        # Parse mean specification
        if 'ARMA(0,1)' in self.mean_spec:
            mean_model = 'MA'
            lags_q = 1
        else:
            mean_model = 'Zero'
            lags_q = 0

        # Parse volatility specification
        if 'GJR' in self.vol_spec:
            vol_model = 'Garch'
            p, q, o = 1, 1, 1  # GJR-GARCH(1,1) with asymmetry
        else:
            vol_model = 'Garch'
            p, q, o = 1, 1, 0  # Standard GARCH(1,1)

        try:
            # Create and fit model
            if mean_model == 'MA':
                self.model = arch_model(
                    returns * 100,  # Scale returns
                    mean='ARX',
                    lags=0,
                    vol=vol_model,
                    p=p, o=o, q=q,
                    dist=self.dist
                )
            else:
                self.model = arch_model(
                    returns * 100,
                    mean='Zero',
                    vol=vol_model,
                    p=p, o=o, q=q,
                    dist=self.dist
                )

            self.result = self.model.fit(disp='off', show_warning=False)
            logger.info(f"Successfully fitted {self.name}")
            logger.info(f"  Log-likelihood: {self.result.loglikelihood:.2f}")
            logger.info(f"  AIC: {self.result.aic:.2f}")

            return True

        except Exception as e:
            logger.error(f"Failed to fit {self.name}: {e}")
            return False

    def forecast(self, horizon):
        """
        Generate forecasts for specified horizon
        Returns: dict with mean, variance, and quantiles
        """
        if self.result is None:
            raise ValueError("Model must be fitted before forecasting")

        # Generate forecasts
        forecasts = self.result.forecast(horizon=horizon, reindex=False)

        # Extract mean and variance forecasts
        mean_forecast = forecasts.mean.values[-1, :]  # Last row is the forecast
        variance_forecast = forecasts.variance.values[-1, :]

        # Get distribution parameters for confidence intervals
        if self.dist == 't':
            # Student-t distribution
            nu = self.result.params.get('nu', 10)  # degrees of freedom

            # Compute quantiles (95% CI)
            lower_q = stats.t.ppf(0.025, nu)
            upper_q = stats.t.ppf(0.975, nu)

        elif self.dist == 'skewt':
            # Skewed-t: approximate with normal for simplicity
            # In practice, would use skewed-t quantiles
            lower_q = -1.96
            upper_q = 1.96
        else:
            # Normal
            lower_q = -1.96
            upper_q = 1.96

        # Compute confidence intervals
        std_forecast = np.sqrt(variance_forecast)
        lower_bound = mean_forecast + lower_q * std_forecast
        upper_bound = mean_forecast + upper_q * std_forecast

        # Scale back (we scaled by 100 earlier)
        return {
            'mean': mean_forecast / 100,
            'variance': variance_forecast / 10000,
            'std': std_forecast / 100,
            'lower_95': lower_bound / 100,
            'upper_95': upper_bound / 100
        }


def load_data(config):
    """Load preprocessed FTSE data"""
    data_path = config['data']['preprocessedDataPath']
    df = pd.read_csv(data_path, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Filter to 2005-2007
    start_date = config['dates']['startDate']
    end_date = config['dates']['endDate']
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    logger.info(f"Loaded {len(df)} observations from {start_date} to {end_date}")
    return df


def create_comparative_plot(forecasts_dict, horizons, output_dir):
    """Create comparative forecast plots for all models"""

    # Plot volatility forecasts
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Volatility Forecasts: Model Comparison', fontsize=18, fontweight='bold')

    colors = {'GJR-GARCH-Skewed-t': '#e74c3c',
              'GJR-GARCH-t': '#3498db',
              'GARCH-t': '#2ecc71'}

    for idx, horizon in enumerate(horizons):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]

        for model_name, forecasts in forecasts_dict.items():
            forecast_data = forecasts[horizon]
            steps = np.arange(1, horizon + 1)

            # Plot volatility (std dev) in percentage
            vol_pct = forecast_data['std'] * 100
            ax.plot(steps, vol_pct,
                   label=model_name, color=colors[model_name],
                   linewidth=2.5, marker='o', markersize=7)

        ax.set_xlabel('Days Ahead', fontsize=12, fontweight='bold')
        ax.set_ylabel('Volatility (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'{horizon}-Day Ahead Volatility Forecast', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Set y-axis limits for better visibility
        all_vols = []
        for model_name, forecasts in forecasts_dict.items():
            all_vols.extend(forecasts[horizon]['std'] * 100)
        y_min, y_max = min(all_vols), max(all_vols)
        y_margin = (y_max - y_min) * 0.15
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparative_volatility_forecasts.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved comparative volatility forecast plot")


def create_individual_plots(model_name, forecasts, horizons, output_dir):
    """Create individual forecast plots with confidence bands for one model"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{model_name}: Multi-Horizon Return Forecasts', fontsize=16, fontweight='bold')

    for idx, horizon in enumerate(horizons):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]

        forecast_data = forecasts[horizon]
        steps = np.arange(1, horizon + 1)

        # Convert to basis points (1 bp = 0.01% = 0.0001)
        mean_bps = forecast_data['mean'] * 10000
        lower_bps = forecast_data['lower_95'] * 10000
        upper_bps = forecast_data['upper_95'] * 10000

        # Plot mean forecast with confidence bands
        ax.plot(steps, mean_bps,
               label='Mean Forecast', color='#2c3e50', linewidth=2.5, marker='o', markersize=6)
        ax.fill_between(steps, lower_bps, upper_bps,
                        alpha=0.25, color='#3498db', label='95% Confidence Interval')

        ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
        ax.set_xlabel('Days Ahead', fontsize=12, fontweight='bold')
        ax.set_ylabel('Expected Return (basis points)', fontsize=12, fontweight='bold')
        ax.set_title(f'{horizon}-Day Ahead Forecast', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Set appropriate y-axis limits for better visualization
        y_range = max(abs(lower_bps.min()), abs(upper_bps.max()))
        ax.set_ylim(-y_range * 1.1, y_range * 1.1)

    plt.tight_layout()
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')
    plt.savefig(os.path.join(output_dir, f'{safe_name}_forecasts.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved individual forecast plot for {model_name}")


def save_forecast_results(forecasts_dict, horizons, output_dir):
    """Save forecast results to CSV files"""

    all_results = []

    for model_name, forecasts in forecasts_dict.items():
        for horizon in horizons:
            forecast_data = forecasts[horizon]

            for step in range(horizon):
                all_results.append({
                    'model': model_name,
                    'horizon': horizon,
                    'step': step + 1,
                    'mean_forecast': forecast_data['mean'][step],
                    'variance_forecast': forecast_data['variance'][step],
                    'std_forecast': forecast_data['std'][step],
                    'lower_95': forecast_data['lower_95'][step],
                    'upper_95': forecast_data['upper_95'][step]
                })

    # Save combined results
    df_results = pd.DataFrame(all_results)
    csv_path = os.path.join(output_dir, 'all_forecasts.csv')
    df_results.to_csv(csv_path, index=False)
    logger.info(f"Saved all forecasts to {csv_path}")

    # Save summary statistics
    summary = df_results.groupby(['model', 'horizon']).agg({
        'mean_forecast': ['mean', 'std'],
        'variance_forecast': ['mean', 'max'],
        'std_forecast': ['mean', 'max']
    }).reset_index()

    summary_path = os.path.join(output_dir, 'forecast_summary.csv')
    summary.to_csv(summary_path, index=False)
    logger.info(f"Saved forecast summary to {summary_path}")

    return df_results


def main():
    # Setup
    log_dir = "logs"
    setup_logging(log_dir=log_dir, log_level=logging.INFO)
    config = load_config()
    seed_everything(config.get("seed", 42))

    logger.info("=" * 80)
    logger.info("FTSE 100 Time Series Forecasting")
    logger.info("=" * 80)

    # Create output directory
    output_dir = "results/forecasting"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df = load_data(config)
    returns = df['Log Returns'].values

    # IMPORTANT: Models are trained on the FULL dataset (2005-2007)
    # This is for in-sample forecasting demonstration purposes only
    # Out-of-sample testing will use rolling windows in backtesting.py
    logger.info(f"Training on FULL dataset: {len(returns)} observations (2005-2007)")

    # Define forecast horizons
    horizons = [1, 5, 10, 20]
    logger.info(f"Forecast horizons: {horizons}")

    # Define three candidate models
    models = [
        ForecastModel(
            name='GJR-GARCH-Skewed-t',
            mean_spec='ARMA(0,1)',
            vol_spec='GJR-GARCH(1,1)',
            dist='skewt'
        ),
        ForecastModel(
            name='GJR-GARCH-t',
            mean_spec='ARMA(0,1)',
            vol_spec='GJR-GARCH(1,1)',
            dist='t'
        ),
        ForecastModel(
            name='GARCH-t',
            mean_spec='ARMA(0,1)',
            vol_spec='GARCH(1,1)',
            dist='t'
        )
    ]

    # Fit models and generate forecasts
    forecasts_dict = {}

    for model in models:
        logger.info("\n" + "-" * 80)

        # Fit model
        success = model.fit(returns)
        if not success:
            logger.warning(f"Skipping {model.name} due to fitting failure")
            continue

        # Generate forecasts for each horizon
        model_forecasts = {}
        for horizon in horizons:
            logger.info(f"Generating {horizon}-day ahead forecast for {model.name}...")
            forecast_data = model.forecast(horizon)
            model_forecasts[horizon] = forecast_data

            # Log summary
            logger.info(f"  Mean return (1-day): {forecast_data['mean'][0]*10000:.2f} bps")
            logger.info(f"  Volatility (1-day): {forecast_data['std'][0]*100:.4f}%")
            if horizon > 1:
                logger.info(f"  Mean return ({horizon}-day): {forecast_data['mean'][-1]*10000:.2f} bps")
                logger.info(f"  Volatility ({horizon}-day): {forecast_data['std'][-1]*100:.4f}%")

        forecasts_dict[model.name] = model_forecasts

    # Save results
    logger.info("\n" + "=" * 80)
    logger.info("Saving results...")
    logger.info("=" * 80)

    df_results = save_forecast_results(forecasts_dict, horizons, output_dir)

    # Create visualizations
    logger.info("\nCreating visualizations...")
    create_comparative_plot(forecasts_dict, horizons, output_dir)

    for model_name, forecasts in forecasts_dict.items():
        create_individual_plots(model_name, forecasts, horizons, output_dir)

    logger.info("\n" + "=" * 80)
    logger.info("Forecasting completed successfully!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 80)

    # Print summary comparison
    print("\n" + "=" * 80)
    print("FORECAST SUMMARY COMPARISON (1-Day Ahead)")
    print("=" * 80)

    summary = df_results[df_results['step'] == 1].groupby('model').agg({
        'mean_forecast': 'mean',
        'std_forecast': 'mean'
    }).copy()

    # Convert returns to basis points for display
    summary['mean_return_bps'] = summary['mean_forecast'] * 10000
    summary['volatility_pct'] = summary['std_forecast'] * 100
    summary_display = summary[['mean_return_bps', 'volatility_pct']]
    summary_display.columns = ['Mean Return (bps)', 'Volatility (%)']

    print("\n")
    print(summary_display.round(2).to_string())
    print("\n" + "=" * 80)
    print("Note: Returns shown in basis points (1 bp = 0.01%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
