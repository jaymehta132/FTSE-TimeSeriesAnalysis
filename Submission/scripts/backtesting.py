import os
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
from tqdm import tqdm
import random

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

logger = logging.getLogger(__name__)


def setup_logging(log_dir="logs", log_level=logging.INFO):
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = log_path / f"backtesting_{timestamp}.log"
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
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def seed_everything(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)


def load_data(config):
    data_path = config['data']['preprocessedDataPath']
    df = pd.read_csv(data_path, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    start_date = config['dates']['startDate']
    end_date = config['dates']['endDate']
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    logger.info(f"Loaded {len(df)} observations from {start_date} to {end_date}")
    return df


class ARMAGARCHModel:
    def __init__(self, name, vol_type, dist):
        self.name = name
        self.vol_type = vol_type  # 'garch' or 'gjr-garch'
        self.dist = dist  # 't' or 'skewt'
        self.model = None
        self.result = None

    def fit(self, returns):
        if 'GJR' in self.vol_type:
            vol_model = 'Garch'
            p, q, o = 1, 1, 1  # GJR-GARCH(1,1)
        else:
            vol_model = 'Garch'
            p, q, o = 1, 1, 0  # GXRCH(1,1)
        self.model = arch_model(
            returns * 100, 
            mean='ARX',
            lags=0,
            vol=vol_model,
            p=p, o=o, q=q,
            dist=self.dist
        )

        self.result = self.model.fit(disp='off', show_warning=False)
        return True

    def forecast_multi_step(self, horizon):
        forecasts = self.result.forecast(horizon=horizon, reindex=False)

        mean_forecast = forecasts.mean.values[-1, :]  
        variance_forecast = forecasts.variance.values[-1, :]

        return {
            'mean': mean_forecast / 100,
            'variance': variance_forecast / 10000
        }



def compute_metrics(actual_returns, forecast_returns, actual_variance_proxy, forecast_variance):
    metrics = {}

    errors = actual_returns - forecast_returns

    metrics['rmse'] = np.sqrt(np.mean(errors ** 2))
    metrics['mae'] = np.mean(np.abs(errors))

    correct_direction = np.sum(np.sign(actual_returns) == np.sign(forecast_returns))
    metrics['direction_accuracy'] = correct_direction / len(actual_returns)

    vol_errors = actual_variance_proxy - forecast_variance

    metrics['mse_vol'] = np.mean(vol_errors ** 2)

    valid_idx = (actual_variance_proxy > 1e-10) & (forecast_variance > 1e-10)
    if np.sum(valid_idx) > 0:
        ratio = actual_variance_proxy[valid_idx] / forecast_variance[valid_idx]
        qlike = ratio - np.log(ratio) - 1
        metrics['qlike'] = np.mean(qlike)
    else:
        metrics['qlike'] = np.nan

    return metrics


def rolling_window_backtest_optimized(returns, train_size, test_sizes, models):
    max_test_size = max(test_sizes)
    n = len(returns)
    n_windows = n - train_size - max_test_size + 1

    if n_windows <= 0:
        logger.warning(f"Not enough data for train_size={train_size}, max_test_size={max_test_size}")
        return {}

    results = {}
    for model in models:
        for test_size in test_sizes:
            results[(model.name, test_size)] = []

    ct=0
    for start_idx in range(n_windows):
        train_end = start_idx + train_size

        train_data = returns[start_idx:train_end]
        if ct==0:
            print(train_data.shape)
            ct+=1
        for model in models:
            success = model.fit(train_data)
            if not success:
                continue

            for test_size in test_sizes:
                test_end = train_end + test_size
                if test_end > n:
                    continue

                test_data = returns[train_end:test_end]

                forecasts = model.forecast_multi_step(test_size)
                if forecasts is None:
                    continue

                forecast_returns = forecasts['mean'][:len(test_data)]
                forecast_variance = forecasts['variance'][:len(test_data)]

                actual_variance_proxy = test_data ** 2

                metrics = compute_metrics(
                    test_data,
                    forecast_returns,
                    actual_variance_proxy,
                    forecast_variance
                )

                results[(model.name, test_size)].append(metrics)

    return results


def aggregate_metrics(results):
    aggregated = {}

    for (model_name, test_size), metrics_list in results.items():
        if len(metrics_list) == 0:
            aggregated[(model_name, test_size)] = {
                'rmse': np.nan,
                'mae': np.nan,
                'direction_accuracy': np.nan,
                'mse_vol': np.nan,
                'qlike': np.nan,
                'n_windows': 0
            }
            continue

        metrics_dict = {key: [] for key in metrics_list[0].keys()}
        for m in metrics_list:
            for key, val in m.items():
                metrics_dict[key].append(val)

        aggregated[(model_name, test_size)] = {
            key: np.nanmean(vals) for key, vals in metrics_dict.items()
        }
        aggregated[(model_name, test_size)]['n_windows'] = len(metrics_list)

    return aggregated


def run_backtesting_grid(df, train_sizes, test_sizes, model_specs):
    returns = df['Log Returns'].values
    all_results = []

    logger.info(f"Running optimized backtesting across {len(train_sizes)} training window sizes...")
    logger.info(f"\nmFor each window, forecasting {len(test_sizes)} horizons: {test_sizes}")

    for train_idx, train_size in enumerate(train_sizes, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Train size {train_idx}/{len(train_sizes)}: {train_size} days")
        logger.info(f"{'='*80}")

        models = [
            ARMAGARCHModel(name, vol_type, dist)
            for name, vol_type, dist in model_specs
        ]

        results = rolling_window_backtest_optimized(returns, train_size, test_sizes, models)

        agg_results = aggregate_metrics(results)

        for (model_name, test_size), metrics in agg_results.items():
            all_results.append({
                'model': model_name,
                'train_size': train_size,
                'test_size': test_size,
                'n_windows': metrics.get('n_windows', 0),
                'rmse': metrics.get('rmse', np.nan),
                'mae': metrics.get('mae', np.nan),
                'direction_accuracy': metrics.get('direction_accuracy', np.nan),
                'mse_vol': metrics.get('mse_vol', np.nan),
                'qlike': metrics.get('qlike', np.nan)
            })

        n_windows = agg_results.get((model_specs[0][0], test_sizes[0]), {}).get('n_windows', 0)
        logger.info(f"Completed {n_windows} rolling windows for train_size={train_size}")
        logger.info(f"Generated forecasts for {len(test_sizes)} horizons from each window")

    return pd.DataFrame(all_results)


def create_heatmaps(df_results, output_dir):
    metrics = ['rmse', 'mae', 'direction_accuracy', 'mse_vol', 'qlike']
    metric_labels = {
        'rmse': 'RMSE (Returns)',
        'mae': 'MAE (Returns)',
        'direction_accuracy': 'Direction Accuracy (%)',
        'mse_vol': 'MSE (Volatility)',
        'qlike': 'QLIKE (Volatility)'
    }

    test_sizes = sorted(df_results['test_size'].unique())
    models = sorted(df_results['model'].unique())

    for metric in metrics:
        for test_size in test_sizes:
            subset = df_results[df_results['test_size'] == test_size]

            pivot = subset.pivot(index='model', columns='train_size', values=metric)

            fig, ax = plt.subplots(figsize=(12, 8))

            if metric == 'direction_accuracy':
                pivot = pivot * 100

            if metric == 'direction_accuracy':
                cmap = 'RdYlGn'  # Green - heigher acc
                fmt = '.1f'
            else:
                cmap = 'RdYlGn_r'  # Green 0 lower err
                fmt = '.4f'

            sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap,
                       cbar_kws={'label': metric_labels[metric]},
                       linewidths=0.5, linecolor='gray', ax=ax)

            ax.set_title(f'{metric_labels[metric]} - {test_size}-Day Ahead Forecast\n' +
                        '(Averaged across all rolling windows)',
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Training Window Size (days)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Model', fontsize=12, fontweight='bold')

            plt.tight_layout()
            filename = f'heatmap_{metric}_test{test_size}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Saved heatmap: {filename}")


def create_summary_plots(df_results, output_dir):
    metrics = ['rmse', 'mae', 'direction_accuracy', 'mse_vol', 'qlike']
    models = sorted(df_results['model'].unique())

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Performance Comparison Across Forecast Horizons\n(Averaged across all training window sizes)',
                 fontsize=16, fontweight='bold')

    colors = {'GARCH-t': '#2ecc71',
              'GJR-GARCH-t': '#3498db',
              'GJR-GARCH-Skewed-t': '#e74c3c'}

    for idx, metric in enumerate(metrics):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]

        for model in models:
            subset = df_results[df_results['model'] == model]
            avg_by_test = subset.groupby('test_size')[metric].mean()

            if metric == 'direction_accuracy':
                avg_by_test = avg_by_test * 100

            ax.plot(avg_by_test.index, avg_by_test.values,
                   marker='o', linewidth=2.5, markersize=8,
                   label=model, color=colors.get(model, 'gray'))

        ax.set_xlabel('Forecast Horizon (days)', fontsize=11, fontweight='bold')

        if metric == 'direction_accuracy':
            ax.set_ylabel('Direction Accuracy (%)', fontsize=11, fontweight='bold')
        elif 'vol' in metric:
            ax.set_ylabel(metric.upper(), fontsize=11, fontweight='bold')
        else:
            ax.set_ylabel(metric.upper(), fontsize=11, fontweight='bold')

        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_title(f'{metric.upper()}', fontsize=12, fontweight='bold')

    fig.delaxes(axes[1, 2])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved summary comparison plot")


def main():
    setup_logging(log_dir="logs", log_level=logging.INFO)
    config = load_config()
    seed_everything(config.get("seed", 42))

    logger.info("=" * 80)
    logger.info("FTSE 100 Rolling Window Backtesting\n")
    logger.info("=" * 80)

    output_dir = "results/backtesting"
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(config)

    train_sizes = [50, 75, 100, 125, 150, 200]
    test_sizes = [1, 5, 10, 20]

    logger.info(f"\nTrain sizes: {train_sizes}")
    logger.info(f"Test sizes: {test_sizes}")
    logger.info(f"Total configurations: {len(train_sizes) * len(test_sizes)}")

    model_specs = [
        ('GJR-GARCH-Skewed-t', 'GJR-GARCH', 'skewt'),
        ('GJR-GARCH-t', 'GJR-GARCH', 't'),
        ('GARCH-t', 'GARCH', 't')
    ]

    logger.info(f"Models to test: {[spec[0] for spec in model_specs]}")

    df_results = run_backtesting_grid(df, train_sizes, test_sizes, model_specs)

    results_path = os.path.join(output_dir, 'backtesting_results.csv')
    df_results.to_csv(results_path, index=False)
    logger.info(f"\nSaved results to: {results_path}")

    logger.info("\nCreating heatmaps...")
    create_heatmaps(df_results, output_dir)

    logger.info("\nCreating summary plots...")
    create_summary_plots(df_results, output_dir)

    logger.info("\n" + "=" * 80)
    logger.info("BACKTESTING COMPLETED")
    logger.info("=" * 80)

    ranking_path = os.path.join(output_dir, "model_ranking.txt")
    with open(ranking_path, "w") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("OVERALL MODEL RANKING\n")
        f.write("=" * 80 + "\n")

        for metric in ['rmse', 'mae', 'direction_accuracy', 'mse_vol', 'qlike']:
            f.write(f"\n{metric.upper()}:\n")
            ranking = df_results.groupby('model')[metric].mean().sort_values(
                ascending=(metric != 'direction_accuracy')  # Higher is better for accuracy
            )
            for rank, (model, score) in enumerate(ranking.items(), 1):
                marker = ""
                if metric == 'direction_accuracy':
                    f.write(f"  {rank}. {marker} {model:25s} {score*100:6.2f}%\n")
                else:
                    f.write(f"  {rank}. {marker} {model:25s} {score:.14f}\n")

        f.write("\n" + "=" * 80 + "\n")
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
