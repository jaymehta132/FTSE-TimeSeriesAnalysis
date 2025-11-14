import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

results_path = "results/backtesting/backtesting_results.csv"
df = pd.read_csv(results_path)

print("Loaded backtesting results")
print(f"Models in data: {df['model'].unique()}")

df_filtered = df.copy()
print(f"\nAfter filtering: {df_filtered['model'].unique()}\nm")

output_dir = Path("results/backtesting/filtered_heatmaps")
output_dir.mkdir(exist_ok=True)

metrics = {
    'rmse': 'RMSE (Return Forecast Error)',
    'mae': 'MAE (Mean Absolute Error)',
    'direction_accuracy': 'Direction Accuracy (%)',
    'mse_vol': 'MSE (Volatility Forecast Error) *1e4',
    'qlike': 'QLIKE (Volatility Loss)'
}

test_sizes = [1, 5, 10, 20]
train_sizes = [50, 75, 100, 125, 150, 200]
models = ['GJR-GARCH-Skewed-t','GJR-GARCH-t', 'GARCH-t']

print("\nCreating filtered heatmaps ...")
print("=" * 80)

for metric_col, metric_name in metrics.items():
    for test_size in test_sizes:
        subset = df_filtered[df_filtered['test_size'] == test_size].copy()

        pivot = subset.pivot_table(
            values=metric_col,
            index='model',
            columns='train_size',
            aggfunc='mean'
        )
        if metric_col == 'mae' or metric_col == 'mse_vol' or metric_col == 'rmse':
            pivot.loc['GJR-GARCH-Skewed-t', 50] = np.nan
            print(pivot)


        pivot = pivot.reindex(index=models, columns=train_sizes)

        fig, ax = plt.subplots(figsize=(12, 4))

        if metric_col == 'direction_accuracy':
            pivot_display = pivot * 100
            fmt = '.2f'
            cmap = 'RdYlGn'  
        elif metric_col in ['rmse', 'mae', 'qlike']:
            pivot_display = pivot
            fmt = '.6f' if metric_col in ['rmse', 'mae', 'mse_vol'] else '.4f'
            cmap = 'RdYlGn_r'  
        elif metric_col == 'mse_vol':
            pivot_display = pivot*10000  
            fmt = '.6f'
            cmap = 'RdYlGn_r'  
        else:
            pivot_display = pivot
            fmt = '.6f'
            cmap = 'RdYlGn_r'

        sns.heatmap(
            pivot_display,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            cbar_kws={'label': metric_name},
            ax=ax,
            linewidths=1,
            linecolor='white'
        )

        ax.set_title(
            f'{metric_name} - {test_size}-Day Ahead Forecast\n(GJR-GARCH-t vs GARCH-t)',
            fontsize=14,
            fontweight='bold'
        )
        ax.set_xlabel('Training Window Size (days)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Model', fontsize=12, fontweight='bold')

        plt.tight_layout()

        filename = f"heatmap_{metric_col}_test{test_size}_filtered.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {filename}")

print("=" * 80)
print(f"\nAll filtered heatmaps saved to: {output_dir}")
