"""
Replot heatmaps excluding GJR-GARCH-Skewed-t to see differences between other models
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Load results
results_path = "results/backtesting/backtesting_results.csv"
df = pd.read_csv(results_path)

print("Loaded backtesting results")
print(f"Models in data: {df['model'].unique()}")

# Filter out GJR-GARCH-Skewed-t
df_filtered = df[df['model'] != 'GJR-GARCH-Skewed-t'].copy()
print(f"\nAfter filtering: {df_filtered['model'].unique()}")

# Create output directory
output_dir = Path("results/backtesting/filtered_heatmaps")
output_dir.mkdir(exist_ok=True)

# Metrics to plot
metrics = {
    'rmse': 'RMSE (Return Forecast Error)',
    'mae': 'MAE (Mean Absolute Error)',
    'direction_accuracy': 'Direction Accuracy (%)',
    'mse_vol': 'MSE (Volatility Forecast Error)',
    'qlike': 'QLIKE (Volatility Loss)'
}

test_sizes = [1, 5, 10, 20]
train_sizes = [50, 75, 100, 125, 150, 200]
models = ['GJR-GARCH-t', 'GARCH-t']

print("\nCreating filtered heatmaps (excluding GJR-GARCH-Skewed-t)...")
print("=" * 80)

for metric_col, metric_name in metrics.items():
    for test_size in test_sizes:
        # Filter data for this test size
        subset = df_filtered[df_filtered['test_size'] == test_size].copy()

        # Create pivot table for heatmap
        pivot = subset.pivot_table(
            values=metric_col,
            index='model',
            columns='train_size',
            aggfunc='mean'
        )

        # Ensure consistent ordering
        pivot = pivot.reindex(index=models, columns=train_sizes)

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 4))

        # Special formatting for direction accuracy (convert to percentage)
        if metric_col == 'direction_accuracy':
            pivot_display = pivot * 100
            fmt = '.2f'
            cmap = 'RdYlGn'  # Red-Yellow-Green (higher is better)
        elif metric_col in ['rmse', 'mae', 'mse_vol', 'qlike']:
            pivot_display = pivot
            fmt = '.6f' if metric_col in ['rmse', 'mae', 'mse_vol'] else '.4f'
            cmap = 'RdYlGn_r'  # Reverse (lower is better)
        else:
            pivot_display = pivot
            fmt = '.6f'
            cmap = 'RdYlGn_r'

        # Create heatmap
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

        # Save figure
        filename = f"heatmap_{metric_col}_test{test_size}_filtered.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved: {filename}")

print("=" * 80)
print(f"\nAll filtered heatmaps saved to: {output_dir}")
print("\nNow you can clearly see the differences between GJR-GARCH-t and GARCH-t!")
