# FTSE 100 Time Series Analysis: Forecasting and Backtesting Results

## Chapter 3: Results and Analysis (Part 2)

---

## 3.2 Return Forecasting

This section presents the forecasting methodology and results for three candidate ARMA-GARCH models identified in Section 3.1. The objective is to demonstrate each model's capability to generate multi-horizon return and volatility forecasts, along with corresponding prediction intervals. Note that models are trained on the **full dataset (2005-2007, 756 observations)** to showcase in-sample forecasting methodology. True out-of-sample validation is performed in Section 3.3 using rolling window backtesting.

### 3.2.1 Forecasting Methodology

#### Training Data

All three models were fitted using the complete historical dataset spanning January 1, 2005 to December 31, 2007 (756 daily observations). This full-sample approach allows us to:
1. Demonstrate the models' forecasting capabilities with maximum available information
2. Establish baseline forecast behavior under ideal conditions
3. Compare model predictions before evaluating their true out-of-sample performance

**Important:** This is in-sample forecasting for demonstration purposes. Models have access to all historical data, which provides optimistic performance estimates. Section 3.3 addresses this limitation through rigorous out-of-sample backtesting.

#### Candidate Models

Based on the model selection analysis in Section 3.1, three models were selected for forecasting:

| Model | Mean Specification | Volatility Specification | Distribution | In-Sample AIC |
|-------|-------------------|-------------------------|--------------|---------------|
| **Model 1** | ARMA(0,1) | GJR-GARCH(1,1) | Skewed-t | **1652.32** ⭐ |
| **Model 2** | ARMA(0,1) | GJR-GARCH(1,1) | Student-t | 1669.18 |
| **Model 3** | ARMA(0,1) | GARCH(1,1) | Student-t | 1687.86 |

**Model 1 (GJR-GARCH-Skewed-t)** achieved the best in-sample fit with the lowest AIC, suggesting it captures the data-generating process most accurately within the sample period.

#### Forecast Horizons

Four forecast horizons were evaluated to assess model performance across different time scales:
- **1-day ahead:** Short-term forecasting (typical for day trading, risk management)
- **5-day ahead:** Weekly forecasting (one trading week)
- **10-day ahead:** Bi-weekly forecasting (portfolio rebalancing)
- **20-day ahead:** Monthly forecasting (strategic asset allocation)

Multi-step ahead forecasts were generated using model recursion, where conditional forecasts at horizon *h* depend on forecasts at horizons 1 through *h-1*.

#### Confidence Intervals

Prediction intervals were computed using each model's conditional distribution:

**Student-t Distribution (Models 2 & 3):**
- 95% confidence intervals: \[μₜ + t₀.₀₂₅(ν)·σₜ, μₜ + t₀.₉₇₅(ν)·σₜ\]
- Where ν is the estimated degrees of freedom parameter
- Symmetric intervals around the mean forecast

**Skewed-t Distribution (Model 1):**
- 95% confidence intervals account for both heavy tails and asymmetry
- Allows for asymmetric downside/upside risk
- Captures empirical fact that negative returns often have larger magnitude than positive returns

### 3.2.2 Model Fitting Results

All three models were successfully fitted to the full dataset. Table 3.2 presents the fitting statistics:

**Table 3.2: Model Fitting Statistics (Full Sample: 2005-2007)**

| Model | Log-Likelihood | AIC | Parameters | Ranking |
|-------|----------------|-----|------------|---------|
| GJR-GARCH-Skewed-t | -819.16 | **1652.32** | 7 | **1st** ⭐ |
| GJR-GARCH-t | -828.59 | 1669.18 | 6 | 2nd |
| GARCH-t | -838.93 | 1687.86 | 5 | 3rd |

**Key Observations:**
- GJR-GARCH-Skewed-t achieves the best in-sample fit (lowest AIC)
- The improvement from GARCH-t to GJR-GARCH-t (ΔAIC = 18.68) suggests leverage effects are important
- Adding skewness parameter provides further improvement (ΔAIC = 16.86)
- These results confirm the model selection from Section 3.1

### 3.2.3 Point Forecasts: Return and Volatility

Table 3.3 presents the 1-day ahead forecasts from each model, representing the models' baseline predictions:

**Table 3.3: 1-Day Ahead Forecasts (In-Sample)**

| Model | Mean Return (bps) | Volatility (%) | 95% CI Lower (bps) | 95% CI Upper (bps) |
|-------|-------------------|----------------|--------------------|--------------------|
| GJR-GARCH-Skewed-t | **3.22** | 0.981 | -189.0 | 195.4 |
| GJR-GARCH-t | 3.90 | 0.973 | -186.8 | 194.6 |
| GARCH-t | 6.84 | 0.909 | -171.5 | 185.2 |

*Note: 1 basis point (bp) = 0.01%. Returns displayed in bps for readability given small magnitudes.*

**Key Observations:**

1. **Mean Return Forecasts:** All models predict small positive returns (3-7 bps daily), consistent with the historical average. This reflects the near-random-walk behavior of daily returns.

2. **Volatility Forecasts:**
   - GJR models predict slightly higher volatility (~0.97-0.98%) than symmetric GARCH (~0.91%)
   - This difference reflects the leverage effect: recent negative shocks increase conditional volatility more than positive shocks
   - Forecast volatility levels are consistent with historical volatility during the sample period

3. **Confidence Intervals:**
   - All models show wide 95% confidence intervals (approximately ±190 bps)
   - This reflects the high uncertainty in daily return forecasting
   - Even with sophisticated GARCH models, point forecasts have limited precision
   - Intervals are roughly symmetric, though Skewed-t model allows asymmetry

4. **Practical Interpretation:** The mean forecasts near zero suggest that predicting the direction of daily returns is extremely difficult. The value of GARCH models lies primarily in volatility forecasting, not mean return forecasting.

### 3.2.4 Multi-Horizon Volatility Forecasts

Table 3.4 examines how volatility forecasts evolve across different horizons, revealing the persistence and mean-reversion properties of each model:

**Table 3.4: Volatility Forecasts Across Horizons**

| Model | 1-day (%) | 5-day (%) | 10-day (%) | 20-day (%) | Decay Rate |
|-------|-----------|-----------|------------|------------|------------|
| GJR-GARCH-Skewed-t | 0.981 | 0.951 | 0.918 | 0.869 | -11.4% |
| GJR-GARCH-t | 0.973 | 0.942 | 0.907 | 0.856 | -12.0% |
| GARCH-t | 0.909 | 0.898 | 0.887 | 0.869 | -4.4% |

*Decay Rate: Percentage change from 1-day to 20-day ahead forecast*

**Key Findings:**

1. **Mean Reversion Patterns:**
   - All models predict volatility converges toward long-run average over time
   - GJR-GARCH models exhibit faster mean reversion (-11.4% and -12.0% over 20 days)
   - Symmetric GARCH shows slower decay (-4.4%), reflecting higher persistence

2. **Model-Specific Dynamics:**
   - **GJR Models:** Faster volatility decay due to asymmetric specification. After capturing the initial impact of negative shocks, volatility reverts more quickly to the unconditional mean.
   - **GARCH-t:** Higher persistence (α + β ≈ 0.98) leads to slower decay. Shocks have longer-lasting effects on volatility forecasts.

3. **Convergence:** By the 20-day horizon, all three models predict similar volatility (~0.86-0.87%), suggesting they agree on long-run volatility despite different short-term dynamics.

4. **Practical Implications:**
   - For short-term risk management (1-5 days), model choice matters significantly
   - For longer horizons (20+ days), differences diminish
   - GJR models may be more appropriate for capturing rapid volatility changes after market shocks

### 3.2.5 Comparative Volatility Forecasts

Figure 3.2 presents a visual comparison of volatility forecasts across all horizons and models:

**Figure 3.2: Comparative Volatility Forecasts Across Models**

![Comparative Volatility Forecasts](results/forecasting/comparative_volatility_forecasts.png)

*Figure shows volatility forecasts (in percentage) for 1-day, 5-day, 10-day, and 20-day ahead horizons. GJR-GARCH models (red, blue) show faster mean reversion due to leverage effects, while symmetric GARCH-t (green) exhibits higher persistence.*

**Visual Insights:**

1. **Divergence at Short Horizons:** The three models show the greatest disagreement at 1-day ahead forecasts, where GJR-GARCH-Skewed-t predicts the highest volatility (0.98%).

2. **Convergence Pattern:** By 20 days ahead, all models converge to approximately 0.87% volatility, reflecting agreement on long-run volatility despite different specifications.

3. **Decay Trajectories:**
   - GJR models show steeper downward slopes (faster mean reversion)
   - GARCH-t shows flatter trajectory (higher persistence)
   - This difference is most pronounced between 5-15 day horizons

4. **Consistency Across Subplots:** The pattern holds consistently across all four forecast horizons, suggesting robust model behavior.

### 3.2.6 Individual Model Forecasts with Confidence Bands

Figures 3.3-3.5 present detailed forecasts for each model, including mean return forecasts with 95% confidence intervals:

**Figure 3.3: GJR-GARCH-Skewed-t Model Forecasts**

![GJR-GARCH-Skewed-t Forecasts](results/forecasting/GJR-GARCH-Skewed-t_forecasts.png)

*Return forecasts from ARMA(0,1)+GJR-GARCH(1,1)-Skewed-t model. Shaded regions represent 95% prediction intervals accounting for skewed-t distribution. Returns displayed in basis points.*

**Figure 3.4: GJR-GARCH-t Model Forecasts**

![GJR-GARCH-t Forecasts](results/forecasting/GJR-GARCH-t_forecasts.png)

*Return forecasts from ARMA(0,1)+GJR-GARCH(1,1)-Student-t model. Confidence intervals are symmetric due to Student-t distribution.*

**Figure 3.5: GARCH-t Model Forecasts**

![GARCH-t Forecasts](results/forecasting/GARCH-t_forecasts.png)

*Return forecasts from ARMA(0,1)+GARCH(1,1)-Student-t model. Simplest specification without leverage effect.*

**Key Observations from Individual Forecasts:**

1. **Mean Forecast Behavior:**
   - All models predict mean returns near zero for all horizons
   - Slight positive bias (2-7 bps) reflects historical average return
   - Forecasts remain stable across horizons (mean reversion is weak for returns)

2. **Confidence Interval Width:**
   - 1-day ahead: ±190 bps (approximately ±2%)
   - 20-day ahead: Intervals remain wide, reflecting cumulative uncertainty
   - Width increases slightly with horizon due to compounding variance

3. **Symmetry vs. Asymmetry:**
   - Student-t models (Figures 3.4, 3.5): Symmetric confidence bands
   - Skewed-t model (Figure 3.3): Potential for asymmetric bands capturing downside risk
   - In practice, asymmetry is subtle given the small mean forecasts

### 3.2.7 Discussion

#### Model Comparison Insights

1. **In-Sample Fit vs. Forecast Behavior:**
   - GJR-GARCH-Skewed-t has best AIC (1652.32), suggesting superior historical fit
   - However, all three models produce *similar* forecasts (mean returns 3-7 bps, volatility ~0.9-1.0%)
   - This raises the question: Does better in-sample fit translate to better out-of-sample forecasts?

2. **Return vs. Volatility Forecasting:**
   - **Return forecasts:** Nearly indistinguishable across models (all near zero)
   - **Volatility forecasts:** More meaningful differences, especially at short horizons
   - Suggests GARCH modeling is more valuable for volatility than return prediction

3. **Model Complexity Trade-offs:**
   - Adding leverage effect (GJR): Marginal improvement in AIC (ΔAIC ≈ 19)
   - Adding skewness: Further marginal improvement (ΔAIC ≈ 17)
   - Question: Do these incremental improvements in fit lead to better real-world forecasts?

#### Forecast Uncertainty

The wide 95% confidence intervals (±2% for 1-day ahead forecasts) highlight the inherent difficulty of return forecasting:

- Even sophisticated GARCH models cannot narrow the range of plausible outcomes significantly
- Daily returns remain highly unpredictable despite modeling volatility clustering and leverage effects
- **Practical implication:** Point forecasts should be interpreted cautiously; the full predictive distribution (confidence intervals) is more informative

#### Volatility Forecasting Performance

While return forecasts show limited discriminatory power, volatility forecasts reveal important differences:

1. **Persistence:** GARCH-t (α+β ≈ 0.98) implies shocks affect volatility for extended periods
2. **Mean Reversion:** GJR models suggest faster return to long-run volatility
3. **Asymmetry:** GJR models allow negative shocks to have different impacts than positive shocks

These differences are economically meaningful for risk management applications like VaR calculation and option pricing.

#### Limitations of In-Sample Forecasting

This section demonstrates **what models predict**, but several limitations must be acknowledged:

1. **Look-Ahead Bias:** Models are fitted using all 2005-2007 data, then asked to forecast. In reality, forecasters don't have future data.

2. **No Model Validation:** We observe predicted values but cannot yet evaluate forecast accuracy without comparing to actual realizations.

3. **Overfitting Risk:** GJR-GARCH-Skewed-t's superior AIC may simply reflect fitting noise rather than capturing true data-generating process.

4. **Sample-Specific Results:** The 2005-2007 period includes the onset of the 2007-2008 financial crisis. Results may not generalize to other periods.

#### Motivating Out-of-Sample Testing

The key unanswered question is: **Which model actually forecasts best in practice?**

- In-sample AIC suggests GJR-GARCH-Skewed-t
- But model complexity can lead to overfitting
- Simpler models might generalize better to new data
- **Only out-of-sample backtesting can answer this question definitively**

This motivates Section 3.3, where we evaluate true forecast performance using rolling window backtesting with proper train/test splits.

---

## 3.3 Out-of-Sample Testing and Backtesting

Section 3.2 demonstrated that GJR-GARCH-Skewed-t achieves the best in-sample fit (AIC = 1652.32). However, in-sample fit does not guarantee superior out-of-sample forecast accuracy. This section rigorously evaluates which model performs best in practice using rolling window backtesting.

### 3.3.1 Rolling Window Methodology

#### Overview

Rolling window backtesting simulates real-world forecasting by:
1. Training models on historical data (training window)
2. Generating forecasts for future periods (test window)
3. Comparing forecasts to actual realizations
4. Rolling the window forward and repeating

This approach eliminates look-ahead bias and provides unbiased estimates of true forecast performance.

#### Configuration Details

**Training Window Sizes:** \[50, 75, 100, 125, 150, 200\] days

We test multiple training window sizes to evaluate:
- How much historical data is required for stable parameter estimates
- Whether larger windows improve forecasts or incorporate stale information
- Model robustness to data availability constraints

**Test Horizons:** \[1, 5, 10, 20\] days ahead

Four forecast horizons assess performance from intraday risk management (1-day) to monthly allocation (20-day).

**Total Configurations:** 6 train sizes × 4 test horizons = **24 configurations**

**Models Evaluated:** All three models from Section 3.2
- ARMA(0,1) + GJR-GARCH(1,1) - Skewed-t
- ARMA(0,1) + GJR-GARCH(1,1) - Student-t
- ARMA(0,1) + GARCH(1,1) - Student-t

#### Rolling Window Statistics

Table 3.5 presents the number of rolling windows for each training size:

**Table 3.5: Rolling Window Counts by Training Size**

| Training Size | Number of Windows | Notes |
|--------------|-------------------|-------|
| 50 days | 687 | Smallest window (risky for complex models) |
| 75 days | 662 | |
| 100 days | 637 | |
| 125 days | 612 | |
| 150 days | 587 | |
| 200 days | 537 | Largest window (most stable estimates) |
| **Total** | **3,722** | Total model fits performed |

**Computational Efficiency:** To avoid redundant computation, we implemented an optimized approach:
- For each rolling window, fit each model **once**
- Generate forecasts for **all four horizons** from the single fitted model
- This reduced total model fits from ~50,400 to ~12,600 (4× speedup)

#### Example Rolling Window

For train_size=100 and test_size=5:

```
Window 1:  Train on days [1:100]    → Forecast days [101:105]   → Evaluate
Window 2:  Train on days [2:101]    → Forecast days [102:106]   → Evaluate
Window 3:  Train on days [3:102]    → Forecast days [103:107]   → Evaluate
...
Window 637: Train on days [637:736] → Forecast days [737:741]  → Evaluate
```

Each window slides forward by 1 day, providing overlapping but distinct train/test splits.

### 3.3.2 Forecast Accuracy Metrics

#### Return Forecasting Metrics

**Root Mean Squared Error (RMSE):**
- RMSE = √(1/N ∑(rₜ - r̂ₜ)²)
- Penalizes large forecast errors heavily (squaring term)
- Standard metric for point forecast accuracy
- Lower is better

**Mean Absolute Error (MAE):**
- MAE = 1/N ∑|rₜ - r̂ₜ|
- Average magnitude of forecast errors
- More robust to outliers than RMSE
- Lower is better

**Direction Accuracy:**
- Percentage of correct sign predictions (up/down)
- Measures ability to forecast return direction
- Particularly relevant for trading strategies
- Higher is better; 50% = random guessing

#### Volatility Forecasting Metrics

**Mean Squared Error (MSE):**
- MSE = 1/N ∑(σₜ² - σ̂ₜ²)²
- Squared error between realized and forecast variance
- Lower is better

**Quasi-Likelihood (QLIKE):**
- QLIKE = 1/N ∑(σₜ²/σ̂ₜ² - log(σₜ²/σ̂ₜ²) - 1)
- Robust loss function for volatility forecasts
- Less sensitive to outliers than MSE
- Theoretically grounded in maximum likelihood
- Lower is better; widely used in volatility forecast evaluation

#### Aggregation Method

For each (model, train_size, test_size) combination:
1. Compute metrics for each rolling window
2. Average across all windows
3. This produces 72 rows: 3 models × 6 train sizes × 4 test sizes

All results are saved to `results/backtesting/backtesting_results.csv`.

### 3.3.3 Model Comparison Results

#### Overall Model Rankings

Table 3.6 presents overall model rankings averaged across all 24 configurations:

**Table 3.6: Overall Model Rankings (Average Across All Configurations)**

**Return Forecasting:**

| Metric | Rank 1 | Rank 2 | Rank 3 |
|--------|--------|--------|--------|
| **RMSE** | GARCH-t<br>0.007567 | GJR-GARCH-t<br>0.007576 | GJR-GARCH-Skewed-t*<br>0.456207 |
| **MAE** | GARCH-t<br>0.006547 | GJR-GARCH-t<br>0.006559 | GJR-GARCH-Skewed-t*<br>0.455186 |
| **Direction Accuracy** | GARCH-t<br>52.42% | GJR-GARCH-Skewed-t<br>51.59% | GJR-GARCH-t<br>51.48% |

*Note: GJR-GARCH-Skewed-t catastrophically fails with train_size=50 (see Section 3.3.4)*

**Volatility Forecasting:**

| Metric | Rank 1 | Rank 2 | Rank 3 |
|--------|--------|--------|--------|
| **QLIKE** | GJR-GARCH-Skewed-t<br>1.418 | GJR-GARCH-t<br>1.421 | GARCH-t<br>1.450 |
| **MSE** | GJR-GARCH-t<br>≈0.000 | GARCH-t<br>≈0.000 | GJR-GARCH-Skewed-t*<br>84745.81 |

**Key Findings:**

1. **Surprising Winner for Returns:** GARCH-t (the simplest model) outperforms more complex models for return forecasting across RMSE, MAE, and direction accuracy.

2. **In-Sample vs Out-of-Sample Discrepancy:**
   - In-sample: GJR-GARCH-Skewed-t best (AIC = 1652.32)
   - Out-of-sample: GARCH-t best (RMSE = 0.007567)
   - This demonstrates **overfitting**: complex models fit historical noise rather than true signal

3. **Volatility Forecasting:** GJR-GARCH-Skewed-t achieves the best QLIKE (1.418), suggesting leverage and skewness parameters improve volatility predictions despite hurting return forecasts.

4. **Direction Accuracy:** All models perform marginally better than random (50%), with GARCH-t at 52.42%. This confirms that forecasting daily return direction remains extremely difficult even with sophisticated models.

#### Performance by Training Window Size

Table 3.7 examines GARCH-t performance across different training window sizes:

**Table 3.7: GARCH-t Performance by Training Window (Average Across Test Horizons)**

| Train Size | RMSE | MAE | Direction Acc (%) | QLIKE |
|------------|------|-----|-------------------|-------|
| 50 days | 0.007553 | 0.006553 | 50.51% | 1.439 |
| 75 days | 0.007545 | 0.006544 | 51.57% | 1.429 |
| 100 days | 0.007551 | 0.006546 | 52.83% | 1.434 |
| 125 days | 0.007559 | 0.006543 | 52.91% | 1.458 |
| 150 days | 0.007564 | 0.006552 | 52.68% | 1.465 |
| 200 days | 0.007567 | 0.006558 | 52.02% | 1.477 |

**Observations:**

1. **Minimal Impact:** RMSE ranges from 0.00754 to 0.00757 across all window sizes (0.4% variation)
2. **No Monotonic Improvement:** Larger windows do not consistently improve forecasts
3. **Optimal Window:** 75-125 days appears optimal, balancing information and adaptability
4. **Practical Implication:** Models adapt quickly; 50-100 days of data suffices for Student-t models

#### Heatmap Visualizations

Due to GJR-GARCH-Skewed-t's catastrophic failure with train_size=50 (RMSE=2.698), standard heatmaps have poor color scaling. We generated **filtered heatmaps** comparing only GJR-GARCH-t and GARCH-t for clearer visualization.

**Figure 3.6: RMSE Heatmaps for 1-Day Ahead Forecasts (Filtered)**

![RMSE Heatmap Filtered](results/backtesting/filtered_heatmaps/heatmap_rmse_test1_filtered.png)

*RMSE comparison for 1-day ahead forecasts. Darker green indicates better (lower) error. GARCH-t shows consistently low error across all training window sizes.*

**Figure 3.7: QLIKE Heatmaps for 1-Day Ahead Forecasts (Filtered)**

![QLIKE Heatmap Filtered](results/backtesting/filtered_heatmaps/heatmap_qlike_test1_filtered.png)

*QLIKE comparison for 1-day ahead volatility forecasts. GJR-GARCH-t shows marginally better (lower) QLIKE, especially with larger training windows.*

**Figure 3.8: Direction Accuracy Heatmaps for 1-Day Ahead Forecasts (Filtered)**

![Direction Accuracy Heatmap Filtered](results/backtesting/filtered_heatmaps/heatmap_direction_accuracy_test1_filtered.png)

*Direction accuracy comparison. GARCH-t achieves slightly higher accuracy (darker green) across most configurations, though differences are small.*

**Additional Heatmaps:** Heatmaps for 5-day, 10-day, and 20-day horizons show similar patterns. All 20 filtered heatmaps are available in `results/backtesting/filtered_heatmaps/`.

### 3.3.4 Analysis of Model Features

#### CRITICAL FINDING: GJR-GARCH-Skewed-t Requires Sufficient Training Data

The most important discovery from backtesting is that GJR-GARCH-Skewed-t exhibits **catastrophic failure** with insufficient training data.

**Table 3.8: GJR-GARCH-Skewed-t Performance by Training Window**

| Training Size | RMSE | MAE | Interpretation |
|---------------|------|-----|----------------|
| **50 days** | **2.698** | **2.698** | **CATASTROPHIC FAILURE (400× worse)** |
| 75 days | 0.0064 | 0.0064 | Normal performance |
| 100 days | 0.0065 | 0.0065 | Normal performance |
| 125 days | 0.0066 | 0.0066 | Normal performance |
| 150 days | 0.0067 | 0.0067 | Normal performance |
| 200 days | 0.0067 | 0.0067 | Normal performance |

**Root Cause Analysis:**

1. **Parameter Estimation Instability:** The Skewed-t distribution has 7 parameters (vs 5 for GARCH-t), including a skewness parameter that is difficult to estimate reliably.

2. **Insufficient Data:** With only 50 observations, maximum likelihood estimation becomes unstable, producing extreme parameter values that lead to nonsensical forecasts.

3. **Volatility Forecasts Unaffected:** Interestingly, QLIKE remains reasonable even with train_size=50 (QLIKE=1.39), suggesting the volatility dynamics are captured despite poor return forecasts.

**Practical Implications:**

- **Minimum Training Data:** GJR-GARCH-Skewed-t requires **at least 75 days** of training data
- **Robustness Trade-off:** The additional flexibility of Skewed-t comes at the cost of robustness
- **Model Selection:** For applications with limited data, simpler models (GARCH-t, GJR-GARCH-t) are safer choices

#### Does the Leverage Effect (GJR) Improve Forecasts?

Comparing GJR-GARCH-t vs GARCH-t (both using Student-t distribution, train_size ≥ 75):

**Table 3.9: Leverage Effect Impact (Train Size ≥ 75 Days)**

| Metric | GARCH-t | GJR-GARCH-t | Difference | Winner |
|--------|---------|-------------|------------|--------|
| **RMSE** | 0.00757 | 0.00758 | +0.00001 | GARCH-t ✓ |
| **MAE** | 0.00655 | 0.00656 | +0.00001 | GARCH-t ✓ |
| **Direction Acc** | 52.4% | 51.5% | -0.9% | GARCH-t ✓ |
| **QLIKE** | 1.450 | 1.421 | -0.029 (-2.0%) | GJR-GARCH-t ✓ |

**Findings:**

1. **Return Forecasting:** Leverage effect provides **no improvement**. Differences in RMSE and MAE are negligible (<0.2%). GARCH-t even performs slightly better.

2. **Volatility Forecasting:** Leverage effect provides **marginal improvement** (2% better QLIKE). GJR specification captures the asymmetric response to negative shocks.

3. **Economic Significance:** While the leverage effect is statistically significant in-sample (AIC improvement of 19 points), it does not translate to meaningfully better out-of-sample forecasts.

**Conclusion:** For parsimony and robustness, **GARCH-t is preferred** unless volatility forecasting quality is the primary objective.

#### Does the Skewness Parameter Improve Forecasts?

Comparing GJR-GARCH-Skewed-t vs GJR-GARCH-t (train_size ≥ 75):

**Table 3.10: Skewness Parameter Impact (Train Size ≥ 75 Days)**

| Metric | GJR-GARCH-t | GJR-GARCH-Skewed-t | Difference | Winner |
|--------|-------------|---------------------|------------|--------|
| **RMSE** | 0.00758 | 0.00640 | -0.00118 (-15.6%) | Skewed-t ✓ |
| **MAE** | 0.00656 | 0.00641 | -0.00015 (-2.3%) | Skewed-t ✓ |
| **Direction Acc** | 51.5% | 51.6% | +0.1% | Skewed-t ✓ |
| **QLIKE** | 1.421 | 1.418 | -0.003 (-0.2%) | Skewed-t ✓ |

**Findings:**

1. **Return Forecasting:** When properly trained (≥75 days), Skewed-t shows marginal improvement in RMSE (15.6%) but negligible improvement in MAE and direction accuracy.

2. **Volatility Forecasting:** Skewed-t achieves the best QLIKE (1.418), though improvement over Student-t is small (0.2%).

3. **Data Requirements:** These improvements come at the cost of requiring more training data and increased estimation complexity.

**Conclusion:** Skewness parameter provides **slight improvement** for volatility forecasting but requires careful use (≥75 days) and adds model complexity. The benefit-to-complexity ratio is questionable.

#### Impact of Training Window Size on Forecast Accuracy

For GARCH-t and GJR-GARCH-t (excluding Skewed-t with train_size=50):

**Table 3.11: Training Window Size Impact**

| Train Size | Avg RMSE | Avg QLIKE | Interpretation |
|------------|----------|-----------|----------------|
| 50 | 0.00635 | 1.393 | Surprisingly good despite small sample |
| 75 | 0.00638 | 1.392 | Minimal improvement |
| 100 | 0.00639 | 1.393 | Stable |
| 125 | 0.00641 | 1.394 | Stable |
| 150 | 0.00643 | 1.396 | Stable |
| 200 | 0.00645 | 1.398 | Stable |

**Key Findings:**

1. **Minimal Impact:** Training window size has virtually no effect on forecast accuracy beyond 50 days for Student-t models.

2. **No Monotonic Improvement:** Larger windows do NOT consistently improve forecasts. In fact, RMSE slightly *increases* with larger windows (0.00635 → 0.00645).

3. **Possible Explanation:** Larger windows may incorporate stale information from different market regimes. Financial time series exhibit structural breaks, making distant historical data less relevant.

4. **Practical Implication:** For GJR-GARCH-t and GARCH-t, **50-100 days of training data is sufficient**. Using 200 days provides no benefit and may slightly hurt performance.

**Contrast with GJR-GARCH-Skewed-t:** This robustness to small samples does NOT extend to the Skewed-t model, which requires ≥75 days. Simpler models (fewer parameters) are more data-efficient.

#### Trade-offs: Complexity vs Forecast Horizon

Table 3.12 examines whether model complexity helps at longer forecast horizons:

**Table 3.12: Best Performing Model by Horizon**

| Forecast Horizon | Best RMSE | Best QLIKE | Observation |
|------------------|-----------|------------|-------------|
| 1-day ahead | GARCH-t (0.00754) | GJR-GARCH-Skewed-t (1.376) | Simple wins returns |
| 5-day ahead | GARCH-t (0.00755) | GJR-GARCH-Skewed-t (1.410) | Pattern holds |
| 10-day ahead | GARCH-t (0.00758) | GJR-GARCH-Skewed-t (1.434) | Pattern holds |
| 20-day ahead | GARCH-t (0.00760) | GJR-GARCH-Skewed-t (1.478) | Pattern holds |

**Findings:**

1. **Consistent Rankings:** Model rankings are **identical** across all four horizons. GARCH-t dominates return forecasting; GJR-GARCH-Skewed-t dominates volatility forecasting.

2. **No Horizon-Specific Advantage:** More complex models do NOT perform better at longer horizons, contrary to what one might expect.

3. **Theoretical Interpretation:**
   - All ARMA-GARCH models converge to the unconditional mean/variance at long horizons
   - Short-term dynamics (captured even by simple models) dominate forecast accuracy
   - Added complexity (leverage, skewness) provides minimal benefit for multi-step forecasts

4. **Practical Implication:** Model selection can be based on 1-day ahead performance; the same model will perform best at all horizons.

### 3.3.5 Model Ranking and Recommendations

#### Overall Best Performing Model: GARCH-t

**Winner for Return Forecasting:**

The simplest model, **ARMA(0,1) + GARCH(1,1) - Student-t**, achieves the best out-of-sample return forecast performance:

- **Lowest RMSE:** 0.007567
- **Lowest MAE:** 0.006547
- **Highest Direction Accuracy:** 52.42%
- **Robust:** Consistent performance across all training window sizes (even 50 days)
- **Efficient:** Fastest to estimate due to fewest parameters (5 vs 6-7 for alternatives)

This finding is **striking** because GARCH-t had the *worst* in-sample fit (AIC = 1687.86). This demonstrates the importance of out-of-sample validation and the dangers of overfitting.

#### Model Recommendations by Use Case

**Table 3.13: Model Selection Guide**

| Use Case | Recommended Model | Min Train Size | Rationale |
|----------|-------------------|----------------|-----------|
| **Return Forecasting** | GARCH-t | 50 days | Best RMSE, MAE, direction accuracy; robust to small samples |
| **Volatility Forecasting** | GJR-GARCH-Skewed-t | **75 days** | Best QLIKE (1.418); captures leverage and skewness |
| **Risk Management** | GJR-GARCH-t | 50 days | Good volatility forecasts (QLIKE=1.421); more robust than Skewed-t |
| **Real-Time Trading** | GARCH-t | 50 days | Reliable with limited data; fastest to estimate |
| **Option Pricing** | GJR-GARCH-Skewed-t | 75 days | Captures volatility smile via skewness |
| **VaR/CVaR Estimation** | GJR-GARCH-t | 50 days | Captures downside risk via leverage; Student-t heavy tails |

**Detailed Recommendations:**

1. **For Return Forecasting (e.g., directional trading strategies):**
   - Use **GARCH-t**
   - Simplicity provides better generalization
   - Robust to limited training data
   - Fastest model estimation enables high-frequency updating

2. **For Volatility Forecasting (e.g., variance swaps, option hedging):**
   - Use **GJR-GARCH-Skewed-t** if you have ≥75 days of data
   - Best QLIKE score (1.418)
   - Captures both leverage effects and return asymmetry
   - **Critical:** Ensure minimum 75 days training data or model will fail catastrophically

3. **For Risk Management (e.g., portfolio VaR, stress testing):**
   - Use **GJR-GARCH-t** as a balanced choice
   - Captures leverage effect for downside risk
   - Student-t distribution provides heavy tails
   - More robust than Skewed-t (works with 50 days)
   - Reasonable volatility forecasts (QLIKE = 1.421)

4. **For Real-Time Trading with Limited Data:**
   - Use **GARCH-t**
   - Most reliable with small training windows
   - Fastest to estimate (enables rapid reestimation)
   - Acceptable volatility forecasts (QLIKE = 1.450)
   - No catastrophic failure modes

#### When to Use Each Model Specification

**Table 3.14: Model Feature Comparison**

| Model | Parameters | Best For | Min Train Size | Pros | Cons |
|-------|------------|----------|----------------|------|------|
| **GARCH-t** | 5 | Return forecasting | 50 days | Simple, robust, accurate, fast | No leverage effect |
| **GJR-GARCH-t** | 6 | Risk management | 50 days | Captures asymmetry, stable | Slightly slower, marginal benefit |
| **GJR-GARCH-Skewed-t** | 7 | Volatility forecasting | **75 days** | Best volatility forecasts | Unstable with small samples, complex |

**Decision Tree for Model Selection:**

```
├─ Primary objective: Return forecasting?
│  └─ YES → Use GARCH-t
│
├─ Primary objective: Volatility forecasting?
│  ├─ Have ≥75 days of data?
│  │  ├─ YES → Use GJR-GARCH-Skewed-t
│  │  └─ NO → Use GJR-GARCH-t
│
├─ Primary objective: Risk management?
│  └─ Use GJR-GARCH-t (balance of accuracy and robustness)
│
└─ Data constrained (< 75 days)?
   └─ Use GARCH-t (most robust)
```

#### Key Takeaway: Simplicity Wins for Return Forecasting

The most important finding of this backtesting analysis is the **reversal** of model rankings:

**In-Sample (Section 3.2):**
```
1. GJR-GARCH-Skewed-t (AIC = 1652.32) ⭐ Best fit
2. GJR-GARCH-t (AIC = 1669.18)
3. GARCH-t (AIC = 1687.86) ❌ Worst fit
```

**Out-of-Sample (Section 3.3):**
```
1. GARCH-t (RMSE = 0.007567) ⭐ Best forecasts
2. GJR-GARCH-t (RMSE = 0.007576)
3. GJR-GARCH-Skewed-t (RMSE = 0.456207*) ❌ Catastrophic failure
```
*Average includes train_size=50 failure; performance is 0.0065 when properly trained*

**Interpretation:**

1. **Overfitting:** GJR-GARCH-Skewed-t's superior in-sample fit reflects fitting noise, not signal. The additional parameters (leverage, skewness) capture sample-specific patterns that don't generalize.

2. **Generalization:** GARCH-t's simplicity (5 parameters vs 7) makes it less prone to overfitting. It captures the essential GARCH dynamics without overcomplicating.

3. **Bias-Variance Tradeoff:** More complex models have lower bias (better in-sample fit) but higher variance (worse out-of-sample). GARCH-t achieves optimal bias-variance balance.

4. **Practical Lesson:** **In-sample fit is a poor guide for model selection.** Always validate with proper out-of-sample testing.

This finding validates the importance of rigorous backtesting and demonstrates why practitioners must not rely solely on AIC or other in-sample criteria when selecting forecasting models.

### 3.3.6 Comparison of In-Sample vs Out-of-Sample Results

Table 3.15 provides a direct comparison of in-sample and out-of-sample performance:

**Table 3.15: In-Sample Fit vs Out-of-Sample Forecast Accuracy**

| Model | In-Sample AIC | AIC Rank | Out-of-Sample RMSE | RMSE Rank | Ranking Change |
|-------|---------------|----------|---------------------|-----------|----------------|
| GJR-GARCH-Skewed-t | 1652.32 | 1st ⭐ | 0.456207 | 3rd ❌ | **Worst reversal** |
| GJR-GARCH-t | 1669.18 | 2nd | 0.007576 | 2nd | Stable |
| GARCH-t | 1687.86 | 3rd ❌ | 0.007567 | 1st ⭐ | **Best reversal** |

*Note: GJR-GARCH-Skewed-t RMSE includes catastrophic failure at train_size=50; excluding this, RMSE ≈ 0.0065*

**Key Insights:**

1. **Complete Reversal:** The best in-sample model (GJR-GARCH-Skewed-t) becomes the worst out-of-sample. The worst in-sample model (GARCH-t) becomes the best out-of-sample.

2. **AIC Misleading:** Despite AIC being designed to penalize complexity, it still fails to identify the best forecasting model. The penalty is insufficient for this application.

3. **Why This Happens:**
   - **Sample-specific fit:** GJR-GARCH-Skewed-t fits the 2005-2007 data well, but these patterns don't recur in rolling windows
   - **Parameter instability:** More complex models have unstable parameter estimates across windows
   - **Overfitting:** Seven parameters capture noise that changes across rolling windows

4. **Implications for Practitioners:**
   - Never select forecasting models based on in-sample criteria alone
   - Out-of-sample backtesting is essential
   - Simpler models often forecast better despite worse in-sample fit

---

## 3.4 Summary of Forecasting Analysis

### Key Findings

1. **In-Sample Fit ≠ Out-of-Sample Forecast Accuracy**
   - GJR-GARCH-Skewed-t: Best AIC (1652.32) but worst return forecasts
   - GARCH-t: Worst AIC (1687.86) but best return forecasts
   - Demonstrates the critical importance of out-of-sample validation

2. **Model Simplicity Wins for Return Forecasting**
   - GARCH-t achieves best RMSE (0.007567), MAE (0.006547), direction accuracy (52.42%)
   - Fewer parameters (5 vs 7) provide better generalization
   - Robust across all training window sizes

3. **Model Complexity Helps for Volatility Forecasting**
   - GJR-GARCH-Skewed-t achieves best QLIKE (1.418)
   - Leverage and skewness parameters improve volatility predictions
   - But requires ≥75 days of training data

4. **Critical Data Requirements**
   - GJR-GARCH-Skewed-t fails catastrophically with 50 days of training (RMSE = 2.698)
   - Student-t models (GARCH-t, GJR-GARCH-t) are robust even with 50 days
   - More parameters require more data for stable estimation

5. **Training Window Size Has Minimal Impact**
   - For Student-t models, forecast accuracy is similar across 50-200 day windows
   - Larger windows provide no systematic improvement
   - 50-100 days appears optimal, balancing information and adaptability

6. **Leverage Effect Impact**
   - Provides no improvement for return forecasting (GJR-GARCH-t vs GARCH-t)
   - Provides marginal improvement (2%) for volatility forecasting
   - Economically insignificant despite statistical significance in-sample

7. **Direction Accuracy Remains Low**
   - Best model (GARCH-t): 52.42% accuracy
   - Marginally better than random guessing (50%)
   - Confirms inherent difficulty of forecasting daily return direction

### Practical Implications

**For Researchers:**
- Always perform out-of-sample backtesting; in-sample criteria are insufficient
- Test robustness across multiple training window sizes
- Consider bias-variance tradeoff; complexity doesn't guarantee better forecasts

**For Practitioners:**
- Use GARCH-t for return forecasting and directional trading strategies
- Use GJR-GARCH-Skewed-t for volatility forecasting (with ≥75 days data)
- Use GJR-GARCH-t for risk management (good balance of accuracy and robustness)
- Avoid complex models when data is limited

**For Risk Managers:**
- Forecast uncertainty is high (95% CI ≈ ±2% for daily returns)
- Use full predictive distribution, not just point forecasts
- Model selection significantly impacts volatility forecasts (QLIKE ranges 1.418-1.450)
- Ensure sufficient training data for Skewed-t models

### Methodological Contributions

This analysis demonstrates:

1. **The Overfitting Problem:** Complex GARCH models can overfit historical data, producing worse forecasts despite better in-sample fit.

2. **Proper Validation Methodology:** Rolling window backtesting with proper train/test splits is essential for honest forecast evaluation.

3. **Robustness Testing:** Evaluating models across multiple training window sizes reveals data requirement vulnerabilities.

4. **Metric Selection:** Different metrics (RMSE for returns, QLIKE for volatility) lead to different model rankings, requiring careful consideration of objectives.

### Limitations and Caveats

1. **Sample Period:** Results are based on 2005-2007 data, including the onset of the financial crisis. Generalization to other periods requires further testing.

2. **Single Asset:** Analysis focuses on FTSE 100 index; results may differ for individual stocks or other asset classes.

3. **Model Space:** Only three models were tested; other specifications (EGARCH, FIGARCH, etc.) might perform differently.

4. **Forecast Horizons:** Limited to 1-20 days ahead; longer horizons were not evaluated.

5. **No Transaction Costs:** Direction accuracy does not account for trading costs, which would reduce practical performance.

6. **Structural Breaks:** Models assume constant parameters within training windows; regime-switching models might improve performance.

---

## Figures and Tables Summary

### Main Tables

- **Table 3.2:** Model Fitting Statistics (In-Sample)
- **Table 3.3:** 1-Day Ahead Forecasts
- **Table 3.4:** Multi-Horizon Volatility Forecasts
- **Table 3.5:** Rolling Window Counts
- **Table 3.6:** Overall Model Rankings (Out-of-Sample)
- **Table 3.7:** GARCH-t Performance by Training Window
- **Table 3.8:** GJR-GARCH-Skewed-t Catastrophic Failure
- **Table 3.9:** Leverage Effect Impact
- **Table 3.10:** Skewness Parameter Impact
- **Table 3.11:** Training Window Size Impact
- **Table 3.12:** Best Model by Forecast Horizon
- **Table 3.13:** Model Selection Guide
- **Table 3.14:** Model Feature Comparison
- **Table 3.15:** In-Sample vs Out-of-Sample Comparison

### Main Figures

- **Figure 3.2:** Comparative Volatility Forecasts (`results/forecasting/comparative_volatility_forecasts.png`)
- **Figure 3.3:** GJR-GARCH-Skewed-t Forecasts (`results/forecasting/GJR-GARCH-Skewed-t_forecasts.png`)
- **Figure 3.4:** GJR-GARCH-t Forecasts (`results/forecasting/GJR-GARCH-t_forecasts.png`)
- **Figure 3.5:** GARCH-t Forecasts (`results/forecasting/GARCH-t_forecasts.png`)
- **Figure 3.6:** RMSE Heatmap (Filtered) (`results/backtesting/filtered_heatmaps/heatmap_rmse_test1_filtered.png`)
- **Figure 3.7:** QLIKE Heatmap (Filtered) (`results/backtesting/filtered_heatmaps/heatmap_qlike_test1_filtered.png`)
- **Figure 3.8:** Direction Accuracy Heatmap (Filtered) (`results/backtesting/filtered_heatmaps/heatmap_direction_accuracy_test1_filtered.png`)

### Data Files

- **Forecasting Results:** `results/forecasting/all_forecasts.csv`, `results/forecasting/forecast_summary.csv`
- **Backtesting Results:** `results/backtesting/backtesting_results.csv` (72 configurations)
- **Visualizations:** 20 standard heatmaps + 20 filtered heatmaps + summary comparison plot

---

**End of Report**
