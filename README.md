# FTSE-TimeSeriesAnalysis

## Instructions for running-
For running the scripts, ensure you have a python virtual environment with Python 3.12+. Given below are the steps to make a venv. Ensure Python 3.12 is installed. 
```bash
python3.12 -m venv ec602
```
Here `ec602` is the name of the virtual environment. This will create a directory of the name of the environment in the working directory. To activate the environment, follow the steps given below
```bash
source ec602/bin/activate # For macOS/Linux
.\ec602\Scripts\activate # For Windows Powershell
```
To deactivate the environment, run
```bash
deactivate # For macOS/Linux/Windows Powershell
```  
Once you are inside the environment, run the following command to install all the necessary libraries
```bash
pip install -r requirements.txt
```
### For running Data Preprocessing, EDA and Analysis scripts:
---
Make sure you are in this directory where the `scripts`, `results` and `data` are present.
```bash
python -m scripts.dataPreprocessing
python -m scripts.EDA
python -m scripts.analysis
```
- Data Preprocessing script preprocesses the raw data and produces the FTSE_PreprocessedData.csv file which contains the columns `Prices`, `Returns` and `Log Returns`.
- EDA script performs exploratory data analysis on the preprocessed data. It runs different tests on the `Returns` and `Log Returns` columns to check for stationarity and other properties like Heteroskedasticity.
- Analysis script tests a simple model and plots different metrics with respect to the results of the model.

Results for `EDA.py` are stored in `results/eda/` while results for `analysis.py` are stored in `results/analysis/`. Logs are created and stored in the `logs/` folder.
### For running the model selection part
---
Make sure you are in the `scripts/` directory. Ensure that you have set up the dependencies from requirements.txt on your system an run the following two commands: 

```bash
python3 model_grid_search.py
python3 advanced_model_grid_search.py
```

- `scripts\model_grid_search.py` searches over simple models keeping in mind the insights obtained in the EDA phase to obtain the best combination of parameters and distributions 
- `scripts\advanced_model_grid_search.py` searches over more complicated volatility models keeping the returns model fixed to better explain volatility effects and figure out optimal parameter settings

Plots and results will be stored in `results/model_grid_outputs/` and `results/model_grid_advanced_outputs/` respectively.

### For running forecasting part
---
Make sure you are in this directory where `scripts`, `results` and `data` are present.
```bash
python3 scripts/forecasting.py
python3 scripts/backtesting.py
python3 scripts/replotting.py
```
- `forecasting.py` - trains the models chosen (as mentioned in the report) on the entire dataset and creates sample forecasts for analysis and stores it in `results/forecasting`
- `backtesting.py` - Analyses the models rigorously on a rolling window basis and outputs the average values for proper comparisons and stores it in `results/backtesting`
- `replotting.py` - Takes the output of `backtesting.py` and replots the plots to make them more visually clear - since the GJR-Garch-Skewed-t model fails on training size 50 and stores it in `results/backtesting/filtered_heatmaps`

Logs will be generated and stored in `logs/` and the plots and data will be stored in `results/`