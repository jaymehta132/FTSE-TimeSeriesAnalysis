# FTSE-TimeSeriesAnalysis

Instructions for running-

For running forecasting part
---
Make sure you are in this directory where `scripts`, `results` and `data` are
```bash
python3 -m venv myenv
source venv/bin/activate
pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn
pip install arch
pip install scipy
pip install logging
pip install pyyaml
pip install tqdm

python3 scripts/forecasting.py
python3 scripts/backtesting.py
python3 scripts/replotting.py
```

Logs will be generated and stored in `logs/` and the plots and data will be stored in `results/`