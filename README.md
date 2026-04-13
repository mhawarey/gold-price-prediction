# Gold Price Prediction System

An ensemble machine learning platform for gold price forecasting, combining statistical models (ARIMA, GARCH, Exponential Smoothing), ML models (XGBoost, Random Forest, SVR, LSTM), and meta-learning ensemble methods.

## Architecture

```
├── main.py                        # Application entry point (Tkinter GUI)
├── setup.py                       # Package setup
├── requirements.txt
├── golddata.csv                   # Sample dataset
├── assets/
│   └── a2.ico                     # Application icon
├── data/
│   ├── loader.py                  # Data loading & ingestion
│   ├── preprocessor.py            # Feature engineering & normalization
│   └── window_manager.py          # Sliding window generation
├── models/
│   ├── model_manager.py           # Unified model training & inference
│   ├── statistical/
│   │   ├── arima_model.py
│   │   ├── exp_smoothing.py
│   │   └── garch_model.py
│   ├── machine_learning/
│   │   ├── lstm_model.py
│   │   ├── random_forest_model.py
│   │   ├── svr_model.py
│   │   └── xgboost_model.py
│   └── ensemble/
│       ├── meta_learner.py        # Stacking meta-learner
│       └── weighted_ensemble.py   # Weighted model combination
├── evaluation/
│   ├── evaluator.py               # Metrics & backtesting
│   └── visualizer.py              # Result plotting
└── gui/
    └── dashboard.py               # Interactive prediction dashboard
```

## Models

- **Statistical**: ARIMA, GARCH, Exponential Smoothing
- **Machine Learning**: LSTM, XGBoost, Random Forest, SVR
- **Ensemble**: Weighted combination + stacking meta-learner

## Requirements

```
pip install -r requirements.txt
```

Key dependencies: TensorFlow, XGBoost, scikit-learn, statsmodels, pmdarima, arch, pandas, matplotlib

## Usage

```bash
python main.py
```

Load gold price data, select models, configure sliding window parameters, and generate forecasts through the interactive dashboard.

## Author

**Dr. Mosab Hawarey** — [github.com/mhawarey](https://github.com/mhawarey)

## License

MIT License
