# models/statistical/exp_smoothing.py
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

class ExponentialSmoothingModel:
    """Exponential Smoothing model for gold price prediction"""
    
    def __init__(self, trend='add', seasonal=None, seasonal_periods=None):
        """Initialize Exponential Smoothing model
        
        Args:
            trend (str): Trend component type ('add', 'mul', or None)
            seasonal (str): Seasonal component type ('add', 'mul', or None)
            seasonal_periods (int): Number of periods in a season
        """
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.model = None
        self.result = None
        self.last_data_len = 0
    
    def fit(self, data):
        """Fit Exponential Smoothing model to data
        
        Args:
            data (np.ndarray): Training data
        """
        # Convert to pandas Series if numpy array
        if isinstance(data, np.ndarray):
            if data.ndim > 1:
                data = data.flatten()
            data = pd.Series(data)
            
        try:
            # Only refit if data changed significantly
            if self.model is None or abs(len(data) - self.last_data_len) > 5:
                # Try different model specifications if needed
                trend_options = [self.trend, None] if self.trend else [None, 'add']
                seasonal_options = [self.seasonal, None] if self.seasonal else [None]
                
                best_aic = float('inf')
                best_model = None
                best_result = None
                
                # Try different combinations
                for trend in trend_options:
                    for seasonal in seasonal_options:
                        try:
                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore")
                                
                                # If using seasonal, ensure we have enough data
                                if seasonal and self.seasonal_periods:
                                    if len(data) < 2 * self.seasonal_periods:
                                        continue
                                
                                # Create and fit model
                                model = ExponentialSmoothing(
                                    data,
                                    trend=trend,
                                    seasonal=seasonal,
                                    seasonal_periods=self.seasonal_periods,
                                    damped=False
                                )
                                result = model.fit(optimized=True)
                                
                                # Check if this model is better
                                if result.aic < best_aic:
                                    best_aic = result.aic
                                    best_model = model
                                    best_result = result
                        except:
                            continue
                
                # Use the best model found
                if best_model is not None:
                    self.model = best_model
                    self.result = best_result
                else:
                    # Fall back to simple exponential smoothing
                    self.model = ExponentialSmoothing(data, trend=None, seasonal=None)
                    self.result = self.model.fit()
            
            # Update last data length
            self.last_data_len = len(data)
            
        except Exception as e:
            print(f"Exponential Smoothing fitting error: {str(e)}")
            # Try simple exponential smoothing as fallback
            try:
                self.model = ExponentialSmoothing(data, trend=None, seasonal=None)
                self.result = self.model.fit()
            except:
                self.model = None
                self.result = None
    
    def predict(self, data, steps=5):
        """Make predictions
        
        Args:
            data (np.ndarray): Input data
            steps (int): Number of steps to predict
            
        Returns:
            np.ndarray: Predictions
        """
        if self.model is None or self.result is None:
            self.fit(data)
            
        if self.result is None:
            # Return zeros if fitting failed
            return np.zeros((steps, 1))
            
        try:
            # Forecast future values
            forecast = self.result.forecast(steps)
            return np.array(forecast).reshape(-1, 1)
        except Exception as e:
            print(f"Exponential Smoothing prediction error: {str(e)}")
            return np.zeros((steps, 1))
    
    def adapt(self, train_data, validation_data):
        """Adapt model based on validation data
        
        Args:
            train_data (np.ndarray): Training data
            validation_data (np.ndarray): Validation data
        """
        try:
            # Combine data
            combined_data = np.append(train_data.flatten(), validation_data.flatten())
            
            # Refit with combined data
            self.fit(combined_data)
        except Exception as e:
            print(f"Exponential Smoothing adaptation error: {str(e)}")