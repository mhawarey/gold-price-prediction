# models/statistical/garch_model.py
import numpy as np
import pandas as pd
from arch import arch_model
import warnings

class GARCHModel:
    """GARCH model for gold price prediction and volatility"""
    
    def __init__(self, p=1, q=1, mean='constant'):
        """Initialize GARCH model
        
        Args:
            p (int): GARCH lag order
            q (int): ARCH lag order
            mean (str): Mean model type
        """
        self.p = p
        self.q = q
        self.mean = mean
        self.model = None
        self.result = None
        self.last_data_len = 0
        
    def _preprocess_data(self, data):
        """Preprocess data for GARCH modeling
        
        Args:
            data (np.ndarray): Input data
            
        Returns:
            pd.Series: Processed data (returns)
        """
        if isinstance(data, np.ndarray):
            if data.ndim > 1:
                data = data.flatten()
            data = pd.Series(data)
        
        # Calculate returns (GARCH models work with returns)
        returns = 100 * data.pct_change().dropna()
        
        # Replace extreme values
        returns = returns.clip(lower=returns.quantile(0.01), upper=returns.quantile(0.99))
            
        return returns
    
    def fit(self, data):
        """Fit GARCH model to data
        
        Args:
            data (np.ndarray): Training data
        """
        try:
            # Preprocess data
            returns = self._preprocess_data(data)
            
            if len(returns) < 10:
                # Not enough data after preprocessing
                print("Not enough data for GARCH fitting")
                return
                
            # Only refit if data changed significantly
            if self.model is None or abs(len(data) - self.last_data_len) > 5:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    # Create and fit GARCH model
                    self.model = arch_model(
                        returns,
                        p=self.p, q=self.q,
                        mean=self.mean,
                        vol='GARCH',
                        rescale=False
                    )
                self.result = self.model.fit(disp='off', show_warning=False)
            
            # Update last data length
            self.last_data_len = len(data)
            
        except Exception as e:
            print(f"GARCH fitting error: {str(e)}")
    
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
            # Preprocess data
            returns = self._preprocess_data(data)
            
            if len(returns) < 10:
                # Not enough data after preprocessing
                return np.zeros((steps, 1))
                
            # Get forecasts
            forecasts = self.result.forecast(horizon=steps)
            
            # Extract mean forecasts
            mean_forecasts = forecasts.mean.iloc[-1].values
            
            # Convert returns forecasts back to price changes
            last_price = data[-1]
            forecasted_prices = np.zeros(steps)
            
            # Calculate cumulative returns
            for i in range(steps):
                if i == 0:
                    forecasted_prices[i] = last_price * (1 + mean_forecasts[i]/100)
                else:
                    forecasted_prices[i] = forecasted_prices[i-1] * (1 + mean_forecasts[i]/100)
            
            return forecasted_prices.reshape(-1, 1)
        
        except Exception as e:
            print(f"GARCH prediction error: {str(e)}")
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
            print(f"GARCH adaptation error: {str(e)}")