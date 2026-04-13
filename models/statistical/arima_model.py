# models/statistical/arima_model.py
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from sklearn.metrics import mean_squared_error
import warnings

class AdaptiveARIMA:
    """Adaptive ARIMA model for gold price prediction"""
    
    def __init__(self, start_p=0, max_p=5, start_q=0, max_q=5, start_d=0, max_d=2):
        """Initialize the adaptive ARIMA model
        
        Args:
            start_p (int): Starting p value for auto ARIMA
            max_p (int): Maximum p value for auto ARIMA
            start_q (int): Starting q value for auto ARIMA
            max_q (int): Maximum q value for auto ARIMA
            start_d (int): Starting d value for auto ARIMA
            max_d (int): Maximum d value for auto ARIMA
        """
        self.start_p = start_p
        self.max_p = max_p
        self.start_q = start_q
        self.max_q = max_q
        self.start_d = start_d
        self.max_d = max_d
        
        # Current order
        self.order = None
        
        # Fitted model
        self.model = None
        
        # Last seen data length (to detect changes)
        self.last_data_len = 0
    
    def fit(self, data):
        """Fit ARIMA model to data
        
        Args:
            data (np.ndarray): Training data
        """
        # Convert to pandas Series if numpy array
        if isinstance(data, np.ndarray):
            if data.ndim > 1:
                data = data.flatten()
            data = pd.Series(data)
        
        try:
            # Only search for optimal parameters if data changed significantly
            # or if we haven't fit the model yet
            if self.order is None or abs(len(data) - self.last_data_len) > 5:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    # Use auto_arima to find optimal parameters
                    auto_model = pm.auto_arima(
                        data,
                        start_p=self.start_p, max_p=self.max_p,
                        start_q=self.start_q, max_q=self.max_q,
                        start_d=self.start_d, max_d=self.max_d,
                        seasonal=False,
                        stepwise=True,
                        suppress_warnings=True,
                        error_action='ignore',
                        trace=False
                    )
                    self.order = auto_model.order
                
            # Update last seen data length
            self.last_data_len = len(data)
            
            # Fit ARIMA with optimal parameters
            self.model = ARIMA(data, order=self.order).fit()
        except Exception as e:
            # Fall back to default order if auto_arima fails
            print(f"ARIMA fitting error: {str(e)}. Using default parameters.")
            try:
                self.order = (1, 1, 0)  # Simple default
                self.model = ARIMA(data, order=self.order).fit()
            except Exception as e2:
                # If that fails too, try even simpler model
                print(f"ARIMA fallback fitting error: {str(e2)}. Using random walk model.")
                try:
                    self.order = (0, 1, 0)  # Random walk
                    self.model = ARIMA(data, order=self.order).fit()
                except Exception as e3:
                    print(f"ARIMA final fallback error: {str(e3)}. ARIMA model will be unavailable.")
                    self.model = None
    
    def predict(self, data, steps=5):
        """Make predictions
        
        Args:
            data (np.ndarray): Input data
            steps (int): Number of steps to predict
            
        Returns:
            np.ndarray: Predictions
        """
        if self.model is None:
            try:
                self.fit(data)
            except Exception as e:
                print(f"Error fitting ARIMA model in predict: {str(e)}")
                return np.zeros((steps, 1))
            
        if self.model is None:  # If fitting still failed
            print("ARIMA model is not available. Returning zeros.")
            return np.zeros((steps, 1))
            
        try:
            # Forecast future values
            forecast = self.model.forecast(steps=steps)
            return np.array(forecast).reshape(-1, 1)
        except Exception as e:
            print(f"ARIMA prediction error: {str(e)}. Returning zeros.")
            return np.zeros((steps, 1))
    
    def adapt(self, train_data, validation_data):
        """Adapt model based on validation data
        
        Args:
            train_data (np.ndarray): Training data
            validation_data (np.ndarray): Validation data
        """
        # Try to improve parameters if needed
        if self.model is not None:
            try:
                # Make one-step forecasts
                forecasts = []
                for i in range(len(validation_data)):
                    # Update model with actual value
                    extended_train = np.append(train_data.flatten(), validation_data[:i].flatten())
                    forecast = self.model.apply(extended_train).forecast(steps=1)[0]
                    forecasts.append(forecast)
                
                # Calculate error
                mse = mean_squared_error(validation_data.flatten(), forecasts)
                
                # If error is high, refine parameters
                if mse > 0.1:  # Threshold can be adjusted
                    combined_data = np.append(train_data.flatten(), validation_data.flatten())
                    # Re-fit with combined data and adjusted parameters
                    self.fit(combined_data)
                else:
                    # Just update with new data without changing parameters
                    combined_data = np.append(train_data.flatten(), validation_data.flatten())
                    try:
                        self.model = ARIMA(combined_data, order=self.order).fit(disp=False)
                    except Exception as e:
                        print(f"ARIMA adaptation update error: {str(e)}")
            except Exception as e:
                print(f"ARIMA adaptation error: {str(e)}")