# models/machine_learning/svr_model.py
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os
import pickle

class SVRModel:
    """Support Vector Regression model for gold price prediction"""
    
    def __init__(self, lookback=30, kernel='rbf', C=1.0, epsilon=0.1, gamma='scale'):
        """Initialize SVR model
        
        Args:
            lookback (int): Number of previous time steps to use as input
            kernel (str): Kernel type to be used
            C (float): Regularization parameter
            epsilon (float): Epsilon in the epsilon-SVR model
            gamma (str or float): Kernel coefficient
        """
        self.lookback = lookback
        self.params = {
            'kernel': kernel,
            'C': C,
            'epsilon': epsilon,
            'gamma': gamma
        }
        
        # Initialize models (one for each step ahead)
        self.models = [
            Pipeline([
                ('scaler', StandardScaler()),
                ('svr', SVR(**self.params))
            ]) for _ in range(5)  # 5 models for 5 steps ahead
        ]
    
    def _prepare_features(self, data, predict_ahead=5):
        """Prepare features for SVR model
        
        Args:
            data (np.ndarray): Input data
            predict_ahead (int): Number of steps ahead to predict
            
        Returns:
            tuple: (X, y) features and targets
        """
        if len(data) <= self.lookback + predict_ahead:
            raise ValueError(f"Not enough data for lookback and prediction. Need at least {self.lookback + predict_ahead + 1} points.")
            
        # Flatten data if needed
        if data.ndim > 1:
            data = data.flatten()
            
        X, y = [], []
        
        for i in range(len(data) - self.lookback - predict_ahead + 1):
            X.append(data[i:i + self.lookback])
            y.append(data[i + self.lookback:i + self.lookback + predict_ahead])
            
        return np.array(X), np.array(y)
    
    def fit(self, data):
        """Fit SVR model to data
        
        Args:
            data (np.ndarray): Training data
        """
        try:
            # Prepare features
            X, y = self._prepare_features(data)
            
            if len(X) < 10:
                # Not enough data after feature preparation
                print("Not enough data for SVR training")
                return
                
            # Train a model for each prediction step
            for i in range(5):  # 5 steps ahead prediction
                # Extract target for this step
                y_step = y[:, i]
                
                # Train model
                self.models[i].fit(X, y_step)
                
        except Exception as e:
            print(f"SVR fitting error: {str(e)}")
    
    def predict(self, data, steps=5):
        """Make predictions
        
        Args:
            data (np.ndarray): Input data
            steps (int): Number of steps to predict (ignored, always predicts 5 steps)
            
        Returns:
            np.ndarray: Predictions
        """
        try:
            # Use the last lookback points as input
            if len(data) < self.lookback:
                # Not enough data
                return np.zeros((5, 1))
                
            # Prepare input sequence
            if data.ndim > 1:
                data = data.flatten()
                
            X = data[-self.lookback:].reshape(1, -1)
            
            # Generate predictions for each step
            predictions = []
            for model in self.models:
                pred = model.predict(X)[0]
                predictions.append(pred)
                
            return np.array(predictions).reshape(-1, 1)
        except Exception as e:
            print(f"SVR prediction error: {str(e)}")
            return np.zeros((5, 1))
    
    def adapt(self, train_data, validation_data):
        """Adapt model based on validation data
        
        Args:
            train_data (np.ndarray): Training data
            validation_data (np.ndarray): Validation data
        """
        try:
            # Combine data
            combined_data = np.vstack((train_data, validation_data))
            
            # Update model with combined data
            self.fit(combined_data)
        except Exception as e:
            print(f"SVR adaptation error: {str(e)}")
    
    def save(self, path='models/saved/svr_model'):
        """Save model to disk
        
        Args:
            path (str): Path to save the model
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save model
            with open(path, 'wb') as f:
                pickle.dump({
                    'models': self.models,
                    'params': self.params,
                    'lookback': self.lookback
                }, f)
        except Exception as e:
            print(f"SVR model save error: {str(e)}")
    
    def load(self, path='models/saved/svr_model'):
        """Load model from disk
        
        Args:
            path (str): Path to load the model from
        """
        try:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    model_data = pickle.load(f)
                    
                self.models = model_data['models']
                self.params = model_data['params']
                self.lookback = model_data['lookback']
                
                print("SVR model loaded successfully")
        except Exception as e:
            print(f"SVR model load error: {str(e)}")