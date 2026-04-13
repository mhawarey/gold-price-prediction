# models/machine_learning/random_forest_model.py
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os
import pickle

class RandomForestModel:
    """Random Forest model for gold price prediction"""
    
    def __init__(self, lookback=30, n_estimators=100, max_depth=10, random_state=42):
        """Initialize Random Forest model
        
        Args:
            lookback (int): Number of previous time steps to use as input
            n_estimators (int): Number of trees in the forest
            max_depth (int): Maximum depth of the trees
            random_state (int): Random state for reproducibility
        """
        self.lookback = lookback
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'n_jobs': -1,
            'random_state': random_state
        }
        
        # Initialize models (one for each step ahead)
        self.models = [RandomForestRegressor(**self.params) for _ in range(5)]
    
    def _prepare_features(self, data, predict_ahead=5):
        """Prepare features for Random Forest model
        
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
            # Base features (lagged values)
            features = data[i:i + self.lookback]
            
            # Add derived features
            # Moving averages
            ma5 = np.mean(features[-5:]) if len(features) >= 5 else np.mean(features)
            ma10 = np.mean(features[-10:]) if len(features) >= 10 else np.mean(features)
            
            # Volatility
            vol5 = np.std(features[-5:]) if len(features) >= 5 else np.std(features)
            vol10 = np.std(features[-10:]) if len(features) >= 10 else np.std(features)
            
            # Price momentum (rate of change)
            momentum = (features[-1] / features[0] - 1) if features[0] != 0 else 0
            
            # Combine all features
            all_features = np.append(features, [ma5, ma10, vol5, vol10, momentum])
            
            X.append(all_features)
            y.append(data[i + self.lookback:i + self.lookback + predict_ahead])
            
        return np.array(X), np.array(y)
    
    def fit(self, data):
        """Fit Random Forest model to data
        
        Args:
            data (np.ndarray): Training data
        """
        try:
            # Prepare features
            X, y = self._prepare_features(data)
            
            if len(X) < 10:
                # Not enough data after feature preparation
                print("Not enough data for Random Forest training")
                return
                
            # Train a model for each prediction step
            for i in range(5):  # 5 steps ahead prediction
                # Extract target for this step
                y_step = y[:, i]
                
                # Train model
                self.models[i].fit(X, y_step)
                
        except Exception as e:
            print(f"Random Forest fitting error: {str(e)}")
    
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
                
            # Prepare input features
            if data.ndim > 1:
                data = data.flatten()
                
            features = data[-self.lookback:]
            
            # Add derived features
            ma5 = np.mean(features[-5:]) if len(features) >= 5 else np.mean(features)
            ma10 = np.mean(features[-10:]) if len(features) >= 10 else np.mean(features)
            vol5 = np.std(features[-5:]) if len(features) >= 5 else np.std(features)
            vol10 = np.std(features[-10:]) if len(features) >= 10 else np.std(features)
            momentum = (features[-1] / features[0] - 1) if features[0] != 0 else 0
            
            # Combine all features
            all_features = np.append(features, [ma5, ma10, vol5, vol10, momentum])
            X = all_features.reshape(1, -1)
            
            # Generate predictions for each step
            predictions = []
            for model in self.models:
                pred = model.predict(X)[0]
                predictions.append(pred)
                
            return np.array(predictions).reshape(-1, 1)
        except Exception as e:
            print(f"Random Forest prediction error: {str(e)}")
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
            print(f"Random Forest adaptation error: {str(e)}")
    
    def save(self, path='models/saved/rf_model'):
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
            print(f"Random Forest model save error: {str(e)}")
    
    def load(self, path='models/saved/rf_model'):
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
                
                print("Random Forest model loaded successfully")
        except Exception as e:
            print(f"Random Forest model load error: {str(e)}")