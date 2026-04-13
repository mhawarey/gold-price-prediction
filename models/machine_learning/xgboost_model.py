# models/machine_learning/xgboost_model.py
import numpy as np
import xgboost as xgb
import os
import pickle

class XGBoostModel:
    """XGBoost model for gold price prediction"""
    
    def __init__(self, lookback=30, max_depth=5, learning_rate=0.1, n_estimators=100):
        """Initialize XGBoost model
        
        Args:
            lookback (int): Number of previous time steps to use as input
            max_depth (int): Maximum tree depth
            learning_rate (float): Learning rate
            n_estimators (int): Number of boosting rounds
        """
        self.lookback = lookback
        self.params = {
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'objective': 'reg:squarederror',
            'n_jobs': -1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        # Initialize model
        self.model = xgb.XGBRegressor(**self.params)
        
        # Store feature names (for saving/loading)
        self.feature_names = [f'lag_{i+1}' for i in range(lookback)]
    
    def _prepare_features(self, data, predict_ahead=5):
        """Prepare features for XGBoost model
        
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
            features = data[i:i + self.lookback]
            
            # Add additional features
            # 1. Moving averages
            features_ma5 = np.mean(features[-5:]) if len(features) >= 5 else np.mean(features)
            features_ma10 = np.mean(features[-10:]) if len(features) >= 10 else np.mean(features)
            
            # 2. Volatility (standard deviation)
            features_std5 = np.std(features[-5:]) if len(features) >= 5 else np.std(features)
            
            # Combine all features
            all_features = np.append(features, [features_ma5, features_ma10, features_std5])
            
            X.append(all_features)
            y.append(data[i + self.lookback:i + self.lookback + predict_ahead])
            
        return np.array(X), np.array(y)
    
    def fit(self, data, validation_split=0.1):
        """Fit XGBoost model to data
        
        Args:
            data (np.ndarray): Training data
            validation_split (float): Fraction of data to use for validation
        """
        try:
            # Prepare features
            X, y = self._prepare_features(data)
            
            if len(X) < 10:
                # Not enough data after feature preparation
                print("Not enough data for XGBoost training")
                return
                
            # Update feature names to include additional features
            self.feature_names = [f'lag_{i+1}' for i in range(self.lookback)]
            self.feature_names.extend(['ma5', 'ma10', 'std5'])
                
            # Train model for each step ahead
            self.models = []
            for i in range(5):  # 5 steps ahead prediction
                # Extract target for this step
                y_step = y[:, i]
                
                # Create and train model
                model = xgb.XGBRegressor(**self.params)
                model.fit(
                    X, y_step,
                    eval_set=[(X[-int(len(X) * validation_split):], 
                              y_step[-int(len(X) * validation_split):])],
                    verbose=False
                )
                
                self.models.append(model)
        except Exception as e:
            print(f"XGBoost fitting error: {str(e)}")
    
    def predict(self, data, steps=5):
        """Make predictions
        
        Args:
            data (np.ndarray): Input data
            steps (int): Number of steps to predict (ignored, always predicts 5 steps)
            
        Returns:
            np.ndarray: Predictions
        """
        try:
            # Check if models have been trained
            if not hasattr(self, 'models') or not self.models:
                # Not trained yet
                return np.zeros((5, 1))
                
            # Use the last lookback points as input
            if len(data) < self.lookback:
                # Not enough data
                return np.zeros((5, 1))
                
            # Prepare input features
            if data.ndim > 1:
                data = data.flatten()
                
            features = data[-self.lookback:]
            
            # Calculate additional features
            features_ma5 = np.mean(features[-5:]) if len(features) >= 5 else np.mean(features)
            features_ma10 = np.mean(features[-10:]) if len(features) >= 10 else np.mean(features)
            features_std5 = np.std(features[-5:]) if len(features) >= 5 else np.std(features)
            
            # Combine all features
            all_features = np.append(features, [features_ma5, features_ma10, features_std5])
            X = all_features.reshape(1, -1)
            
            # Generate predictions for each step
            predictions = []
            for model in self.models:
                pred = model.predict(X)[0]
                predictions.append(pred)
                
            return np.array(predictions).reshape(-1, 1)
        except Exception as e:
            print(f"XGBoost prediction error: {str(e)}")
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
            
            # Create a temporary copy of the parameters with fewer estimators
            adapt_params = self.params.copy()
            adapt_params['n_estimators'] = 20  # Fewer rounds for fine-tuning
            
            # Prepare features
            X, y = self._prepare_features(combined_data)
            
            if not hasattr(self, 'models') or not self.models:
                # Full training if not yet trained
                self.fit(combined_data)
                return
                
            # Fine-tune each model
            for i, model in enumerate(self.models):
                # Extract target for this step
                y_step = y[:, i]
                
                # Fine-tune with combined data
                model.fit(
                    X, y_step,
                    eval_set=[(X[-len(validation_data):], 
                              y_step[-len(validation_data):])],
                    verbose=False,
                    xgb_model=model  # Use existing model as starting point
                )
        except Exception as e:
            print(f"XGBoost adaptation error: {str(e)}")
    
    def save(self, path='models/saved/xgboost_model'):
        """Save model to disk
        
        Args:
            path (str): Path to save the model
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save models
            if hasattr(self, 'models') and self.models:
                with open(path, 'wb') as f:
                    pickle.dump({
                        'models': self.models,
                        'params': self.params,
                        'feature_names': self.feature_names,
                        'lookback': self.lookback
                    }, f)
        except Exception as e:
            print(f"XGBoost model save error: {str(e)}")
    
    def load(self, path='models/saved/xgboost_model'):
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
                self.feature_names = model_data['feature_names']
                self.lookback = model_data['lookback']
                
                print("XGBoost model loaded successfully")
        except Exception as e:
            print(f"XGBoost model load error: {str(e)}")