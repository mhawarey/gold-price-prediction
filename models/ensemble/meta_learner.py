# models/ensemble/meta_learner.py
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import os
import pickle

class MetaLearner:
    """Meta-learner model that learns from base model predictions"""
    
    def __init__(self, base_models=None, n_estimators=100, learning_rate=0.1, max_depth=3):
        """Initialize Meta-learner
        
        Args:
            base_models (list): List of base model objects
            n_estimators (int): Number of boosting stages
            learning_rate (float): Learning rate
            max_depth (int): Maximum depth of individual regression estimators
        """
        self.base_models = base_models if base_models else []
        self.params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'random_state': 42
        }
        
        # Initialize meta models (one for each step ahead)
        self.meta_models = [GradientBoostingRegressor(**self.params) for _ in range(5)]
    
    def _collect_base_predictions(self, data):
        """Collect predictions from all base models
        
        Args:
            data (np.ndarray): Input data
            
        Returns:
            np.ndarray: Array of base model predictions
        """
        # Get predictions from each base model
        base_predictions = []
        
        for model in self.base_models:
            try:
                pred = model.predict(data)
                base_predictions.append(pred)
            except Exception as e:
                print(f"Error getting prediction from base model: {str(e)}")
                # Add zeros if prediction fails
                pred = np.zeros((5, 1))
                base_predictions.append(pred)
        
        # Reshape and combine predictions
        # Shape will be (n_samples, n_models * 5)
        X_meta = []
        
        # Create training samples from overlapping windows
        for i in range(len(data) - 5):
            # Extract actual values for this window
            actual = data[i:i+5].flatten()
            
            # Collect predictions from all models for this point
            sample_predictions = []
            for model_preds in base_predictions:
                if i < len(model_preds):
                    sample_predictions.extend(model_preds[i].flatten())
                else:
                    # Padding if needed
                    sample_predictions.extend([0] * 5)
            
            # Add actual value as a feature
            features = np.append(sample_predictions, actual[-1])
            
            X_meta.append(features)
        
        if not X_meta:
            return np.array([])
            
        return np.array(X_meta)
    
    def fit(self, data):
        """Fit meta-learner to data
        
        Args:
            data (np.ndarray): Training data
        """
        try:
            if not self.base_models:
                print("No base models provided. Cannot train meta-learner.")
                return
                
            # Ensure data has enough samples
            if len(data) < 15:  # Need at least enough for a few samples
                print("Not enough data for meta-learner training")
                return
                
            # Collect base model predictions
            X_meta = self._collect_base_predictions(data)
            
            if len(X_meta) < 5:
                print("Not enough meta features for training")
                return
                
            # Prepare target values (next 5 future values for each sample)
            y_meta = []
            for i in range(len(X_meta)):
                # Use the next 5 values after this sample as targets
                if i + 10 <= len(data):  # +5 from the window, +5 for prediction
                    y_meta.append(data[i+5:i+10].flatten())
                
            if not y_meta:
                print("Not enough target values for meta-learner")
                return
                
            y_meta = np.array(y_meta)
            
            # Ensure X and y have same number of samples
            min_samples = min(len(X_meta), len(y_meta))
            X_meta = X_meta[:min_samples]
            y_meta = y_meta[:min_samples]
            
            # Train a meta-model for each prediction step
            for i in range(5):  # 5 steps ahead prediction
                # Extract target for this step
                y_step = y_meta[:, i]
                
                # Train model
                self.meta_models[i].fit(X_meta, y_step)
                
        except Exception as e:
            print(f"Meta-learner fitting error: {str(e)}")
    
    def predict(self, data, steps=5):
        """Make meta-predictions
        
        Args:
            data (np.ndarray): Input data
            steps (int): Number of steps to predict (ignored, always predicts 5 steps)
            
        Returns:
            np.ndarray: Meta-predictions
        """
        try:
            if not self.base_models or not self.meta_models:
                # Not trained yet
                return np.zeros((5, 1))
                
            # Get the last available window of data
            if len(data) < 5:
                return np.zeros((5, 1))
                
            # Collect predictions from base models
            base_preds = []
            for model in self.base_models:
                try:
                    pred = model.predict(data).flatten()
                    base_preds.extend(pred)
                except:
                    # Add zeros if prediction fails
                    base_preds.extend([0] * 5)
            
            # Add last actual value as feature
            last_value = data[-1] if data.ndim == 1 else data[-1, 0]
            X_meta = np.append(base_preds, last_value).reshape(1, -1)
            
            # Generate meta-predictions
            meta_predictions = []
            for model in self.meta_models:
                pred = model.predict(X_meta)[0]
                meta_predictions.append(pred)
                
            return np.array(meta_predictions).reshape(-1, 1)
        except Exception as e:
            print(f"Meta-learner prediction error: {str(e)}")
            return np.zeros((5, 1))
    
    def save(self, path='models/saved/meta_learner'):
        """Save model to disk
        
        Args:
            path (str): Path to save the model
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save model (only the meta-models, not the base models)
            with open(path, 'wb') as f:
                pickle.dump({
                    'meta_models': self.meta_models,
                    'params': self.params
                }, f)
        except Exception as e:
            print(f"Meta-learner save error: {str(e)}")
    
    def load(self, path='models/saved/meta_learner'):
        """Load model from disk
        
        Args:
            path (str): Path to load the model from
        """
        try:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    model_data = pickle.load(f)
                    
                self.meta_models = model_data['meta_models']
                self.params = model_data['params']
                
                print("Meta-learner loaded successfully")
        except Exception as e:
            print(f"Meta-learner load error: {str(e)}")