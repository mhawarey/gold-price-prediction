# models/model_manager.py - Update
import numpy as np
import pandas as pd
from models.statistical.arima_model import AdaptiveARIMA
from models.statistical.garch_model import GARCHModel
from models.statistical.exp_smoothing import ExponentialSmoothingModel
from models.machine_learning.lstm_model import GoldPriceLSTM
from models.machine_learning.xgboost_model import XGBoostModel
from models.machine_learning.svr_model import SVRModel
from models.machine_learning.random_forest_model import RandomForestModel
from models.ensemble.meta_learner import MetaLearner

class ModelManager:
    """Class for managing all prediction models"""
    
    def __init__(self, selected_models=None, lookback_window=30):
        """Initialize the model manager
        
        Args:
            selected_models (list, optional): List of model names to use
            lookback_window (int, optional): Lookback window for sequence models
        """
        self.lookback_window = lookback_window
        
        # Define available models
        self.available_models = {
            'ARIMA': AdaptiveARIMA(),
            'GARCH': GARCHModel(),
            'ExpSmoothing': ExponentialSmoothingModel(),
            'LSTM': GoldPriceLSTM(lookback=lookback_window),
            'XGBoost': XGBoostModel(lookback=lookback_window),
            'SVR': SVRModel(lookback=lookback_window),
            'RandomForest': RandomForestModel(lookback=lookback_window),
            # Add MetaLearner to available models but don't initialize yet
            # Will be initialized later when base models are ready
            'MetaLearner': None  
        }
        
        # Initialize selected models
        if selected_models is None:
            # Use all models by default except MetaLearner (will be added separately)
            self.selected_models = [model for model in self.available_models.keys() 
                                  if model != 'MetaLearner']
        else:
            # Validate selected models
            invalid_models = [m for m in selected_models if m not in self.available_models]
            if invalid_models:
                raise ValueError(f"Invalid model(s): {', '.join(invalid_models)}")
            
            # Remove MetaLearner from selected_models if present - will handle separately
            self.selected_models = [model for model in selected_models if model != 'MetaLearner']
            
        # Dictionary to store active models
        self.models = {name: self.available_models[name] for name in self.selected_models}
        
        # Initialize meta-learner (if requested and enough base models)
        self.use_meta_learner = False
        
        # Initialize ensemble weights
        self.ensemble_weights = {name: 1.0/len(self.models) for name in self.models}
        
        # Initialize meta-learner if it was requested
        if selected_models is not None and 'MetaLearner' in selected_models and len(self.models) >= 3:
            self._initialize_meta_learner()
    
    def _initialize_meta_learner(self):
        """Initialize the meta-learner with base models"""
        if len(self.models) >= 3:  # Need at least 3 base models for meta-learning
            self.use_meta_learner = True
            self.meta_learner = MetaLearner(base_models=list(self.models.values()))
            # Add meta-learner to models dictionary
            self.models['MetaLearner'] = self.meta_learner
    
    def train_models(self, train_data):
        """Train all selected models
        
        Args:
            train_data (np.ndarray): Training data
        """
        # Train regular models first
        for name, model in self.models.items():
            if name != 'MetaLearner':  # Skip meta-learner for now
                model.fit(train_data)
            
        # Initialize meta-learner if it wasn't done yet
        if 'MetaLearner' in self.selected_models and 'MetaLearner' not in self.models:
            self._initialize_meta_learner()
            
        # Train meta-learner if using
        if self.use_meta_learner and 'MetaLearner' in self.models:
            self.models['MetaLearner'].fit(train_data)
    
    def predict(self, data, steps=5):
        """Generate predictions using all selected models
        
        Args:
            data (np.ndarray): Input data
            steps (int, optional): Number of steps to predict
            
        Returns:
            dict: Dictionary mapping model names to predictions
        """
        predictions = {}
        
        for name, model in self.models.items():
            try:
                pred = model.predict(data, steps=steps)
                predictions[name] = pred
            except Exception as e:
                print(f"Error during prediction with {name}: {str(e)}")
                # Return zeros if prediction fails
                predictions[name] = np.zeros((steps, 1))
            
        return predictions
    
    def get_ensemble_prediction(self, predictions=None, weights=None):
        """Generate ensemble prediction from individual model predictions
        
        Args:
            predictions (dict, optional): Model predictions
            weights (dict, optional): Model weights
            
        Returns:
            np.ndarray: Ensemble prediction
        """
        if predictions is None:
            return None
            
        if weights is None:
            weights = self.ensemble_weights
            
        # Ensure all predictions have the same shape
        first_pred = next(iter(predictions.values()))
        ensemble_prediction = np.zeros_like(first_pred)
        
        # Create weighted average (excluding MetaLearner)
        total_weight = 0
        for model_name, prediction in predictions.items():
            if model_name != 'MetaLearner' and model_name in weights:
                weight = weights[model_name]
                ensemble_prediction += weight * prediction
                total_weight += weight
        
        # Normalize if not all weights were used
        if total_weight > 0 and total_weight != 1.0:
            ensemble_prediction /= total_weight
                
        return ensemble_prediction
    
    def predict_future(self, data, steps=5):
        """Generate future predictions
        
        Args:
            data (np.ndarray): Input data
            steps (int, optional): Number of steps to predict
            
        Returns:
            dict: Dictionary with model predictions and ensemble prediction
        """
        # Generate individual model predictions
        predictions = self.predict(data, steps=steps)
        
        # Generate ensemble prediction
        ensemble_prediction = self.get_ensemble_prediction(predictions, self.ensemble_weights)
        
        # Add ensemble to predictions
        predictions['Ensemble'] = ensemble_prediction
        
        return predictions
    
    def update_ensemble_weights(self, performances):
        """Update ensemble weights based on model performances
        
        Args:
            performances (dict): Dictionary mapping model names to performance metrics
        """
        if not performances:
            return
            
        # Extract relevant performance metric (e.g., RMSE)
        errors = {}
        for name, perf in performances.items():
            if name != 'MetaLearner':  # Exclude MetaLearner from ensemble weights
                if isinstance(perf, dict) and 'rmse' in perf:
                    errors[name] = perf['rmse']
                else:
                    errors[name] = float('inf')
        
        # Invert errors (lower error = higher weight)
        inverse_errors = {name: 1.0 / (err + 1e-10) for name, err in errors.items()}
        
        # Normalize weights to sum to 1
        total = sum(inverse_errors.values())
        if total > 0:
            self.ensemble_weights = {name: val / total for name, val in inverse_errors.items()}
    
    def adapt_models(self, train_data, validation_data):
        """Adapt models based on validation results
        
        Args:
            train_data (np.ndarray): Training data
            validation_data (np.ndarray): Validation data
        """
        for name, model in self.models.items():
            # Skip MetaLearner for adaptation if it doesn't have adapt method
            if name == 'MetaLearner' and not hasattr(model, 'adapt'):
                continue
                
            if hasattr(model, 'adapt'):
                try:
                    model.adapt(train_data, validation_data)
                except Exception as e:
                    print(f"Error adapting {name}: {str(e)}")