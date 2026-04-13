# models/ensemble/weighted_ensemble.py
import numpy as np

class WeightedEnsemble:
    """Weighted ensemble for combining multiple model predictions"""
    
    def __init__(self, models=None, initial_weights=None):
        """Initialize weighted ensemble
        
        Args:
            models (dict): Dictionary mapping model names to model objects
            initial_weights (dict): Dictionary mapping model names to initial weights
        """
        self.models = models if models else {}
        
        # Initialize weights
        if initial_weights:
            self.weights = initial_weights
        else:
            # Equal weights by default
            n_models = len(self.models)
            if n_models > 0:
                self.weights = {name: 1.0 / n_models for name in self.models}
            else:
                self.weights = {}
                
        # Performance history for adaptation
        self.performance_history = {}
    
    def predict(self, data, steps=5):
        """Make ensemble predictions
        
        Args:
            data (np.ndarray): Input data
            steps (int): Number of steps to predict
            
        Returns:
            np.ndarray: Ensemble predictions
        """
        if not self.models:
            return np.zeros((steps, 1))
            
        # Get predictions from all models
        predictions = {}
        for name, model in self.models.items():
            try:
                pred = model.predict(data, steps=steps)
                predictions[name] = pred
            except Exception as e:
                print(f"Error in {name} prediction: {str(e)}")
                # Skip if prediction fails
                
        if not predictions:
            return np.zeros((steps, 1))
            
        # Create weighted ensemble prediction
        ensemble_pred = np.zeros((steps, 1))
        total_weight = 0
        
        for name, pred in predictions.items():
            if name in self.weights:
                weight = self.weights[name]
                ensemble_pred += weight * pred
                total_weight += weight
                
        # Normalize if weights don't sum to 1
        if total_weight > 0 and total_weight != 1.0:
            ensemble_pred /= total_weight
            
        return ensemble_pred
    
    def update_weights(self, performances):
        """Update weights based on model performances
        
        Args:
            performances (dict): Dictionary mapping model names to performance metrics
        """
        if not performances:
            return
            
        # Update performance history
        for name, perf in performances.items():
            if name not in self.performance_history:
                self.performance_history[name] = []
            self.performance_history[name].append(perf)
            
        # Extract RMSE as the performance metric
        errors = {}
        for name, perf in performances.items():
            if isinstance(perf, dict) and 'rmse' in perf:
                errors[name] = perf['rmse']
            elif isinstance(perf, (int, float)):
                errors[name] = perf
                
        if not errors:
            return
            
        # Compute inverse errors (lower error = higher weight)
        inverse_errors = {}
        for name, error in errors.items():
            if error > 0:
                inverse_errors[name] = 1.0 / error
            else:
                inverse_errors[name] = 1.0
                
        # Normalize weights to sum to 1
        total = sum(inverse_errors.values())
        if total > 0:
            self.weights = {name: val / total for name, val in inverse_errors.items()}
    
    def adaptive_update(self, alpha=0.3):
        """Update weights using exponential decay of historical performance
        
        Args:
            alpha (float): Learning rate for weight updates
        """
        if not self.performance_history:
            return
            
        # Calculate weighted average of historical performance
        weighted_errors = {}
        
        for name, history in self.performance_history.items():
            if not history:
                continue
                
            # Extract RMSE from each performance record
            errors = []
            for perf in history:
                if isinstance(perf, dict) and 'rmse' in perf:
                    errors.append(perf['rmse'])
                elif isinstance(perf, (int, float)):
                    errors.append(perf)
                    
            if not errors:
                continue
                
            # More recent errors have higher weight
            weighted_error = 0
            total_weight = 0
            
            for i, error in enumerate(errors):
                weight = (1 - alpha) ** (len(errors) - i - 1)
                weighted_error += weight * error
                total_weight += weight
                
            if total_weight > 0:
                weighted_errors[name] = weighted_error / total_weight
                
        if not weighted_errors:
            return
            
        # Compute inverse errors
        inverse_errors = {}
        for name, error in weighted_errors.items():
            if error > 0:
                inverse_errors[name] = 1.0 / error
            else:
                inverse_errors[name] = 1.0
                
        # Normalize weights to sum to 1
        total = sum(inverse_errors.values())
        if total > 0:
            self.weights = {name: val / total for name, val in inverse_errors.items()}