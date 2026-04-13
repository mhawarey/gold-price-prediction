# evaluation/evaluator.py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

class Evaluator:
    """Class for evaluating model performance"""
    
    def __init__(self):
        """Initialize evaluator"""
        self.metrics = {
            'rmse': self._calculate_rmse,
            'mae': self._calculate_mae,
            'mape': self._calculate_mape,
            'r2': self._calculate_r2,
            'direction_accuracy': self._calculate_direction_accuracy
        }
    
    def evaluate(self, actual, predicted):
        """Evaluate model performance
        
        Args:
            actual (np.ndarray): Actual values
            predicted (np.ndarray): Predicted values
            
        Returns:
            dict: Dictionary of performance metrics
        """
        # Ensure data is in the right shape
        actual = self._reshape_data(actual)
        predicted = self._reshape_data(predicted)
        
        # If lengths are different, trim to the shorter one
        min_length = min(len(actual), len(predicted))
        actual = actual[:min_length]
        predicted = predicted[:min_length]
        
        # Calculate all metrics
        results = {}
        for name, metric_fn in self.metrics.items():
            try:
                results[name] = metric_fn(actual, predicted)
            except Exception as e:
                print(f"Error calculating {name}: {str(e)}")
                results[name] = None
                
        return results
    
    def _reshape_data(self, data):
        """Reshape data to 1D array
        
        Args:
            data (np.ndarray): Input data
            
        Returns:
            np.ndarray: Reshaped data
        """
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            data = data.values
            
        if isinstance(data, list):
            data = np.array(data)
            
        # Flatten if more than 1D
        if data.ndim > 1:
            data = data.flatten()
            
        return data
    
    def _calculate_rmse(self, actual, predicted):
        """Calculate Root Mean Squared Error
        
        Args:
            actual (np.ndarray): Actual values
            predicted (np.ndarray): Predicted values
            
        Returns:
            float: RMSE value
        """
        return np.sqrt(mean_squared_error(actual, predicted))
    
    def _calculate_mae(self, actual, predicted):
        """Calculate Mean Absolute Error
        
        Args:
            actual (np.ndarray): Actual values
            predicted (np.ndarray): Predicted values
            
        Returns:
            float: MAE value
        """
        return mean_absolute_error(actual, predicted)
    
    def _calculate_mape(self, actual, predicted):
        """Calculate Mean Absolute Percentage Error
        
        Args:
            actual (np.ndarray): Actual values
            predicted (np.ndarray): Predicted values
            
        Returns:
            float: MAPE value
        """
        # Avoid division by zero
        mask = actual != 0
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    
    def _calculate_r2(self, actual, predicted):
        """Calculate R-squared score
        
        Args:
            actual (np.ndarray): Actual values
            predicted (np.ndarray): Predicted values
            
        Returns:
            float: R-squared value
        """
        return r2_score(actual, predicted)
    
    def _calculate_direction_accuracy(self, actual, predicted):
        """Calculate direction accuracy
        
        Args:
            actual (np.ndarray): Actual values
            predicted (np.ndarray): Predicted values
            
        Returns:
            float: Direction accuracy as percentage
        """
        # Calculate direction of change
        actual_diff = np.diff(actual)
        predicted_diff = np.diff(predicted)
        
        # Count correct directions
        correct_dirs = (np.sign(actual_diff) == np.sign(predicted_diff)).sum()
        
        # Return as percentage
        if len(actual_diff) > 0:
            return (correct_dirs / len(actual_diff)) * 100
        else:
            return 0.0
    
    def compare_models(self, actual, predictions, names=None):
        """Compare multiple model predictions
        
        Args:
            actual (np.ndarray): Actual values
            predictions (list or dict): List or dictionary of predictions
            names (list, optional): List of model names
            
        Returns:
            pd.DataFrame: Comparison table of metrics
        """
        results = []
        
        # Convert predictions to dictionary if it's a list
        if isinstance(predictions, list):
            if names is None:
                names = [f"Model {i+1}" for i in range(len(predictions))]
            predictions = dict(zip(names, predictions))
        
        # Evaluate each model
        for name, pred in predictions.items():
            metrics = self.evaluate(actual, pred)
            metrics['model'] = name
            results.append(metrics)
            
        # Create DataFrame
        if results:
            return pd.DataFrame(results).set_index('model')
        else:
            return pd.DataFrame()
    
    def plot_predictions(self, actual, predicted, model_name=None, ax=None):
        """Plot actual vs predicted values
        
        Args:
            actual (np.ndarray): Actual values
            predicted (np.ndarray): Predicted values
            model_name (str, optional): Name of the model
            ax (matplotlib.axes, optional): Axes to plot on
            
        Returns:
            matplotlib.axes: The axes with the plot
        """
        # Reshape data
        actual = self._reshape_data(actual)
        predicted = self._reshape_data(predicted)
        
        # If lengths are different, trim to the shorter one
        min_length = min(len(actual), len(predicted))
        actual = actual[:min_length]
        predicted = predicted[:min_length]
        
        # Create plot
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        # Plot data
        ax.plot(actual, 'b-', label='Actual')
        ax.plot(predicted, 'r--', label='Predicted')
        
        # Set labels and title
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        
        if model_name:
            ax.set_title(f'{model_name} - Actual vs Predicted')
        else:
            ax.set_title('Actual vs Predicted')
            
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_multiple_predictions(self, actual, predictions, names=None, figsize=(12, 8)):
        """Plot actual vs multiple model predictions
        
        Args:
            actual (np.ndarray): Actual values
            predictions (list or dict): List or dictionary of predictions
            names (list, optional): List of model names
            figsize (tuple, optional): Figure size
            
        Returns:
            matplotlib.figure: The figure with the plots
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Reshape actual data
        actual = self._reshape_data(actual)
        
        # Plot actual data
        ax.plot(actual, 'k-', linewidth=2, label='Actual')
        
        # Convert predictions to dictionary if it's a list
        if isinstance(predictions, list):
            if names is None:
                names = [f"Model {i+1}" for i in range(len(predictions))]
            predictions = dict(zip(names, predictions))
        
        # Plot each prediction
        colors = plt.cm.tab10.colors
        for i, (name, pred) in enumerate(predictions.items()):
            # Reshape prediction
            pred = self._reshape_data(pred)
            
            # Trim to actual data length
            min_length = min(len(actual), len(pred))
            plot_actual = actual[:min_length]
            plot_pred = pred[:min_length]
            
            # Plot prediction
            ax.plot(plot_pred, '--', color=colors[i % len(colors)], 
                    alpha=0.8, label=name)
        
        # Set labels and title
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.set_title('Model Predictions Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_error_distribution(self, actual, predicted, model_name=None, ax=None):
        """Plot error distribution
        
        Args:
            actual (np.ndarray): Actual values
            predicted (np.ndarray): Predicted values
            model_name (str, optional): Name of the model
            ax (matplotlib.axes, optional): Axes to plot on
            
        Returns:
            matplotlib.axes: The axes with the plot
        """
        # Reshape data
        actual = self._reshape_data(actual)
        predicted = self._reshape_data(predicted)
        
        # If lengths are different, trim to the shorter one
        min_length = min(len(actual), len(predicted))
        actual = actual[:min_length]
        predicted = predicted[:min_length]
        
        # Calculate errors
        errors = actual - predicted
        
        # Create plot
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        # Plot histogram
        ax.hist(errors, bins=20, alpha=0.7, color='skyblue')
        
        # Add a vertical line at zero
        ax.axvline(0, color='red', linestyle='--', linewidth=1)
        
        # Set labels and title
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Frequency')
        
        if model_name:
            ax.set_title(f'{model_name} - Error Distribution')
        else:
            ax.set_title('Error Distribution')
            
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_learning_curve(self, iterations, performances, metric='rmse', figsize=(10, 6)):
        """Plot learning curve across iterations
        
        Args:
            iterations (list): List of iteration numbers
            performances (list): List of performance dictionaries
            metric (str): Metric to plot
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure: The figure with the plot
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract models
        models = set()
        for perf in performances:
            models.update(perf.keys())
            
        # Remove non-model keys
        models = [m for m in models if m not in ['iteration', 'date']]
        
        # Plot learning curve for each model
        colors = plt.cm.tab10.colors
        for i, model in enumerate(models):
            values = []
            for perf in performances:
                if model in perf and metric in perf[model]:
                    values.append(perf[model][metric])
                else:
                    values.append(None)
                    
            # Plot
            ax.plot(iterations, values, 'o-', color=colors[i % len(colors)], 
                   label=model)
        
        # Set labels and title
        ax.set_xlabel('Iteration')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'Learning Curve - {metric.upper()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig