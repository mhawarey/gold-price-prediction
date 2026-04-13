# evaluation/visualizer.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

class Visualizer:
    """Class for creating visualizations"""
    
    def __init__(self):
        """Initialize visualizer"""
        # Set up style
        plt.style.use('ggplot')
        sns.set_style("whitegrid")
    
    def plot_gold_data(self, data, title="Gold Price History", figsize=(10, 6)):
        """Plot gold price data
        
        Args:
            data (np.ndarray or pd.DataFrame): Gold price data
            title (str): Plot title
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure: The figure with the plot
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot data
        if isinstance(data, pd.DataFrame):
            data.plot(ax=ax)
        else:
            ax.plot(data)
            
        # Set labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Gold Price')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_training_windows(self, data, initial_size, step_size, iterations, 
                             current_iteration=None, figsize=(12, 4)):
        """Visualize training windows
        
        Args:
            data (np.ndarray): Input data
            initial_size (int): Initial window size
            step_size (int): Step size
            iterations (int): Number of iterations
            current_iteration (int, optional): Current iteration (highlighted)
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure: The figure with the plot
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate window positions
        positions = []
        for i in range(iterations):
            start = 0
            train_end = initial_size + (i * step_size)
            val_end = train_end + step_size
            
            positions.append((start, train_end, val_end))
        
        # Plot data
        if isinstance(data, pd.DataFrame):
            ax.plot(data.values)
        else:
            ax.plot(data)
            
        # Plot windows
        for i, (start, train_end, val_end) in enumerate(positions):
            # Determine color based on current iteration
            train_color = 'blue' if i == current_iteration else 'lightblue'
            val_color = 'red' if i == current_iteration else 'lightcoral'
            
            # Plot training window
            ax.axvspan(start, train_end, alpha=0.2, color=train_color)
            
            # Plot validation window
            ax.axvspan(train_end, val_end, alpha=0.2, color=val_color)
            
            # Add labels
            if i == current_iteration:
                ax.text(train_end - step_size, ax.get_ylim()[1] * 0.9, 
                       f"Train {i+1}", ha='center', 
                       bbox=dict(facecolor='white', alpha=0.8))
                       
                ax.text(train_end + step_size/2, ax.get_ylim()[1] * 0.9, 
                       f"Val {i+1}", ha='center', 
                       bbox=dict(facecolor='white', alpha=0.8))
        
        # Set labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title('Training and Validation Windows')
        
        # Create legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightblue', alpha=0.2, label='Training Window'),
            Patch(facecolor='lightcoral', alpha=0.2, label='Validation Window')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        return fig
    
    def plot_model_comparison(self, metrics_df, metric='rmse', figsize=(10, 6)):
        """Plot model comparison based on a specific metric
        
        Args:
            metrics_df (pd.DataFrame): DataFrame with metrics
            metric (str): Metric to compare
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure: The figure with the plot
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Sort by the metric
        if metric in metrics_df.columns:
            sorted_df = metrics_df.sort_values(by=metric)
        else:
            sorted_df = metrics_df
            
        # Create bar chart
        sorted_df[metric].plot(kind='bar', ax=ax, color='skyblue')
        
        # Set labels and title
        ax.set_xlabel('Model')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'Model Comparison - {metric.upper()}')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, value in enumerate(sorted_df[metric]):
            ax.text(i, value, f'{value:.4f}', ha='center', va='bottom')
        
        return fig
    
    def plot_predictions_vs_actual(self, actual, predictions, model_names=None, 
                                  figsize=(12, 6)):
        """Plot predictions vs actual values
        
        Args:
            actual (np.ndarray): Actual values
            predictions (list or dict): Predictions
            model_names (list, optional): Model names
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure: The figure with the plot
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Convert predictions to dictionary if list
        if isinstance(predictions, list):
            if model_names is None:
                model_names = [f"Model {i+1}" for i in range(len(predictions))]
            predictions = dict(zip(model_names, predictions))
        
        # Plot actual values
        ax.plot(actual, 'k-', linewidth=2, label='Actual')
        
        # Plot predictions
        colors = plt.cm.tab10.colors
        for i, (name, pred) in enumerate(predictions.items()):
            ax.plot(pred, '--', color=colors[i % len(colors)], 
                   alpha=0.8, label=name)
        
        # Set labels and title
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.set_title('Predictions vs Actual Values')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_ensemble_weights(self, weights_history, figsize=(12, 6)):
        """Plot evolution of ensemble weights
        
        Args:
            weights_history (list): List of weight dictionaries
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure: The figure with the plot
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Convert to DataFrame
        df = pd.DataFrame(weights_history)
        
        # Plot weights evolution
        df.plot(ax=ax)
        
        # Set labels and title
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Weight')
        ax.set_title('Evolution of Ensemble Weights')
        ax.grid(True, alpha=0.3)
        ax.legend(title='Model')
        
        return fig
    
    def plot_final_prediction(self, historical_data, prediction, confidence_interval=None,
                             figsize=(12, 6)):
        """Plot final prediction with optional confidence interval
        
        Args:
            historical_data (np.ndarray): Historical data
            prediction (np.ndarray): Predicted values
            confidence_interval (tuple, optional): Lower and upper bounds
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure: The figure with the plot
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot historical data
        ax.plot(range(len(historical_data)), historical_data, 'b-', label='Historical')
        
        # Forecast horizon
        forecast_start = len(historical_data)
        forecast_end = forecast_start + len(prediction)
        forecast_index = range(forecast_start, forecast_end)
        
        # Plot prediction
        ax.plot(forecast_index, prediction, 'r-', label='Prediction')
        
        # Add the last historical point to connect the lines
        ax.plot([forecast_start-1, forecast_start], 
               [historical_data[-1], prediction[0]], 'r-')
        
        # Plot confidence interval if provided
        if confidence_interval is not None:
            lower_bound, upper_bound = confidence_interval
            ax.fill_between(forecast_index, lower_bound, upper_bound, 
                           color='red', alpha=0.2, label='Confidence Interval')
        
        # Set labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title('Final Prediction')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add vertical line separating historical and prediction
        ax.axvline(x=forecast_start-1, color='gray', linestyle='--')
        
        return fig