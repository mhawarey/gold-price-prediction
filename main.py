import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
import sys
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Add project modules
from gui.dashboard import PredictionDashboard
from data.loader import DataLoader
from data.preprocessor import DataPreprocessor
from data.window_manager import WindowManager
from models.model_manager import ModelManager
from evaluation.evaluator import Evaluator

class GoldPricePredictionApp:
    """Main application class for Gold Price Prediction System"""
    
    def __init__(self, root):
        """Initialize the application
    
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("Gold Price Prediction System - Dr. Mosab Hawarey")
        self.root.geometry("1280x800")  # Increase window size
        self.root.minsize(1200, 700)    # Minimum size
        
        # Try to set icon if available
        try:
            if os.path.exists("assets/icon.ico"):
                self.root.iconbitmap("assets/icon.ico")
        except:
            pass  # Skip if icon not available or not supported
        
        # Initialize state variables
        self.gold_data = None
        self.processed_data = None
        self.window_manager = None
        self.model_manager = None
        self.evaluator = None
        self.current_iteration = 0
        self.iteration_results = []
        self.final_prediction = None
        
        # Create main dashboard interface
        self.dashboard = PredictionDashboard(self.root, self)
        
        # Bind action callbacks
        self.dashboard.bind_load_data(self.on_load_data)
        self.dashboard.bind_run_training(self.on_run_training)
        self.dashboard.bind_make_prediction(self.on_make_prediction)
        
    def on_load_data(self):
        """Handle loading data action"""
        try:
            # Open file dialog
            file_path = filedialog.askopenfilename(
                title="Select Gold Price Data File",
                filetypes=(("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*"))
            )
        
            if not file_path:
                return  # User cancelled
            
            # Load data
            data_loader = DataLoader()
            self.gold_data = data_loader.load(file_path)
        
            # If the data has a DatetimeIndex, extract just the gold prices
            if isinstance(self.gold_data.index, pd.DatetimeIndex):
                # Keep the original data for visualization
                self.original_gold_data = self.gold_data.copy()
            
                # Extract gold prices as a numpy array
                if 'gold' in self.gold_data.columns:
                    self.processed_data = self.gold_data['gold'].values.reshape(-1, 1)
                else:
                    self.processed_data = self.gold_data.iloc[:, 0].values.reshape(-1, 1)
            else:
                # Otherwise use the preprocessor
                preprocessor = DataPreprocessor()
                self.processed_data = preprocessor.process(self.gold_data)
        
            # Update UI with loaded data
            self.dashboard.update_data_display(self.processed_data)
        
            # Enable training controls
            self.dashboard.enable_training_controls()
        
            messagebox.showinfo("Success", f"Loaded {len(self.processed_data)} data points successfully.")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
    
    def on_run_training(self):
        """Handle run training action"""
        try:
            if self.processed_data is None:
                messagebox.showwarning("Warning", "Please load data first.")
                return
                
            # Get configuration from UI
            config = self.dashboard.get_configuration()
            
            # Create window manager
            self.window_manager = WindowManager(
                data=self.processed_data,
                initial_size=config['initial_window_size'],
                step_size=config['step_size'],
                iterations=config['iterations']
            )
            
            # Create model manager with selected models
            self.model_manager = ModelManager(
                selected_models=config['selected_models'],
                lookback_window=30  # Default lookback window
            )
            
            # Create evaluator
            self.evaluator = Evaluator()
            
            # Reset iteration tracking
            self.current_iteration = 0
            self.iteration_results = []
            
            # Start first iteration
            self.run_iteration()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error during training setup: {str(e)}")
    
    def run_iteration(self):
        """Run a single training iteration"""
        try:
            if self.current_iteration >= self.window_manager.iterations:
                messagebox.showinfo("Complete", "All iterations completed!")
                self.dashboard.enable_prediction_controls()
                return
                
            # Update UI to show current iteration
            self.dashboard.update_iteration_display(
                self.current_iteration + 1, 
                self.window_manager.iterations
            )
            
            # Get training and validation data for this iteration
            train_data, validation_data = self.window_manager.get_window(self.current_iteration)
            
            # Train models
            self.model_manager.train_models(train_data)
            
            # Generate predictions
            predictions = self.model_manager.predict(train_data)
            ensemble_prediction = self.model_manager.get_ensemble_prediction(predictions)
            
            # Evaluate performance
            model_performances = []
            for model_name, prediction in predictions.items():
                performance = self.evaluator.evaluate(validation_data, prediction)
                model_performances.append((model_name, performance))
                
            ensemble_performance = self.evaluator.evaluate(validation_data, ensemble_prediction)
            
            # Store results
            result = {
                'iteration': self.current_iteration + 1,
                'train_size': len(train_data),
                'model_performances': model_performances,
                'ensemble_performance': ensemble_performance,
                'predictions': predictions,
                'ensemble_prediction': ensemble_prediction,
                'actual': validation_data
            }
            self.iteration_results.append(result)
            
            # Update UI with results
            self.dashboard.update_results_display(result)
            
            # Update ensemble weights
            performance_dict = {name: perf for name, perf in model_performances}
            self.model_manager.update_ensemble_weights(performance_dict)
            
            # Adapt models based on validation results
            self.model_manager.adapt_models(train_data, validation_data)
            
            # Increment iteration counter
            self.current_iteration += 1
            
            # Schedule next iteration (use after to avoid recursion depth issues)
            self.root.after(100, self.run_iteration)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error during iteration {self.current_iteration + 1}: {str(e)}")
    
    def on_make_prediction(self):
        """Handle make prediction action"""
        try:
            if not self.iteration_results:
                messagebox.showwarning("Warning", "Please complete training first.")
                return
                
            # Get configuration
            config = self.dashboard.get_configuration()
            
            # Generate prediction for 5 steps ahead (or as configured)
            prediction_steps = config.get('prediction_steps', 5)
            
            # Make final prediction using latest data
            self.final_prediction = self.model_manager.predict_future(
                self.processed_data, 
                steps=prediction_steps
            )
            
            # Update UI with prediction
            self.dashboard.update_prediction_display(self.final_prediction)
            
            # Save predictions to file automatically
            self._save_predictions(self.final_prediction)
            
            messagebox.showinfo("Success", f"Generated prediction for {prediction_steps} future periods.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error making prediction: {str(e)}")

    def _save_predictions(self, predictions):
        """Save predictions to CSV file
        
        Args:
            predictions (dict): Dictionary of predictions from different models
        """
        try:
            # Create export directory
            export_dir = "output"
            os.makedirs(export_dir, exist_ok=True)
            
            # Generate timestamp
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            
            # Create filename
            file_path = os.path.join(export_dir, f"gold_predictions_{timestamp}.csv")
            
            # Extract all model names
            model_names = [name for name in predictions.keys() if len(predictions[name]) > 0]
            
            # Determine the number of steps
            if 'Ensemble' in predictions and len(predictions['Ensemble']) > 0:
                num_steps = len(predictions['Ensemble'])
            else:
                num_steps = len(next(iter(predictions.values())))
            
            # Prepare data for DataFrame
            data = {'Step': [f"Step {i+1}" for i in range(num_steps)]}
            
            for model_name in model_names:
                model_values = []
                for step in range(num_steps):
                    if len(predictions[model_name]) > step:
                        value = predictions[model_name][step]
                        if isinstance(value, np.ndarray):
                            if value.size > 0:
                                value = value.item() if value.size == 1 else value[0]
                            else:
                                value = None
                    else:
                        value = None
                    model_values.append(value)
                data[model_name] = model_values
            
            # Create DataFrame and export to CSV
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)
            
            print(f"Predictions automatically saved to {file_path}")
            
        except Exception as e:
            print(f"Error saving predictions: {str(e)}")


def ensure_directories():
    """Ensure required directories exist"""
    directories = ['./data', './models', './output', './config', './assets']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def main():
    """Main entry point for the application"""
    # Ensure required directories exist
    ensure_directories()
    
    # Create Tkinter root
    root = tk.Tk()
    
    # Apply basic styling to root
    root.configure(bg="#f0f0f0")
    
    # Create application instance
    app = GoldPricePredictionApp(root)
    
    # Start the main event loop
    root.mainloop()

if __name__ == "__main__":
    main()