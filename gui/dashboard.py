# gui/dashboard.py
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import pandas as pd
import os
import json

from evaluation.visualizer import Visualizer

class PredictionDashboard:
    """Modern dashboard for the Gold Price Prediction System based on mockup"""
    
    def __init__(self, root, app_controller):
        """Initialize the dashboard
        
        Args:
            root (tk.Tk): Root Tkinter window
            app_controller: Application controller
        """
        self.root = root
        self.app_controller = app_controller
        self.visualizer = Visualizer()
        
        # Configure colors
        self.colors = {
            'header_bg': '#2c3e50',  # Dark blue header background
            'header_fg': 'white',    # White header text
            'primary': '#3498db',    # Primary blue (Load Data, Ensemble)
            'secondary': '#2ecc71',  # Green (Run Training)
            'warning': '#f39c12',    # Orange (Make Prediction)
            'danger': '#e74c3c',     # Red (for warnings/errors)
            'tab_active': '#3498db', # Active tab color
            'tab_inactive': '#ecf0f1', # Inactive tab color
            'section_bg': '#f8f9fa', # Light background for sections
            'border': '#dee2e6',     # Border color
            'footer_bg': '#2c3e50',  # Dark blue footer background
            'footer_fg': 'white',    # White footer text
        }
        
        # Configure fonts
        self.fonts = {
            'header': ('Helvetica', 16, 'bold'),
            'subheader': ('Helvetica', 12, 'bold'),
            'normal': ('Helvetica', 10),
            'small': ('Helvetica', 9)
        }
        
        # Configure styles
        self._configure_styles()
        
        # Create main layout
        self._create_main_layout()
        
        # Initialize variables
        self.figure_cache = {}  # Cache for matplotlib figures
        self.data_loaded = False
        self.training_completed = False
        self.prediction_made = False
        
        # Set initial state of controls
        self.disable_training_controls()
        self.disable_prediction_controls()
    
    def _configure_styles(self):
        """Configure custom styles for widgets"""
        style = ttk.Style()
        
        # Try to use a modern theme
        try:
            style.theme_use('clam')
        except:
            pass  # Use default theme if clam is not available
            
        # Configure main styles
        style.configure('Header.TLabel', 
                      background=self.colors['header_bg'],
                      foreground=self.colors['header_fg'],
                      font=self.fonts['header'],
                      padding=10)
        
        style.configure('Footer.TLabel', 
                      background=self.colors['footer_bg'],
                      foreground=self.colors['footer_fg'],
                      font=self.fonts['small'],
                      padding=5)
        
        # Tab styles
        style.configure('Tab.TNotebook', background=self.colors['section_bg'])
        style.configure('Tab.TNotebook.Tab', 
                      font=self.fonts['normal'],
                      padding=[12, 6],
                      background=self.colors['tab_inactive'])
        style.map('Tab.TNotebook.Tab',
                background=[('selected', self.colors['tab_active']), 
                           ('active', self.colors['primary'])],
                foreground=[('selected', 'white'), ('active', 'white')])
        
        # Section styles
        style.configure('Section.TLabelframe', 
                      background=self.colors['section_bg'],
                      font=self.fonts['subheader'])
        style.configure('Section.TLabelframe.Label', 
                      font=self.fonts['subheader'],
                      background=self.colors['section_bg'])
        
        # Button styles
        style.configure('Primary.TButton', 
                      font=self.fonts['normal'],
                      background=self.colors['primary'])
        style.map('Primary.TButton',
                background=[('active', '#2980b9')])
        
        style.configure('Secondary.TButton', 
                      font=self.fonts['normal'],
                      background=self.colors['secondary'])
        style.map('Secondary.TButton',
                background=[('active', '#27ae60')])
        
        style.configure('Warning.TButton', 
                      font=self.fonts['normal'],
                      background=self.colors['warning'])
        style.map('Warning.TButton',
                background=[('active', '#d35400')])
    
    def _create_main_layout(self):
        """Create the main application layout"""
        # Create header frame
        self.header_frame = ttk.Frame(self.root, style='Header.TFrame')
        self.header_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Header title
        ttk.Label(
            self.header_frame, 
            text="Gold Price Prediction System", 
            style='Header.TLabel'
        ).pack(side=tk.LEFT, padx=10)
        
        # Create content frame
        self.content_frame = ttk.Frame(self.root)
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create left sidebar for configuration
        self.sidebar_frame = ttk.Frame(self.content_frame, width=300)
        self.sidebar_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self.sidebar_frame.pack_propagate(False)  # Don't shrink
        
        # Create right content area
        self.main_content = ttk.Frame(self.content_frame)
        self.main_content.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create footer frame
        self.footer_frame = ttk.Frame(self.root, style='Footer.TFrame')
        self.footer_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Footer status text
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(
            self.footer_frame, 
            textvariable=self.status_var, 
            style='Footer.TLabel',
            anchor=tk.W
        )
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # Footer metrics text
        self.metrics_var = tk.StringVar(value="")
        self.metrics_label = ttk.Label(
            self.footer_frame, 
            textvariable=self.metrics_var, 
            style='Footer.TLabel',
            anchor=tk.E
        )
        self.metrics_label.pack(side=tk.RIGHT, padx=10)
        
        # Initialize sidebar components
        self._create_sidebar()
        
        # Initialize main content area
        self._create_main_content()
    
    def _create_sidebar(self):
        """Create sidebar with configuration options"""
        # Main configuration label
        config_label = ttk.Label(
            self.sidebar_frame,
            text="Configuration",
            font=self.fonts['subheader'],
            padding=(5, 10)
        )
        config_label.pack(fill=tk.X)
        
        # 1. Dataset Settings Section
        self.dataset_frame = ttk.LabelFrame(
            self.sidebar_frame, 
            text="Dataset Settings",
            style='Section.TLabelframe'
        )
        self.dataset_frame.pack(fill=tk.X, pady=5, padx=5)
        
        # Total Points
        ttk.Label(self.dataset_frame, text="Total Data Points:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.total_points_var = tk.StringVar(value="N/A")
        ttk.Label(self.dataset_frame, textvariable=self.total_points_var).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Initial Window
        ttk.Label(self.dataset_frame, text="Initial Window:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.initial_window_var = tk.IntVar(value=450)
        initial_window_entry = ttk.Entry(self.dataset_frame, textvariable=self.initial_window_var, width=10)
        initial_window_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Step Size
        ttk.Label(self.dataset_frame, text="Step Size:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.step_size_var = tk.IntVar(value=5)
        step_size_entry = ttk.Entry(self.dataset_frame, textvariable=self.step_size_var, width=10)
        step_size_entry.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Iterations
        ttk.Label(self.dataset_frame, text="Iterations:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.iterations_var = tk.IntVar(value=10)
        iterations_entry = ttk.Entry(self.dataset_frame, textvariable=self.iterations_var, width=10)
        iterations_entry.grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 2. Model Selection Section
        self.model_frame = ttk.LabelFrame(
            self.sidebar_frame, 
            text="Model Selection",
            style='Section.TLabelframe'
        )
        self.model_frame.pack(fill=tk.X, pady=5, padx=5)
        
        # Model selection checkboxes
        self.model_vars = {}
        models = [
            ('ARIMA', 'ARIMA'), 
            ('GARCH', 'GARCH'),
            ('ExpSmoothing', 'Exponential Smoothing'),
            ('LSTM', 'LSTM'),
            ('XGBoost', 'XGBoost'),
            ('SVR', 'SVR'),
            ('RandomForest', 'Random Forest')
            # Remove MetaLearner from the UI for now
            # ('MetaLearner', 'Meta-Learner')
        ]
        
        for i, (model_key, model_name) in enumerate(models):
            self.model_vars[model_key] = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(
                self.model_frame, 
                text=model_name, 
                variable=self.model_vars[model_key]
            )
            cb.grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
        
        # 3. Controls Section
        self.controls_frame = ttk.LabelFrame(
            self.sidebar_frame, 
            text="Controls",
            style='Section.TLabelframe'
        )
        self.controls_frame.pack(fill=tk.X, pady=5, padx=5)
        
        # Load Data button
        self.load_data_btn = ttk.Button(
            self.controls_frame, 
            text="Load Data",
            style='Primary.TButton',
            command=self._on_load_data
        )
        self.load_data_btn.pack(fill=tk.X, pady=5, padx=5)
        
        # Run Training button
        self.run_training_btn = ttk.Button(
            self.controls_frame, 
            text="Run Training",
            style='Secondary.TButton',
            command=self._on_run_training
        )
        self.run_training_btn.pack(fill=tk.X, pady=5, padx=5)
        
        # Make Prediction button
        self.make_prediction_btn = ttk.Button(
            self.controls_frame, 
            text="Make Prediction",
            style='Warning.TButton',
            command=self._on_make_prediction
        )
        self.make_prediction_btn.pack(fill=tk.X, pady=5, padx=5)
    
    def _create_main_content(self):
        """Create main content area with tabs"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.main_content, style='Tab.TNotebook')
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Data View tab
        self.data_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.data_tab, text='Data View')
        
        # Performance tab
        self.performance_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.performance_tab, text='Performance')
        
        # Predictions tab
        self.prediction_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.prediction_tab, text='Predictions')
        
        # Initialize tab contents
        self._init_data_tab()
        self._init_performance_tab()
        self._init_prediction_tab()
        
        # Add training windows section below tabs
        self._create_training_windows_section()
        
        # Add iteration status section
        self._create_iteration_status_section()
    
    def _init_data_tab(self):
        """Initialize data view tab"""
        # Create figure
        self.data_fig = plt.Figure(figsize=(8, 4))
        self.data_canvas = FigureCanvasTkAgg(self.data_fig, self.data_tab)
        self.data_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar_frame = ttk.Frame(self.data_tab)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.data_toolbar = NavigationToolbar2Tk(self.data_canvas, toolbar_frame)
        self.data_toolbar.update()
        
        # Add placeholder text
        self.data_ax = self.data_fig.add_subplot(111)
        self.data_ax.text(0.5, 0.5, "No data loaded. Use 'Load Data' button to load gold price data.",
                         ha='center', va='center', fontsize=12)
        self.data_ax.set_axis_off()
        self.data_canvas.draw()
    
    def _init_performance_tab(self):
        """Initialize performance tab"""
        # Create figure
        self.performance_fig = plt.Figure(figsize=(8, 4))
        self.performance_canvas = FigureCanvasTkAgg(self.performance_fig, self.performance_tab)
        self.performance_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar_frame = ttk.Frame(self.performance_tab)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.performance_toolbar = NavigationToolbar2Tk(self.performance_canvas, toolbar_frame)
        self.performance_toolbar.update()
        
        # Add placeholder text
        self.performance_ax = self.performance_fig.add_subplot(111)
        self.performance_ax.text(0.5, 0.5, "No performance data available. Run training to see results.",
                               ha='center', va='center', fontsize=12)
        self.performance_ax.set_axis_off()
        self.performance_canvas.draw()
    
    def _init_prediction_tab(self):
        """Initialize prediction tab"""
        # Create figure
        self.prediction_fig = plt.Figure(figsize=(8, 4))
        self.prediction_canvas = FigureCanvasTkAgg(self.prediction_fig, self.prediction_tab)
        self.prediction_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar_frame = ttk.Frame(self.prediction_tab)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.prediction_toolbar = NavigationToolbar2Tk(self.prediction_canvas, toolbar_frame)
        self.prediction_toolbar.update()
        
        # Add placeholder text
        self.prediction_ax = self.prediction_fig.add_subplot(111)
        self.prediction_ax.text(0.5, 0.5, "No prediction available. Run training and make predictions to see results.",
                              ha='center', va='center', fontsize=12)
        self.prediction_ax.set_axis_off()
        self.prediction_canvas.draw()
    
    def _create_training_windows_section(self):
        """Create training windows visualization section"""
        # Create frame for training windows
        self.windows_frame = ttk.LabelFrame(
            self.main_content, 
            text="Training Windows",
            style='Section.TLabelframe'
        )
        self.windows_frame.pack(fill=tk.X, pady=10, expand=False)
        
        # Create canvas for visualization
        self.windows_canvas = tk.Canvas(self.windows_frame, height=60, bg='white')
        self.windows_canvas.pack(fill=tk.X, padx=5, pady=5)
        
        # Draw placeholder
        self.windows_canvas.create_rectangle(10, 10, 50, 50, fill=self.colors['primary'], outline='')
        self.windows_canvas.create_text(30, 30, text="Initial Training", anchor='center', font=self.fonts['small'])
        
        # Initial text
        self.windows_text = ttk.Label(
            self.windows_frame, 
            text="No training data available. Start training to see windows.",
            font=self.fonts['small']
        )
        self.windows_text.pack(pady=5)
    
    def _create_iteration_status_section(self):
        """Create section for displaying current iteration status"""
        # Create frame for iteration status
        self.iteration_frame = ttk.LabelFrame(
            self.main_content, 
            text="Current Iteration: 0 of 0",
            style='Section.TLabelframe'
        )
        self.iteration_frame.pack(fill=tk.X, pady=10, expand=False)
        
        # Create progress bars for models
        self.model_progress = {}
        
        # Frame for progress bars
        self.progress_frame = ttk.Frame(self.iteration_frame)
        self.progress_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Placeholder text
        self.iteration_text = ttk.Label(
            self.progress_frame, 
            text="No training in progress. Start training to see progress.",
            font=self.fonts['normal']
        )
        self.iteration_text.pack(pady=10)
    
    def _update_training_windows_visual(self, current_iteration, total_iterations, initial_size, step_size):
        """Update the training windows visualization
        
        Args:
            current_iteration (int): Current iteration number
            total_iterations (int): Total iterations
            initial_size (int): Initial window size
            step_size (int): Step size
        """
        # Clear canvas
        self.windows_canvas.delete("all")
        
        # Calculate dimensions
        width = self.windows_canvas.winfo_width() - 20  # Padding
        height = 40
        y_center = 30
        
        # Draw background line representing all data
        self.windows_canvas.create_line(
            10, y_center, width + 10, y_center,
            width=2, fill='gray'
        )
        
        # Calculate positions
        train_width = width * 0.8  # Training takes 80% of width
        
        # Draw training window (blue)
        train_x1 = 10
        train_x2 = train_x1 + train_width
        
        self.windows_canvas.create_rectangle(
            train_x1, y_center - 15,
            train_x2, y_center + 15,
            fill=self.colors['primary'], outline='',
            tags='train'
        )
        
        # Draw validation window (red)
        val_x1 = train_x2
        val_x2 = val_x1 + (width * 0.2)  # Validation takes 20% of width
        
        self.windows_canvas.create_rectangle(
            val_x1, y_center - 15,
            val_x2, y_center + 15,
            fill=self.colors['danger'], outline='',
            tags='validation'
        )
        
        # Add text
        self.windows_canvas.create_text(
            train_x1 + (train_width / 2), y_center,
            text=f"Initial Training ({initial_size} points)",
            fill='white', font=self.fonts['small']
        )
        
        self.windows_canvas.create_text(
            val_x1 + ((val_x2 - val_x1) / 2), y_center,
            text="Validation Forecast",
            fill='white', font=self.fonts['small']
        )
        
        # Update text label
        self.windows_text.config(
            text=f"Window {current_iteration} of {total_iterations}: Training on {initial_size + (current_iteration - 1) * step_size} points"
        )
    
    def _update_iteration_status(self, current_iteration, total_iterations, model_performances=None):
        """Update the iteration status display
        
        Args:
            current_iteration (int): Current iteration number
            total_iterations (int): Total iterations
            model_performances (dict, optional): Dictionary of model performances
        """
        # Update frame title
        self.iteration_frame.config(text=f"Current Iteration: {current_iteration} of {total_iterations}")
        
        # Clear progress frame
        for widget in self.progress_frame.winfo_children():
            widget.destroy()
        
        # If we have model performances, display them
        if model_performances:
            # Create progress bars for each model
            for i, (model_name, performance) in enumerate(model_performances):
                # Skip MetaLearner which is not in our UI
                if model_name == 'MetaLearner':
                    continue
                    
                # Create label and progress frame for this model
                model_frame = ttk.Frame(self.progress_frame)
                model_frame.pack(fill=tk.X, pady=2)
                
                # Label
                ttk.Label(
                    model_frame, 
                    text=model_name,
                    width=15, 
                    anchor='w'
                ).pack(side=tk.LEFT, padx=(0, 5))
                
                # Progress bar
                if isinstance(performance, dict) and 'rmse' in performance:
                    error = performance['rmse']
                    # Normalize error for progress bar (better models have lower RMSE)
                    # Using arbitrary scale - might need adjustment
                    normalized = max(0, min(100, 100 - (error * 20)))
                    
                    # Determine color based on error
                    if normalized > 75:
                        color = self.colors['secondary']  # Green for good performance
                    elif normalized > 50:
                        color = self.colors['primary']    # Blue for medium performance
                    elif normalized > 25:
                        color = self.colors['warning']   # Orange for poor performance
                    else:
                        color = self.colors['danger']    # Red for very poor performance
                    
                    # Create custom progress bar
                    progress_canvas = tk.Canvas(model_frame, height=20, width=300, bg='#f0f0f0', highlightthickness=0)
                    progress_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True)
                    
                    # Draw progress bar
                    bar_width = (normalized / 100) * 300
                    progress_canvas.create_rectangle(0, 0, bar_width, 20, fill=color, outline='')
                    
                    # Add text showing RMSE
                    ttk.Label(
                        model_frame,
                        text=f"RMSE: {error:.4f}",
                        width=15
                    ).pack(side=tk.RIGHT)
                    
            # Add ensemble model if available
            if any(name == 'Ensemble' for name, _ in model_performances):
                # Create label and progress frame for ensemble
                model_frame = ttk.Frame(self.progress_frame)
                model_frame.pack(fill=tk.X, pady=2)
                
                # Label
                ttk.Label(
                    model_frame, 
                    text="Ensemble",
                    width=15, 
                    anchor='w',
                    font=self.fonts['normal']
                ).pack(side=tk.LEFT, padx=(0, 5))
                
                # Progress bar for ensemble
                ensemble_perf = next((perf for name, perf in model_performances if name == 'Ensemble'), None)
                if ensemble_perf and 'rmse' in ensemble_perf:
                    error = ensemble_perf['rmse']
                    normalized = max(0, min(100, 100 - (error * 20)))
                    
                    # Create custom progress bar
                    progress_canvas = tk.Canvas(model_frame, height=20, width=300, bg='#f0f0f0', highlightthickness=0)
                    progress_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True)
                    
                    # Draw progress bar - always blue for ensemble
                    bar_width = (normalized / 100) * 300
                    progress_canvas.create_rectangle(0, 0, bar_width, 20, fill=self.colors['primary'], outline='')
                    
                    # Add text showing RMSE
                    ttk.Label(
                        model_frame,
                        text=f"RMSE: {error:.4f}",
                        width=15
                    ).pack(side=tk.RIGHT)
        else:
            # Show placeholder
            self.iteration_text = ttk.Label(
                self.progress_frame, 
                text="No training in progress. Start training to see progress.",
                font=self.fonts['normal']
            )
            self.iteration_text.pack(pady=10)
    
    def update_status(self, message):
        """Update status bar message"""
        self.status_var.set(message)
        self.root.update_idletasks()
    
    def update_metrics(self, metrics):
        """Update metrics display in status bar"""
        if not metrics:
            self.metrics_var.set("")
            return
            
        # Format metrics string
        metrics_str = "  |  ".join([f"{k.upper()}: {v:.4f}" for k, v in metrics.items() 
                                  if isinstance(v, (int, float))])
        self.metrics_var.set(metrics_str)
    
    def bind_load_data(self, callback):
        """Bind load data callback"""
        self._load_data_callback = callback
    
    def bind_run_training(self, callback):
        """Bind run training callback"""
        self._run_training_callback = callback
    
    def bind_make_prediction(self, callback):
        """Bind make prediction callback"""
        self._make_prediction_callback = callback
    
    def _on_load_data(self):
        """Handle load data button click"""
        if hasattr(self, '_load_data_callback'):
            self.update_status("Loading data...")
            self._load_data_callback()
    
    def _on_run_training(self):
        """Handle run training button click"""
        if hasattr(self, '_run_training_callback'):
            self.update_status("Starting training...")
            self._run_training_callback()
    
    def _on_make_prediction(self):
        """Handle make prediction button click"""
        if hasattr(self, '_make_prediction_callback'):
            self.update_status("Making prediction...")
            self._make_prediction_callback()
    
    def get_configuration(self):
        """Get current configuration"""
        # Selected models
        selected_models = [model for model, var in self.model_vars.items() if var.get()]
        
        # Create configuration dictionary
        config = {
            'initial_window_size': self.initial_window_var.get(),
            'step_size': self.step_size_var.get(),
            'iterations': self.iterations_var.get(),
            'selected_models': selected_models,
            'prediction_steps': 5  # Default to 5 days ahead
        }
        
        return config
    
    def update_data_display(self, data):
        """Update data display with loaded data"""
        # Update data points counter
        self.total_points_var.set(str(len(data)))
        
        # Update data flag
        self.data_loaded = True
        
        # Clear figure
        self.data_fig.clear()
        
        # Create new plot
        ax = self.data_fig.add_subplot(111)
        
        # Plot data
        if isinstance(data, np.ndarray):
            ax.plot(data, color='orange')
        else:
            ax.plot(data.values, color='orange')
        
        # Set labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Gold Price')
        ax.set_title('Gold Price History')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Update canvas
        self.data_fig.tight_layout()
        self.data_canvas.draw()
        
        # Switch to data tab
        self.notebook.select(0)  # Select Data View tab
        
        # Update status
        self.update_status(f"Loaded {len(data)} data points successfully")
    
    def update_iteration_display(self, current_iteration, total_iterations):
        """Update display for current iteration"""
        # Get configuration
        config = self.get_configuration()
        
        # Update training windows visualization
        self._update_training_windows_visual(
            current_iteration, 
            total_iterations,
            config['initial_window_size'],
            config['step_size']
        )
        
        # Update iteration status (initially without model performances)
        self._update_iteration_status(current_iteration, total_iterations)
        
        # Update status
        self.update_status(f"Training iteration {current_iteration} of {total_iterations}")
    
    def update_results_display(self, results):
        """Update display with iteration results"""
        if not results:
            return
            
        # Extract model performances
        if 'model_performances' in results:
            # Update iteration status with model performances
            self._update_iteration_status(
                results['iteration'], 
                self.get_configuration()['iterations'],
                results['model_performances']
)
            
            # Update performance figure
            self.performance_fig.clear()
            
            model_performances = dict(results['model_performances'])
            
            # Extract RMSE values
            rmse_values = {}
            for model, perf in model_performances.items():
                # Skip MetaLearner in visualization
                if model == 'MetaLearner':
                    continue
                if isinstance(perf, dict) and 'rmse' in perf:
                    rmse_values[model] = perf['rmse']
                    
            # Update metrics display in status bar
            if 'ensemble_performance' in results:
                self.update_metrics(results['ensemble_performance'])
                
            # Create bar chart of model performance
            ax = self.performance_fig.add_subplot(111)
            
            if not rmse_values:
                ax.text(0.5, 0.5, "No performance metrics available.",
                       ha='center', va='center', fontsize=12)
                ax.set_axis_off()
            else:
                # Sort by RMSE
                sorted_models = sorted(rmse_values.items(), key=lambda x: x[1])
                models = [m[0] for m in sorted_models]
                values = [m[1] for m in sorted_models]
                
                # Set colors for bars
                colors = []
                for model in models:
                    if model in ['ARIMA', 'GARCH', 'ExpSmoothing']:
                        colors.append(self.colors['primary'])  # Blue for statistical models
                    elif model in ['LSTM', 'XGBoost', 'SVR', 'RandomForest']:
                        colors.append(self.colors['secondary'])  # Green for ML models
                    else:
                        colors.append(self.colors['warning'])  # Orange for others
                
                # Create bar chart
                bars = ax.bar(models, values, color=colors)
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.4f}', ha='center', va='bottom', fontsize=9)
                    
                # Add ensemble performance
                if 'ensemble_performance' in results and 'rmse' in results['ensemble_performance']:
                    ensemble_rmse = results['ensemble_performance']['rmse']
                    ax.axhline(y=ensemble_rmse, color=self.colors['primary'], linestyle='--', 
                              label=f'Ensemble: {ensemble_rmse:.4f}')
                    ax.legend()
                    
                # Set labels and title
                ax.set_xlabel('Model')
                ax.set_ylabel('RMSE')
                ax.set_title(f'Model Performance - Iteration {results["iteration"]}')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Rotate x-axis labels for better visibility
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                
            # Update canvas
            self.performance_fig.tight_layout()
            self.performance_canvas.draw()
            
            # Switch to performance tab
            self.notebook.select(1)  # Select Performance tab
            
            # Cache this figure
            self.figure_cache['performance'] = self.performance_fig
            
        # Update training flag
        self.training_completed = True
    
    def update_prediction_display(self, predictions):
        """Update display with predictions"""
        if not predictions:
            return
            
        # Update prediction figure
        self.prediction_fig.clear()
        
        # Extract ensemble prediction
        if 'Ensemble' in predictions:
            ensemble_pred = predictions['Ensemble']
            
            # Create prediction plot
            if hasattr(self.app_controller, 'processed_data'):
                # Get the last 50 points of historical data
                historical_data = self.app_controller.processed_data[-50:]
                
                # Create figure with prediction
                ax = self.prediction_fig.add_subplot(111)
                
                # Plot historical data
                ax.plot(range(len(historical_data)), historical_data, color='orange', label='Historical')
                
                # Plot prediction
                forecast_start = len(historical_data)
                forecast_end = forecast_start + len(ensemble_pred)
                forecast_index = range(forecast_start, forecast_end)
                
                ax.plot(forecast_index, ensemble_pred, color=self.colors['primary'], label='Prediction')
                
                # Connect the lines
                ax.plot([forecast_start-1, forecast_start], 
                       [historical_data[-1], ensemble_pred[0]], color=self.colors['primary'])
                
                # Add vertical line
                ax.axvline(x=forecast_start-1, color='gray', linestyle='--')
                
                # Set labels and title
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Value')
                ax.set_title('Gold Price Prediction')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Update canvas
                self.prediction_fig.tight_layout()
                self.prediction_canvas.draw()
                
                # Cache this figure
                self.figure_cache['prediction'] = self.prediction_fig
                
                # Switch to prediction tab
                self.notebook.select(2)  # Select Predictions tab
                
                # Update prediction flag
                self.prediction_made = True
                
                # Update status
                self.update_status("Prediction complete")
                
                # Show numerical prediction values in a table below the chart
                self._display_prediction_values(predictions)
    
    def _display_prediction_values(self, predictions):
        """Display numerical prediction values in a table
        
        Args:
            predictions (dict): Dictionary of predictions from different models
        """
        # Create frame for prediction values if it doesn't exist
        if not hasattr(self, 'prediction_values_frame'):
            self.prediction_values_frame = ttk.LabelFrame(
                self.prediction_tab, 
                text="Predicted Values",
                style='Section.TLabelframe'
            )
            self.prediction_values_frame.pack(fill=tk.X, pady=10, expand=False, side=tk.BOTTOM)
        else:
            # Clear existing content
            for widget in self.prediction_values_frame.winfo_children():
                widget.destroy()
        
        # Create scrollable frame for the table
        table_frame = ttk.Frame(self.prediction_values_frame)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create canvas with scrollbar for table
        canvas = tk.Canvas(table_frame, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Extract all model names
        model_names = [name for name in predictions.keys() if len(predictions[name]) > 0]
        
        # Determine the number of steps
        if 'Ensemble' in predictions and len(predictions['Ensemble']) > 0:
            num_steps = len(predictions['Ensemble'])
        else:
            num_steps = len(next(iter(predictions.values())))
        
        # Create headers
        ttk.Label(scrollable_frame, text="Step", font=self.fonts['subheader'], width=8).grid(row=0, column=0, padx=5, pady=5, sticky='w')
        
        for i, model_name in enumerate(model_names):
            ttk.Label(scrollable_frame, text=model_name, font=self.fonts['subheader'], width=12).grid(row=0, column=i+1, padx=5, pady=5)
        
        # Add export button
        ttk.Button(
            scrollable_frame, 
            text="Export to CSV",
            style='Primary.TButton',
            command=lambda: self._export_predictions(predictions)
        ).grid(row=0, column=len(model_names)+1, padx=5, pady=5)
        
        # Fill the table with prediction values
        for step in range(num_steps):
            ttk.Label(scrollable_frame, text=f"Step {step+1}").grid(row=step+1, column=0, padx=5, pady=2, sticky='w')
            
            for i, model_name in enumerate(model_names):
                # Get value, handling different shapes
                if len(predictions[model_name]) > step:
                    value = predictions[model_name][step]
                    if isinstance(value, np.ndarray):
                        if value.size > 0:
                            value = value.item() if value.size == 1 else value[0]
                        else:
                            value = "N/A"
                else:
                    value = "N/A"
                    
                # Format value as string with 4 decimal places if it's a number
                if isinstance(value, (int, float)):
                    value_str = f"{value:.4f}"
                else:
                    value_str = str(value)
                    
                ttk.Label(scrollable_frame, text=value_str).grid(row=step+1, column=i+1, padx=5, pady=2)
    
    def _export_predictions(self, predictions):
        """Export prediction values to CSV
        
        Args:
            predictions (dict): Dictionary of predictions from different models
        """
        try:
            # Create export directory
            export_dir = "output"
            os.makedirs(export_dir, exist_ok=True)
            
            # Ask user for filename
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialdir=export_dir,
                initialfile="gold_predictions.csv",
                title="Save Predictions As"
            )
            
            if not file_path:
                return  # User cancelled
            
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
            
            messagebox.showinfo("Export Complete", f"Predictions exported to {file_path}")
        
        except Exception as e:
            messagebox.showerror("Export Error", f"Error exporting predictions: {str(e)}")
    
    def enable_training_controls(self):
        """Enable training controls"""
        self.run_training_btn.config(state=tk.NORMAL)
    
    def disable_training_controls(self):
        """Disable training controls"""
        self.run_training_btn.config(state=tk.DISABLED)
    
    def enable_prediction_controls(self):
        """Enable prediction controls"""
        self.make_prediction_btn.config(state=tk.NORMAL)
    
    def disable_prediction_controls(self):
        """Disable prediction controls"""
        self.make_prediction_btn.config(state=tk.DISABLED)