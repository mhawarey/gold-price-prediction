# data/window_manager.py
import numpy as np

class WindowManager:
    """Class for managing rolling window creation for training and validation"""
    
    def __init__(self, data, initial_size, step_size, iterations):
        """Initialize the window manager
        
        Args:
            data (np.ndarray): Input data (preprocessed)
            initial_size (int): Initial window size for training
            step_size (int): Step size for each iteration
            iterations (int): Number of iterations to perform
        """
        self.data = data
        self.initial_size = initial_size
        self.step_size = step_size
        self.iterations = iterations
        
        # Validate parameters
        self._validate_parameters()
        
    def _validate_parameters(self):
        """Validate window parameters"""
        # Check if there's enough data for all iterations
        total_required = self.initial_size + (self.iterations * self.step_size)
        if total_required > len(self.data):
            raise ValueError(f"Not enough data for specified parameters. Need {total_required} points, have {len(self.data)}.")
            
        # Check if initial size is positive
        if self.initial_size <= 0:
            raise ValueError("Initial window size must be positive")
            
        # Check if step size is positive
        if self.step_size <= 0:
            raise ValueError("Step size must be positive")
            
        # Check if iterations is positive
        if self.iterations <= 0:
            raise ValueError("Number of iterations must be positive")
    
    def get_window(self, iteration):
        """Get training and validation windows for a specific iteration
        
        Args:
            iteration (int): Iteration number (0-based)
            
        Returns:
            tuple: (training_data, validation_data)
        """
        if iteration < 0 or iteration >= self.iterations:
            raise ValueError(f"Iteration must be between 0 and {self.iterations-1}")
            
        # Calculate window bounds
        train_end = self.initial_size + (iteration * self.step_size)
        val_end = train_end + self.step_size
        
        # Extract training and validation data
        training_data = self.data[:train_end]
        validation_data = self.data[train_end:val_end]
        
        return training_data, validation_data
    
    def get_all_windows(self):
        """Get all training and validation windows
        
        Returns:
            list: List of (training_data, validation_data) tuples for all iterations
        """
        return [self.get_window(i) for i in range(self.iterations)]
    
    def get_final_window(self):
        """Get the final window for future prediction
        
        Returns:
            np.ndarray: Final window of data
        """
        final_start = len(self.data) - self.initial_size
        if final_start < 0:
            final_start = 0
            
        return self.data[final_start:]