# data/preprocessor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import stats
import matplotlib.pyplot as plt

class DataPreprocessor:
    """Class for preprocessing gold price data"""
    
    def __init__(self, price_column=None, scaler_type='minmax'):
        """Initialize the data preprocessor
        
        Args:
            price_column (str, optional): The column name for price data. If None, will attempt to find it.
            scaler_type (str, optional): Type of scaler to use ('minmax' or 'standard')
        """
        self.price_column = price_column
        
        # Initialize appropriate scaler
        if scaler_type.lower() == 'minmax':
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        elif scaler_type.lower() == 'standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unsupported scaler type: {scaler_type}. Use 'minmax' or 'standard'.")
            
        self.scaled_data = None
        self.original_data = None
    
    def process(self, data):
        """Preprocess the data for modeling"""
        print("Data type received:", type(data))
        if isinstance(data, pd.DataFrame):
            print("DataFrame columns:", data.columns.tolist())
            print("DataFrame shape:", data.shape)
        elif isinstance(data, np.ndarray):
            print("Array shape:", data.shape)

        self.original_data = data.copy()
        
        # Identify price column if not specified
        if self.price_column is None:
            self.price_column = self._identify_price_column(data)
            
        # Extract price data
        if isinstance(data, pd.DataFrame):
            if self.price_column in data.columns:
                price_data = data[self.price_column]
            else:
                # If specified column not found, try to identify it
                self.price_column = self._identify_price_column(data)
                price_data = data[self.price_column]
        else:
            # Assume data is already a series or array of prices
            if isinstance(data, np.ndarray) and data.ndim == 1:
                # Convert 1D array to DataFrame with date index
                dates = pd.date_range(start='2022-01-01', periods=len(data))
                df = pd.DataFrame({'price': data}, index=dates)
                price_data = df['price']
            else:
                price_data = data
        
        # Convert to numpy array and reshape
        price_array = np.array(price_data).reshape(-1, 1)
        
        # Remove NaN values
        price_array = price_array[~np.isnan(price_array)]
        
        # Handle outliers
        price_array = self._handle_outliers(price_array)
        
        # Scale the data
        self.scaled_data = self.scaler.fit_transform(price_array)
        
        return self.scaled_data
    
    def _identify_price_column(self, data):
        """Attempt to identify the price column in the data
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            str: Identified price column name
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame to identify price column")
            
        # Check common price column names
        price_columns = ['price', 'close', 'gold_price', 'value', 'gold', 'usd']
        for col in price_columns:
            matches = [c for c in data.columns if col.lower() in c.lower()]
            if matches:
                return matches[0]
                
        # If no match found, use the first numeric column
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            return numeric_cols[0]
            
        raise ValueError("Could not identify a suitable price column")
    
    def _handle_outliers(self, data, method='zscore', threshold=3.0):
        """Handle outliers in the data
        
        Args:
            data (np.ndarray): Input data
            method (str): Method for outlier detection ('zscore' or 'iqr')
            threshold (float): Threshold for outlier detection
            
        Returns:
            np.ndarray: Data with outliers handled
        """
        if method.lower() == 'zscore':
            # Remove outliers using Z-score
            z_scores = np.abs(stats.zscore(data))
            data = data[z_scores < threshold]
        elif method.lower() == 'iqr':
            # Remove outliers using IQR
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            lower_bound = q1 - (threshold * iqr)
            upper_bound = q3 + (threshold * iqr)
            data = data[(data >= lower_bound) & (data <= upper_bound)]
        
        return data
    
    def inverse_transform(self, scaled_data):
        """Transform scaled data back to original scale
        
        Args:
            scaled_data (np.ndarray): Scaled data
            
        Returns:
            np.ndarray: Data in original scale
        """
        if self.scaler is None:
            raise ValueError("Scaler has not been fitted. Process data first.")
            
        return self.scaler.inverse_transform(scaled_data)
    
    def plot_data(self, ax=None):
        """Plot original and preprocessed data
        
        Args:
            ax (matplotlib.axes, optional): Axes to plot on
            
        Returns:
            matplotlib.axes: The axes with the plot
        """
        if self.original_data is None or self.scaled_data is None:
            raise ValueError("No data has been processed yet")
            
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        # Plot original data
        if isinstance(self.original_data, pd.DataFrame):
            self.original_data[self.price_column].plot(ax=ax, label='Original Data')
        else:
            ax.plot(self.original_data, label='Original Data')
            
        # Plot scaled data on secondary y-axis
        ax2 = ax.twinx()
        ax2.plot(self.scaled_data, color='orange', label='Scaled Data')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Original Price')
        ax2.set_ylabel('Scaled Value')
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        return ax