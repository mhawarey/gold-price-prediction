# data/loader.py
import pandas as pd
import numpy as np
from datetime import datetime
import os

class DataLoader:
    """Class for loading gold price data from various sources"""
    
    def __init__(self):
        """Initialize the data loader"""
        self.supported_extensions = {
            '.csv': self._load_csv,
            '.xlsx': self._load_excel,
            '.xls': self._load_excel
        }
    
    def load(self, file_path):
        """Load data from the specified file path
        
        Args:
            file_path (str): Path to the data file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        _, extension = os.path.splitext(file_path.lower())
        
        if extension not in self.supported_extensions:
            raise ValueError(f"Unsupported file format: {extension}. Supported formats: {', '.join(self.supported_extensions.keys())}")
            
        # Call appropriate loader based on file extension
        data = self.supported_extensions[extension](file_path)
        
        return data
    
    def _load_csv(self, file_path):
        """Load data from CSV file
        
        Args:
            file_path (str): Path to CSV file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            # Try to read the CSV with explicit parameters
            data = pd.read_csv(file_path, encoding='utf-8', sep=',')
            
            # Print debug information
            print("CSV file loaded successfully")
            print("Columns:", data.columns.tolist())
            print("Shape:", data.shape)
            print("First few rows:")
            print(data.head())
            
            # Check if data contains a date/time column
            date_columns = [col for col in data.columns if any(
                date_term in col.lower() for date_term in ['date', 'time', 'day', 'month', 'year'])]
                
            if date_columns:
                print(f"Found date column: {date_columns[0]}")
                # Try to parse the first date column
                try:
                    data[date_columns[0]] = pd.to_datetime(data[date_columns[0]], dayfirst=True)  # Assuming DD/MM/YYYY format
                    data.set_index(date_columns[0], inplace=True)
                    print("Date column parsed and set as index")
                except Exception as e:
                    print(f"Error parsing date column: {str(e)}")
                    # If date parsing fails, keep as is
                    pass
            else:
                print("No date column found in the CSV")
                    
            return data
        except Exception as e:
            print(f"Detailed CSV loading error: {str(e)}")
            
            # Try with different parameters if the first attempt fails
            try:
                print("Attempting to load CSV with alternative parameters...")
                # Try different encoding and separator detection
                data = pd.read_csv(file_path, encoding='latin1', sep=None, engine='python')
                print("Alternative loading succeeded")
                return data
            except Exception as alt_e:
                print(f"Alternative loading also failed: {str(alt_e)}")
                raise IOError(f"Error loading CSV file: {str(e)}")
    
    def _load_excel(self, file_path):
        """Load data from Excel file
        
        Args:
            file_path (str): Path to Excel file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            # Try to read the first sheet by default
            data = pd.read_excel(file_path, sheet_name=0)
            
            # Print debug information
            print("Excel file loaded successfully")
            print("Columns:", data.columns.tolist())
            print("Shape:", data.shape)
            
            # Check if data contains a date/time column
            date_columns = [col for col in data.columns if any(
                date_term in col.lower() for date_term in ['date', 'time', 'day', 'month', 'year'])]
                
            if date_columns:
                print(f"Found date column: {date_columns[0]}")
                # Try to parse the first date column
                try:
                    data[date_columns[0]] = pd.to_datetime(data[date_columns[0]])
                    data.set_index(date_columns[0], inplace=True)
                    print("Date column parsed and set as index")
                except Exception as e:
                    print(f"Error parsing date column: {str(e)}")
                    # If date parsing fails, keep as is
                    pass
            else:
                print("No date column found in the Excel file")
                    
            return data
        except Exception as e:
            print(f"Excel loading error: {str(e)}")
            raise IOError(f"Error loading Excel file: {str(e)}")
    
    def validate_data(self, data):
        """Validate the loaded data format
        
        Args:
            data (pd.DataFrame): Loaded data
            
        Returns:
            bool: True if valid, False otherwise
        """
        # Check if data has at least one column that could be price
        price_columns = [col for col in data.columns if any(
            price_term in col.lower() for price_term in ['price', 'value', 'close', 'gold', 'usd'])]
            
        if not price_columns:
            print("No price column found in the data")
            return False
            
        # Check if there's enough data
        if len(data) < 30:  # Arbitrary minimum for analysis
            print(f"Not enough data points: {len(data)} (minimum 30 required)")
            return False
            
        print("Data validation passed")
        return True