# models/machine_learning/lstm_model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import os

class GoldPriceLSTM:
    """LSTM model for gold price prediction"""
    
    def __init__(self, lookback=30, units=50, dropout=0.2, learning_rate=0.001):
        """Initialize LSTM model
        
        Args:
            lookback (int): Number of previous time steps to use as input
            units (int): Number of LSTM units in the model
            dropout (float): Dropout rate for regularization
            learning_rate (float): Learning rate for optimizer
        """
        self.lookback = lookback
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
        
        # Initialize model
        self.model = self._build_model()
        
        # Enable memory growth for GPU if available
        try:
            physical_devices = tf.config.list_physical_devices('GPU')
            if physical_devices:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
        except:
            # Ignore if no GPU or other issues
            pass
    
    def _build_model(self):
        """Build LSTM model
        
        Returns:
            tf.keras.Model: Compiled LSTM model
        """
        model = Sequential([
            LSTM(self.units, return_sequences=True, 
                 input_shape=(self.lookback, 1)),
            Dropout(self.dropout),
            LSTM(self.units // 2),
            Dropout(self.dropout),
            Dense(5)  # Predict 5 days ahead directly
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def _prepare_sequences(self, data, predict_ahead=5):
        """Prepare sequences for LSTM model
        
        Args:
            data (np.ndarray): Input data
            predict_ahead (int): Number of steps ahead to predict
            
        Returns:
            tuple: (X, y) sequences
        """
        if len(data) <= self.lookback + predict_ahead:
            raise ValueError(f"Not enough data for lookback and prediction. Need at least {self.lookback + predict_ahead + 1} points.")
            
        # Flatten data if needed
        if data.ndim > 1:
            data = data.flatten()
            
        X, y = [], []
        
        for i in range(len(data) - self.lookback - predict_ahead + 1):
            X.append(data[i:i + self.lookback])
            y.append(data[i + self.lookback:i + self.lookback + predict_ahead])
            
        return np.array(X).reshape(-1, self.lookback, 1), np.array(y)
    
    def fit(self, data, epochs=50, batch_size=32, validation_split=0.1):
        """Fit LSTM model to data
        
        Args:
            data (np.ndarray): Training data
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Fraction of data to use for validation
        """
        try:
            # Prepare sequences
            X, y = self._prepare_sequences(data)
            
            if len(X) < 10:
                # Not enough data after sequence preparation
                print("Not enough data for LSTM training")
                return
                
            # Early stopping to prevent overfitting
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Train model
            self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=[early_stopping],
                verbose=0
            )
        except Exception as e:
            print(f"LSTM fitting error: {str(e)}")
    
    def predict(self, data, steps=5):
        """Make predictions
        
        Args:
            data (np.ndarray): Input data
            steps (int): Number of steps to predict (ignored, always predicts 5 steps)
            
        Returns:
            np.ndarray: Predictions
        """
        try:
            # Use the last lookback points as input
            if len(data) < self.lookback:
                # Not enough data
                return np.zeros((5, 1))
                
            # Prepare input sequence
            if data.ndim > 1:
                data = data.flatten()
                
            X = data[-self.lookback:].reshape(1, self.lookback, 1)
            
            # Generate prediction
            prediction = self.model.predict(X, verbose=0)
            
            # Reshape to match expected format
            return prediction[0].reshape(-1, 1)
        except Exception as e:
            print(f"LSTM prediction error: {str(e)}")
            return np.zeros((5, 1))
    
    def adapt(self, train_data, validation_data):
        """Adapt model based on validation data
        
        Args:
            train_data (np.ndarray): Training data
            validation_data (np.ndarray): Validation data
        """
        try:
            # Combine data
            combined_data = np.vstack((train_data, validation_data))
            
            # Fine-tune with combined data (fewer epochs)
            self.fit(combined_data, epochs=10, batch_size=32)
        except Exception as e:
            print(f"LSTM adaptation error: {str(e)}")
    
    def save(self, path='models/saved/lstm_model'):
        """Save model to disk
        
        Args:
            path (str): Path to save the model
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save model
            self.model.save(path)
        except Exception as e:
            print(f"LSTM model save error: {str(e)}")
    
    def load(self, path='models/saved/lstm_model'):
        """Load model from disk
        
        Args:
            path (str): Path to load the model from
        """
        try:
            if os.path.exists(path):
                self.model = tf.keras.models.load_model(path)
                print("LSTM model loaded successfully")
        except Exception as e:
            print(f"LSTM model load error: {str(e)}")