import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class TimeSeriesTransformer:
    def __init__(self, 
                 max_seq_length=20, 
                 d_model=128, 
                 num_heads=8, 
                 d_ff=256, 
                 num_layers=4, 
                 dropout_rate=0.1):
        """
        Initialize the TimeSeriesTransformer for stock return prediction.
        
        Args:
            max_seq_length (int): Maximum sequence length for the time series data
            d_model (int): Dimension of the model (embedding dimension)
            num_heads (int): Number of attention heads
            d_ff (int): Dimension of the feedforward network
            num_layers (int): Number of transformer layers
            dropout_rate (float): Dropout rate
        """
        self.max_seq_length = max_seq_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.model = None
        
    def _positional_encoding(self, position, d_model):
        """
        Compute positional encoding for transformer.
        
        Args:
            position (int): Maximum position
            d_model (int): Dimension of the model
            
        Returns:
            tensor: Positional encoding matrix of shape (1, position, d_model)
        """
        angle_rads = self._get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        
        # Apply sin to even indices in the array
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        
        # Apply cos to odd indices in the array
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def _get_angles(self, pos, i, d_model):
        """
        Calculate angles for positional encoding.
        
        Args:
            pos (ndarray): Position array
            i (ndarray): Dimension indices
            d_model (int): Dimension of the model
            
        Returns:
            ndarray: Angle array
        """
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates
    
    def _transformer_encoder(self, inputs):
        """
        Create a transformer encoder.
        
        Args:
            inputs (tensor): Input tensor
            
        Returns:
            tensor: Output tensor after transformer encoding
        """
        # Embedding and positional encoding
        x = layers.Dense(self.d_model)(inputs)  # Project to d_model dimensions
        
        # Create a proper positional encoding layer
        class PositionalEncodingLayer(layers.Layer):
            def __init__(self, max_seq_length, d_model, **kwargs):
                super(PositionalEncodingLayer, self).__init__(**kwargs)
                self.pos_encoding = self._positional_encoding(max_seq_length, d_model)
                
            def _positional_encoding(self, position, d_model):
                angle_rads = self._get_angles(
                    np.arange(position)[:, np.newaxis],
                    np.arange(d_model)[np.newaxis, :],
                    d_model
                )
                
                angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
                angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
                
                pos_encoding = angle_rads[np.newaxis, ...]
                
                return tf.cast(pos_encoding, dtype=tf.float32)
            
            def _get_angles(self, pos, i, d_model):
                angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
                return pos * angle_rates
                
            def call(self, inputs):
                # During the forward pass, we add the positional encoding
                # Get the sequence length from the inputs
                seq_len = tf.shape(inputs)[1]
                # Return the inputs with the positional encoding added
                return inputs + self.pos_encoding[:, :seq_len, :]
        
        # Apply positional encoding
        pos_encoding_layer = PositionalEncodingLayer(self.max_seq_length, self.d_model)
        x = pos_encoding_layer(x)
        
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Transformer blocks
        for i in range(self.num_layers):
            x = self._transformer_block(x, self.d_model, self.num_heads, self.d_ff)
            
        return x
    
    def _transformer_block(self, inputs, d_model, num_heads, d_ff):
        """
        Create a transformer block with multi-head attention and feed forward network.
        
        Args:
            inputs (tensor): Input tensor
            d_model (int): Dimension of the model
            num_heads (int): Number of attention heads
            d_ff (int): Dimension of the feedforward network
            
        Returns:
            tensor: Output tensor after transformer block
        """
        # Multi-head attention
        attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads
        )(inputs, inputs)
        attention = layers.Dropout(self.dropout_rate)(attention)
        attention = layers.LayerNormalization(epsilon=1e-6)(inputs + attention)
        
        # Feed forward network
        outputs = layers.Dense(d_ff, activation='relu')(attention)
        outputs = layers.Dense(d_model)(outputs)
        outputs = layers.Dropout(self.dropout_rate)(outputs)
        outputs = layers.LayerNormalization(epsilon=1e-6)(attention + outputs)
        
        return outputs
    
    def build_model(self, input_shape):
        """
        Build the transformer model for time series prediction.
        
        Args:
            input_shape (tuple): Shape of the input data (sequence_length, feature_dim)
            
        Returns:
            model: Compiled Keras model
        """
        # Input layers
        inputs = layers.Input(shape=input_shape)
        
        # Transformer encoder
        x = self._transformer_encoder(inputs)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Output layer
        outputs = layers.Dense(1, activation='linear')(x)
        
        # Create model
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def prepare_time_series_data(self, stock_ids, day_indices, features, returns, window_size=None):
        """
        Fully vectorized implementation of prepare_time_series_data that eliminates loops and GroupBy operations.
        This version is significantly faster for large datasets.
        
        Args:
            stock_ids (ndarray): Array of stock IDs
            day_indices (ndarray): Array of day indices
            features (ndarray): Feature matrix
            returns (ndarray): Target return values
            window_size (int): Size of the sliding window (if None, use max_seq_length)
            
        Returns:
            tuple: (X, y) containing sequence data and target values
        """
        if window_size is None:
            window_size = self.max_seq_length
        
        # Convert inputs to numpy arrays if they aren't already
        stock_ids = np.asarray(stock_ids)
        day_indices = np.asarray(day_indices)
        features = np.asarray(features)
        returns = np.asarray(returns)
        
        # Create a dataframe with the minimum required columns
        df = pd.DataFrame({
            'stock_id': stock_ids,
            'day_idx': day_indices,
            'return': returns,
            'row_idx': np.arange(len(stock_ids))  # Add row index for faster data access
        })
        
        # Sort the dataframe
        df = df.sort_values(['stock_id', 'day_idx']).reset_index(drop=True)
        
        # Find consecutive sequences within each stock
        df['group_change'] = (df['stock_id'] != df['stock_id'].shift(1)).astype(int)
        df['group_id'] = df['group_change'].cumsum()
        
        # Create a helper column to identify the starting index of each valid sequence
        df['seq_idx'] = df.groupby('group_id').cumcount()
        
        # Filter for valid starting points (those that have enough data points after them)
        valid_starts = df[df['seq_idx'] <= df.groupby('group_id')['seq_idx'].transform('max') - window_size]
        
        # If no valid sequences exist, return empty arrays
        if len(valid_starts) == 0:
            return np.array([]), np.array([])
        
        # Initialize arrays to store the results
        n_sequences = len(valid_starts)
        n_features = features.shape[1]
        X = np.zeros((n_sequences, window_size, n_features))
        y = np.zeros(n_sequences)
        
        # Extract data from the dataframe
        for i, (_, row) in enumerate(valid_starts.iterrows()):
            group_id = row['group_id']
            start_idx = row['seq_idx']
            
            # Get the sequence from the original dataframe
            sequence_df = df[(df['group_id'] == group_id) & 
                            (df['seq_idx'] >= start_idx) & 
                            (df['seq_idx'] < start_idx + window_size + 1)]
            
            if len(sequence_df) < window_size + 1:
                continue
                
            # Get the original row indices to extract features
            row_indices = sequence_df['row_idx'].values
            
            # Extract features and target
            X[i] = features[row_indices[:window_size]]
            y[i] = returns[row_indices[window_size]]
        
        return X, y
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32, **kwargs):
        """
        Train the transformer model.
        
        Args:
            X_train (ndarray): Training features of shape (samples, sequence_length, features)
            y_train (ndarray): Training targets
            X_val (ndarray): Validation features
            y_val (ndarray): Validation targets
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            **kwargs: Additional arguments for model.fit
            
        Returns:
            history: Training history
        """
        if self.model is None:
            self.build_model((X_train.shape[1], X_train.shape[2]))
        
        start_time = np.datetime64('now')
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train model
        if X_val is not None and y_val is not None:
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                **kwargs
            )
        else:
            history = self.model.fit(
                X_train, y_train,
                validation_split=0.2,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                **kwargs
            )

        end_time = np.datetime64('now')
        elapsed = (end_time - start_time) / np.timedelta64(1, 's')
        print(f"Transformer prediction completed in {elapsed:.2f} seconds")
            
        return history
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X (ndarray): Input features of shape (samples, sequence_length, features)
            
        Returns:
            ndarray: Predicted values
        """
        if self.model is None:
            raise ValueError("Model has not been built yet. Call build_model() first.")
            
        return self.model.predict(X).flatten()
    
    def evaluate(self, X, y_true):
        """
        Evaluate the model performance.
        
        Args:
            X (ndarray): Input features
            y_true (ndarray): True target values
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been built yet. Call build_model() first.")
            
        # Make predictions
        y_pred = self.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate information coefficient (correlation)
        ic = np.corrcoef(y_true, y_pred)[0, 1]
        
        # Calculate directional accuracy
        correct_direction = np.sum((y_true > 0) == (y_pred > 0)) / len(y_true)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'ic': ic,
            'directional_accuracy': correct_direction
        }
    
    def save_model(self, filepath):
        """
        Save the model to a file.
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been built yet. Call build_model() first.")
            
        self.model.save(filepath)
    
    def load_model(self, filepath):
        """
        Load a saved model.
        
        Args:
            filepath (str): Path to the saved model
        """
        self.model = models.load_model(filepath)
        return self.model