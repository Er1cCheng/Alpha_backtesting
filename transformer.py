import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


import torch
import torch.nn as nn

class DirectionalMSELoss(nn.Module):
    def __init__(self, directional_penalty=1.0):
        """
        Custom loss function that combines MSE with directional penalty.
        
        Args:
            directional_penalty (float): Weight of the directional penalty term
        """
        super(DirectionalMSELoss, self).__init__()
        self.mse = nn.L1Loss()
        self.directional_penalty = directional_penalty
        
    def forward(self, predictions, targets):
        # Standard MSE loss
        mse_loss = self.mse(predictions, targets)
        
        # Directional penalty: apply when prediction < 0 and target > 0
        directional_loss = torch.relu(-(predictions * targets)).sum()
        
        # Calculate the directional penalty (could be modified for different penalties)
        # if torch.any(directional_mask):
        #     directional_loss = torch.mean(
        #         torch.abs(predictions[directional_mask]) + torch.abs(targets[directional_mask])
        #     )
        # else:
        #     directional_loss = torch.tensor(0.0, device=predictions.device)
        
        # Combine the losses
        total_loss = mse_loss + self.directional_penalty * directional_loss
        
        return total_loss

# TimeSeriesDataset for PyTorch
class TimeSeriesDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=20):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        # Apply sin to even indices and cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension
        pe = pe.unsqueeze(0)
        
        # Register the positional encoding as a buffer (not a parameter)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Add positional encoding to the input
        # x is of shape (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout_rate)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # Multi-head attention
        # PyTorch's MultiheadAttention expects input of shape (seq_len, batch_size, d_model)
        x_t = x.transpose(0, 1)
        attn_output, _ = self.attention(x_t, x_t, x_t)
        attn_output = attn_output.transpose(0, 1)  # Back to (batch_size, seq_len, d_model)
        
        # Add & norm
        x1 = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward
        ff_output = self.feed_forward(x1)
        
        # Add & norm
        x2 = self.norm2(x1 + self.dropout(ff_output))
        
        return x2

# PyTorch TimeSeriesTransformer Model
class PyTorchTimeSeriesTransformer(nn.Module):
    def __init__(self, 
                 input_dim,
                 max_seq_length=20, 
                 d_model=128, 
                 num_heads=8, 
                 d_ff=256, 
                 num_layers=4, 
                 dropout_rate=0.1,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the TimeSeriesTransformer for stock return prediction.
        """
        super(PyTorchTimeSeriesTransformer, self).__init__()
        
        self.max_seq_length = max_seq_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.device = device
        
        # Input projection layer
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding layer
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Transformer encoder layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout_rate)
            for _ in range(num_layers)
        ])
        
        # Global average pooling layer
        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)
        
        # Output layer
        self.output_layer = nn.Linear(d_model, 1)
        
        # Move model to specified device
        self.to(device)
        
    def forward(self, x):
        """
        Forward pass of the TimeSeriesTransformer.
        """
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # Global average pooling
        x = x.transpose(1, 2)  # Change to (batch_size, d_model, seq_length)
        x = x[:, :, -1]  # Shape: (batch_size, d_model, 1)
        x = x.squeeze(-1)  # Shape: (batch_size, d_model)
        
        # Output layer
        x = self.output_layer(x)  # Shape: (batch_size, 1)
        x = x.squeeze(-1)  # Shape: (batch_size)
        
        return x

# TimeSeriesTransformer wrapper class that mimics the original API
class TimeSeriesTransformer:
    def __init__(self, 
                 max_seq_length=20, 
                 d_model=128, 
                 num_heads=8, 
                 d_ff=256, 
                 num_layers=4, 
                 dropout_rate=0.1,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the TimeSeriesTransformer for stock return prediction.
        """
        self.max_seq_length = max_seq_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.device = device
        self.model = None
        self.optimizer = None
        self.criterion = None
        
    def build_model(self, input_shape):
        """
        Build the transformer model for time series prediction.
        """
        feature_dim = input_shape[1]

        print("Building, input shape: ", input_shape)
        
        # Create PyTorch model
        self.model = PyTorchTimeSeriesTransformer(
            input_dim=feature_dim,
            max_seq_length=self.max_seq_length,
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            num_layers=self.num_layers,
            dropout_rate=self.dropout_rate,
            device=self.device
        )
        
        # Define optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = DirectionalMSELoss(directional_penalty=100)
        
        return self.model
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32, verbose=1, **kwargs):
        """
        Train the transformer model.
        """
        if self.model is None:
            self.build_model((X_train.shape[1], X_train.shape[2]))
        
        start_time = np.datetime64('now')
        
        # Set model to training mode
        self.model.train()
        
        # Create PyTorch datasets and data loaders
        train_dataset = TimeSeriesDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        del train_dataset
        
        if X_val is not None and y_val is not None:
            val_dataset = TimeSeriesDataset(X_val, y_val)
            del val_dataset 
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize early stopping and learning rate reduction parameters
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        lr_reduction_counter = 0
        lr_patience = 5
        lr_factor = 0.5
        min_lr = 1e-6
        
        # Initialize history
        history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}
        best_model_state = None
        
        # Training loop
        for epoch in range(epochs):
            # Initialize metrics for this epoch
            train_loss = 0
            train_mae = 0
            n_batches = 0
            
            # Training on batches
            for batch_X, batch_y in train_loader:
                # Move data to device
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_X)
                
                # Calculate loss
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                train_loss += loss.item()
                train_mae += torch.mean(torch.abs(outputs - batch_y)).item()
                n_batches += 1
            
            # Calculate average metrics for this epoch
            avg_train_loss = train_loss / n_batches
            avg_train_mae = train_mae / n_batches
            
            # Add to history
            history['train_loss'].append(avg_train_loss)
            history['train_mae'].append(avg_train_mae)
            
            # Validation if validation data is provided
            if X_val is not None and y_val is not None:
                self.model.eval()
                val_loss = 0
                val_mae = 0
                n_val_batches = 0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        # Move data to device
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        # Forward pass
                        outputs = self.model(batch_X)
                        
                        # Calculate loss
                        loss = self.criterion(outputs, batch_y)
                        
                        # Update metrics
                        val_loss += loss.item()
                        val_mae += torch.mean(torch.abs(outputs - batch_y)).item()
                        n_val_batches += 1
                
                # Calculate average metrics
                avg_val_loss = val_loss / n_val_batches
                avg_val_mae = val_mae / n_val_batches
                
                # Add to history
                history['val_loss'].append(avg_val_loss)
                history['val_mae'].append(avg_val_mae)
                
                # Early stopping check
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    # Save the best model
                    best_model_state = {
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'epoch': epoch,
                        'loss': best_val_loss
                    }
                    patience_counter = 0
                    lr_reduction_counter = 0
                else:
                    patience_counter += 1
                    lr_reduction_counter += 1
                    
                    # Learning rate reduction
                    if lr_reduction_counter >= lr_patience:
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = max(param_group['lr'] * lr_factor, min_lr)
                        lr_reduction_counter = 0
                        if verbose > 0:
                            print(f"Reduced learning rate to {self.optimizer.param_groups[0]['lr']}")
                    
                    # Early stopping
                    if patience_counter >= patience:
                        if verbose > 0:
                            print(f"Early stopping at epoch {epoch+1}")
                        # Restore best model
                        if best_model_state:
                            self.model.load_state_dict(best_model_state['model_state_dict'])
                            self.optimizer.load_state_dict(best_model_state['optimizer_state_dict'])
                        break
                
                # Print epoch results
                if verbose > 0 and ((epoch + 1) % 10) == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Train MAE: {avg_train_mae:.4f} - Val Loss: {avg_val_loss:.4f} - Val MAE: {avg_val_mae:.4f}")
            else:
                # Print train-only results
                if verbose > 0 and ((epoch + 1) % 10) == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Train MAE: {avg_train_mae:.4f}")
                
                # For train-only mode, save the model if loss improved
                if avg_train_loss < best_val_loss:
                    best_val_loss = avg_train_loss
                    best_model_state = {
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'epoch': epoch,
                        'loss': best_val_loss
                    }
            
            # Set back to training mode
            self.model.train()
        
        # Restore best model at end of training
        if best_model_state:
            self.model.load_state_dict(best_model_state['model_state_dict'])
            self.optimizer.load_state_dict(best_model_state['optimizer_state_dict'])
        
        end_time = np.datetime64('now')
        elapsed = (end_time - start_time) / np.timedelta64(1, 's')
        print(f"Transformer training completed in {elapsed:.2f} seconds")
        
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
        
        # Convert to PyTorch tensor
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        return predictions
    
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
        
        # Save both model architecture and weights
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hyperparams': {
                'max_seq_length': self.max_seq_length,
                'd_model': self.d_model,
                'num_heads': self.num_heads,
                'd_ff': self.d_ff,
                'num_layers': self.num_layers,
                'dropout_rate': self.dropout_rate,
                'input_dim': self.model.input_projection.in_features
            }
        }, filepath)
    
    def load_model(self, filepath):
        """
        Load a saved model.
        
        Args:
            filepath (str): Path to the saved model
        """
        # Load saved state
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Extract hyperparameters
        hyperparams = checkpoint['hyperparams']
        input_dim = hyperparams['input_dim']
        
        # Create model with saved architecture
        self.max_seq_length = hyperparams['max_seq_length']
        self.d_model = hyperparams['d_model']
        self.num_heads = hyperparams['num_heads']
        self.d_ff = hyperparams['d_ff']
        self.num_layers = hyperparams['num_layers']
        self.dropout_rate = hyperparams['dropout_rate']
        
        # Build the model
        self.model = PyTorchTimeSeriesTransformer(
            input_dim=input_dim,
            max_seq_length=self.max_seq_length,
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            num_layers=self.num_layers,
            dropout_rate=self.dropout_rate,
            device=self.device
        )
        
        # Load model and optimizer states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Define loss function
        self.criterion = nn.MSELoss()
        
        return self.model