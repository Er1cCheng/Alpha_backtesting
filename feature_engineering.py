import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import time

class PyTorchAutoencoder(nn.Module):
    """
    PyTorch Autoencoder for feature dimension reduction.
    """
    def __init__(self, input_dim, encoding_dim=50, hidden_dim=128, dropout_rate=0.2):
        super(PyTorchAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        # Decode
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)


class AutoencoderDataset(Dataset):
    """
    Dataset for autoencoder training.
    """
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]  # Input and target are the same for autoencoder


class PyTorchFeatureEngineering:
    def __init__(self, data_dict, device=None, stock_count = None):
        """
        Initialize the PyTorch FeatureEngineering class with robust NaN handling.
        
        Args:
            data_dict (dict): Dictionary containing dataset keys:
                'x_data', 'y_data', 'si', 'di', 'raw_data', 'list_of_data'
            device (str): Device to use ('cpu' or 'cuda')
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"Using device: {self.device}")

        self.encode = encode
        
        self.data_dict = data_dict
        self.x_data = data_dict['x_data']
        self.y_data = data_dict['y_data']
        self.si = data_dict['si']
        self.di = data_dict['di']
        self.raw_data = data_dict['raw_data']
        self.list_of_data = data_dict['list_of_data']

        print(f"From day {self.di.min()} to day {self.di.max()}")

        if stock_count is not None:
            # Boolean index for desired rows
            mask = self.si < stock_count

            self.x_data  = self.x_data[mask]
            self.y_data  = self.y_data[mask]
            self.si = self.si[mask]
            self.di = self.di[mask]
            self.raw_data = self.raw_data[mask]

            print(f"Only including the top {stock_count} stocks:")
            print(f"Shape of x_data: {self.x_data.shape}, Shape of raw_data: {self.raw_data.shape}, Max stock index: {self.si.max()}")
            print(f"From day {self.di.min()} to day {self.di.max()}")
        
        # Initialize scalers (still use sklearn for preprocessing)
        self.alpha_scaler = StandardScaler()
        self.raw_data_scaler = StandardScaler()
        
        # Initialize encoders
        self.sector_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.industry_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
        # Initialize imputers
        self.x_imputer = SimpleImputer(strategy='mean')
        self.raw_data_imputer = SimpleImputer(strategy='mean')
        
        # Initialize autoencoder
        self.autoencoder = None
        
        # Track NaN statistics
        self._calculate_nan_statistics()
        
    def _calculate_nan_statistics(self):
        """
        Calculate and report statistics about NaN values in the dataset.
        """
        # Check for NaNs in various components
        nan_in_x = np.isnan(self.x_data).any()
        nan_count_x = np.isnan(self.x_data).sum()
        nan_in_y = np.isnan(self.y_data).any()
        nan_count_y = np.isnan(self.y_data).sum()
        nan_in_raw = np.isnan(self.raw_data).any()
        nan_count_raw = np.isnan(self.raw_data).sum()
        
        print(f"NaN Statistics:")
        print(f"  Alpha features (x_data): {nan_count_x} NaNs found")
        print(f"  Target values (y_data): {nan_count_y} NaNs found")
        print(f"  Raw data: {nan_count_raw} NaNs found")
        
        # Calculate percentage of NaNs per column in x_data
        if nan_in_x:
            nan_pct_by_col_x = np.isnan(self.x_data).mean(axis=0) * 100
            worst_cols_x = np.argsort(nan_pct_by_col_x)[-5:]  # Top 5 worst columns
            print(f"  Top 5 alpha features with most NaNs:")
            for col in worst_cols_x:
                print(f"    Feature {col}: {nan_pct_by_col_x[col]:.2f}% NaNs")
        
        # Calculate percentage of NaNs by column in raw_data
        if nan_in_raw:
            nan_pct_by_col_raw = np.isnan(self.raw_data).mean(axis=0) * 100
            worst_cols_raw = np.argsort(nan_pct_by_col_raw)[-5:]  # Top 5 worst columns
            print(f"  Top 5 raw features with most NaNs:")
            for i, col in enumerate(worst_cols_raw):
                print(f"    {self.list_of_data[col]}: {nan_pct_by_col_raw[col]:.2f}% NaNs")
    
    def _impute_input_data(self):
        """
        Impute missing values in input data.
        
        Returns:
            tuple: (imputed_x_data, imputed_raw_data)
        """
        # Impute x_data
        imputed_x_data = self.x_imputer.fit_transform(self.x_data)
        
        # Impute raw_data
        imputed_raw_data = self.raw_data_imputer.fit_transform(self.raw_data)
        
        return imputed_x_data, imputed_raw_data
    
    def train_test_split(self, split_date_idx=4400):
        """
        Split the data into training and testing datasets based on a date index.
        Handles NaN values appropriately.
        
        Args:
            split_date_idx (int): Day index to split training and testing data
                Default is 4400 (approximately 3 months before the end of 2023)
        
        Returns:
            dict: Dictionary containing training and testing datasets
        """
        # First, impute missing values in input data
        imputed_x_data, imputed_raw_data = self._impute_input_data()
        
        train_mask = self.di < split_date_idx
        test_mask = self.di >= split_date_idx
        
        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]
        
        train_test_dict = {
            'train_x': imputed_x_data[train_indices],
            'train_y': self.y_data[train_indices],
            'train_si': self.si[train_indices],
            'train_di': self.di[train_indices],
            'test_x': imputed_x_data[test_indices],
            'test_y': self.y_data[test_indices],
            'test_si': self.si[test_indices],
            'test_di': self.di[test_indices]
        }
        
        # Add raw data to the dictionary (now imputed)
        for i, feature_name in enumerate(self.list_of_data):
            train_test_dict[f'train_{feature_name}'] = imputed_raw_data[train_indices, i]
            train_test_dict[f'test_{feature_name}'] = imputed_raw_data[test_indices, i]
            
        # Check and report any remaining NaNs in the target variable
        train_y_nan = np.isnan(train_test_dict['train_y']).sum()
        test_y_nan = np.isnan(train_test_dict['test_y']).sum()
        if train_y_nan > 0 or test_y_nan > 0:
            print(f"Warning: Target variable still contains NaNs after split")
            print(f"  Training set: {train_y_nan} NaNs in targets")
            print(f"  Testing set: {test_y_nan} NaNs in targets")
            print(f"  These will need to be handled during model training/evaluation")
            
        return train_test_dict, train_indices, test_indices
    
    def build_autoencoder(self, input_dim, encoding_dim=50, hidden_dim=128, dropout_rate=0.2):
        """
        Build a PyTorch autoencoder for dimension reduction and feature extraction.
        
        Args:
            input_dim (int): Dimension of input features
            encoding_dim (int): Dimension of encoded representation
            hidden_dim (int): Dimension of hidden layer
            dropout_rate (float): Dropout rate for regularization
            
        Returns:
            PyTorchAutoencoder: The autoencoder model
        """
        # Create the autoencoder
        self.autoencoder = PyTorchAutoencoder(
            input_dim=input_dim,
            encoding_dim=encoding_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate
        ).to(self.device)
        
        return self.autoencoder
    
    def extract_alpha_embeddings(self, train_data, test_data, encoding_dim=50, 
                               epochs=100, batch_size=256, save_path=None, load_path=None):
        """
        Extract alpha embeddings using PyTorch autoencoder with robust NaN handling.
        
        Args:
            train_data (ndarray): Training data
            test_data (ndarray): Testing data
            encoding_dim (int): Dimension of encoded representation
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            save_path (str): Path to save the model
            load_path (str): Path to load a pretrained model
            
        Returns:
            tuple: (train_embeddings, test_embeddings)
        """
                
        # Handle NaNs in training data
        train_data = np.nan_to_num(train_data, nan=0.0)
        test_data = np.nan_to_num(test_data, nan=0.0)
        
        # Normalize data
        scaled_train_data = self.alpha_scaler.fit_transform(train_data)
        scaled_test_data = self.alpha_scaler.transform(test_data)

        if not self.encode:
            return scaled_train_data, scaled_test_data

        # New loading functionality
        if load_path:
            encoder_path = os.path.join(load_path, 'alpha_encoder.pt')
            scaler_path = os.path.join(load_path, 'alpha_scaler.pickle')
            
            if os.path.exists(encoder_path) and os.path.exists(scaler_path):
                print("Loading pretrained autoencoder...")
                # Load encoder and scaler
                self.autoencoder = torch.load(encoder_path, map_location=self.device, weights_only=False)
                with open(scaler_path, 'rb') as f:
                    self.alpha_scaler = pickle.load(f)

                # Convert to tensors
                train_tensor = torch.tensor(scaled_train_data, dtype=torch.float32).to(self.device)
                test_tensor = torch.tensor(scaled_test_data, dtype=torch.float32).to(self.device)
                
                # Use the loaded encoder to generate embeddings
                self.autoencoder.eval()
                with torch.no_grad():
                    train_embeddings = self.autoencoder.encode(train_tensor).cpu().numpy()
                    test_embeddings = self.autoencoder.encode(test_tensor).cpu().numpy()
                
                # Verify there are no NaNs in the embeddings
                if np.isnan(train_embeddings).any() or np.isnan(test_embeddings).any():
                    print("Warning: NaNs found in embeddings after prediction")
                    # Replace NaNs with zeros
                    train_embeddings = np.nan_to_num(train_embeddings, nan=0.0)
                    test_embeddings = np.nan_to_num(test_embeddings, nan=0.0)
                
                return train_embeddings, test_embeddings
        
        # Convert to PyTorch datasets
        train_dataset = AutoencoderDataset(scaled_train_data)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Build autoencoder if not already built
        if self.autoencoder is None:
            self.build_autoencoder(train_data.shape[1], encoding_dim)
        
        # Train autoencoder
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Training loop
        self.autoencoder.train()
        print(f"Training autoencoder for {epochs} epochs...")
        start_time = time.time()
        
        best_loss = float('inf')
        best_state = None
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_inputs, batch_targets in train_loader:
                # Move to device
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                # Forward pass
                outputs = self.autoencoder(batch_inputs)
                loss = criterion(outputs, batch_targets)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * batch_inputs.size(0)
            
            # Compute average epoch loss
            avg_epoch_loss = epoch_loss / len(train_dataset)
            
            # Early stopping
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_state = self.autoencoder.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                # Load best model
                self.autoencoder.load_state_dict(best_state)
                break
                
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.6f}")
        
        end_time = time.time()
        print(f"Autoencoder training completed in {end_time - start_time:.2f} seconds")
        
        # Save the model if a path is provided
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            encoder_path = os.path.join(save_path, 'alpha_encoder.pt')
            torch.save(self.autoencoder, encoder_path)
            
            # Also save the scaler
            scaler_path = os.path.join(save_path, 'alpha_scaler.pickle')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.alpha_scaler, f)
        
        # Extract embeddings
        self.autoencoder.eval()
        with torch.no_grad():
            train_tensor = torch.tensor(scaled_train_data, dtype=torch.float32).to(self.device)
            test_tensor = torch.tensor(scaled_test_data, dtype=torch.float32).to(self.device)
            
            train_embeddings = self.autoencoder.encode(train_tensor).cpu().numpy()
            test_embeddings = self.autoencoder.encode(test_tensor).cpu().numpy()
        
        # Verify there are no NaNs in the embeddings
        if np.isnan(train_embeddings).any() or np.isnan(test_embeddings).any():
            print("Warning: NaNs found in embeddings after prediction")
            # Replace NaNs with zeros
            train_embeddings = np.nan_to_num(train_embeddings, nan=0.0)
            test_embeddings = np.nan_to_num(test_embeddings, nan=0.0)
        
        return train_embeddings, test_embeddings
    
    def calculate_technical_indicators(self, stock_ids, day_indices, close_prices, high_prices, 
                                     low_prices, volumes, window_sizes=[5, 10, 20, 50]):
        """
        Calculate technical indicators using PyTorch for vectorized operations.
        
        Args:
            stock_ids (ndarray): Array of stock indices
            day_indices (ndarray): Array of day indices
            close_prices (ndarray): Array of closing prices
            high_prices (ndarray): Array of high prices
            low_prices (ndarray): Array of low prices
            volumes (ndarray): Array of trading volumes
            window_sizes (list): List of window sizes for moving averages
            
        Returns:
            ndarray: Array with technical indicators
        """
        # Handle NaNs in input data
        close_prices = np.nan_to_num(close_prices, nan=np.nanmean(close_prices))
        high_prices = np.nan_to_num(high_prices, nan=np.nanmean(high_prices))
        low_prices = np.nan_to_num(low_prices, nan=np.nanmean(low_prices))
        volumes = np.nan_to_num(volumes, nan=np.nanmean(volumes))
        
        # Create DataFrame with stock and day indices
        df = pd.DataFrame({
            'stock_id': stock_ids,
            'day_idx': day_indices,
            'close': close_prices,
            'high': high_prices,
            'low': low_prices,
            'volume': volumes
        })
        
        # Sort by stock ID and day index to ensure proper time series calculations
        df = df.sort_values(['stock_id', 'day_idx'])
        
        # Initialize a list to store all feature columns
        feature_columns = []
        
        # Group by stock ID for calculations that need to be done per stock
        grouped = df.groupby('stock_id')
        
        # Moving Averages (for close price and volume)
        for window in window_sizes:
            # Close price moving averages
            ma_col = f'ma_{window}'
            df[ma_col] = grouped['close'].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
            feature_columns.append(ma_col)
            
            # Volume moving averages
            vol_ma_col = f'volume_ma_{window}'
            df[vol_ma_col] = grouped['volume'].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
            feature_columns.append(vol_ma_col)
        
        # Relative Strength Index (RSI)
        for window in window_sizes:
            rsi_col = f'rsi_{window}'
            # Calculate price changes - Fill NaN with 0 for first row
            df['price_delta'] = grouped['close'].transform(lambda x: x.diff().fillna(0))
            
            # Calculate gains and losses
            df['gain'] = df['price_delta'].clip(lower=0)
            df['loss'] = -df['price_delta'].clip(upper=0)
            
            # Calculate average gains and losses - Use min_periods=1
            df['avg_gain'] = grouped['gain'].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
            df['avg_loss'] = grouped['loss'].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
            
            # Handle division by zero when all losses are zero
            df['rs'] = df['avg_gain'] / df['avg_loss'].replace(0, 1e-10)
            df[rsi_col] = 100 - (100 / (1 + df['rs']))
            feature_columns.append(rsi_col)
            
            # Drop temporary columns
            df = df.drop(['price_delta', 'gain', 'loss', 'avg_gain', 'avg_loss', 'rs'], axis=1)
        
        # Average True Range (ATR)
        # Calculate true range components
        df['hl'] = df['high'] - df['low']
        # Use the row's own value for first point instead of NaN
        df['hc'] = (df['high'] - df.groupby('stock_id')['close'].shift().fillna(df['high'])).abs()
        df['lc'] = (df['low'] - df.groupby('stock_id')['close'].shift().fillna(df['low'])).abs()
        
        # Calculate true range
        df['tr'] = df[['hl', 'hc', 'lc']].max(axis=1)
        
        # Calculate ATR for different windows - Use min_periods=1
        for window in window_sizes:
            atr_col = f'atr_{window}'
            df[atr_col] = df.groupby('stock_id')['tr'].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
            feature_columns.append(atr_col)
        
        # Drop temporary columns
        df = df.drop(['hl', 'hc', 'lc', 'tr'], axis=1)
        
        # Momentum
        for window in window_sizes:
            mom_col = f'momentum_{window}'
            # For first points, use the current value to avoid division by zero (resulting in 0% change)
            df[mom_col] = grouped['close'].transform(
                lambda x: x / x.shift(window).fillna(x) - 1
            )
            feature_columns.append(mom_col)
        
        # VOLATILITY CALCULATION
        for window in window_sizes:
            vol_col = f'volatility_{window}'
            
            # Process each stock group separately to ensure we don't mix data across stocks
            all_volatilities = []
            
            for stock_id, group in grouped:
                # Ensure there are no zeros or NaNs in close prices
                group_close = group['close'].replace(0, np.nan).ffill().bfill()
                
                if len(group_close) <= 1:
                    # If only one data point, use a default volatility
                    group['vol'] = 0.01
                else:
                    # Calculate log returns for better numerical stability
                    log_returns = np.log(group_close / group_close.shift(1))
                    
                    # Replace infinite values that might occur from log(0) with NaN, then with 0
                    log_returns = log_returns.replace([np.inf, -np.inf], np.nan).fillna(0)
                    
                    # Calculate rolling volatility with proper minimum periods
                    min_periods = min(window, len(log_returns) - 1)
                    min_periods = max(1, min_periods)  # Ensure at least 1
                    
                    # Calculate rolling standard deviation of returns
                    vol = log_returns.rolling(window=window, min_periods=min_periods).std()
                    
                    # Fill any NaNs at the beginning with the first valid volatility
                    first_valid = vol.first_valid_index()
                    if first_valid is not None:
                        first_valid_value = vol.loc[first_valid]
                        vol = vol.fillna(first_valid_value)
                    else:
                        # If no valid values, use default
                        vol = vol.fillna(0.01)
                    
                    # Set a minimum volatility floor and annualize (√252 is standard annualization factor)
                    vol = np.maximum(vol, 0.001) * np.sqrt(252)
                    
                    group['vol'] = vol
                
                all_volatilities.append(group[['stock_id', 'day_idx', 'vol']])
            
            # Combine all processed groups
            if all_volatilities:
                vol_df = pd.concat(all_volatilities)
                
                # Merge back to main dataframe
                df = df.drop(vol_col, axis=1, errors='ignore')  # Remove if exists
                df = pd.merge(df, vol_df.rename(columns={'vol': vol_col}), on=['stock_id', 'day_idx'], how='left')
                
                # Final sanity check - replace any remaining NaNs with default volatility
                df[vol_col] = df[vol_col].fillna(0.01)
                
                feature_columns.append(vol_col)
        
        # Extract the feature columns in the original order
        features_df = df[feature_columns]
        
        # Check for any remaining NaNs
        nan_count_by_col = features_df.isna().sum()
        total_nans = nan_count_by_col.sum()
        
        if total_nans > 0:
            print(f"NaN Analysis by Technical Indicator (total: {total_nans}):")
            for col in nan_count_by_col[nan_count_by_col > 0].index:
                print(f"  {col}: {nan_count_by_col[col]} NaNs")
            
            # Fill any remaining NaNs with column means
            print("Filling remaining NaNs with column means")
            features_df = features_df.fillna(features_df.mean())
        else:
            print("No NaNs detected in technical indicators - calculation was successful")
        
        # Make sure we maintain the original order
        df_orig_order = pd.DataFrame({'stock_id': stock_ids, 'day_idx': day_indices})
        df = pd.merge(df_orig_order, df, on=['stock_id', 'day_idx'], how='left')
        
        # Final check for NaNs after the merge operation
        final_nan_count = df[feature_columns].isna().sum().sum()
        if final_nan_count > 0:
            print(f"Warning: {final_nan_count} NaNs introduced after merging back to original order")
            # Fill with column means
            result = df[feature_columns].fillna(features_df.mean()).values
        else:
            result = df[feature_columns].values
        
        # Final sanity check
        if np.isnan(result).any():
            print("Final emergency NaN fixing - replacing with zeros")
            result = np.nan_to_num(result, nan=0.0)
        else:
            print("No NaNs in final output - technical indicators successfully calculated")
        
        return result
    
    def encode_categorical_features(self, train_sector_data, train_industry_data, test_sector_data=None, test_industry_data=None):
        """
        Encode categorical features using one-hot encoding with consistent columns.
        
        Args:
            train_sector_data (ndarray): Array of training sector data
            train_industry_data (ndarray): Array of training industry data
            test_sector_data (ndarray, optional): Array of testing sector data
            test_industry_data (ndarray, optional): Array of testing industry data
            
        Returns:
            tuple: (train_encoded_sector, train_encoded_industry, test_encoded_sector, test_encoded_industry)
                  or (train_encoded_sector, train_encoded_industry) if test data not provided
        """
        # Check for NaNs
        train_nan_in_sector = np.isnan(train_sector_data).any()
        train_nan_in_industry = np.isnan(train_industry_data).any()
        
        if train_nan_in_sector or train_nan_in_industry:
            print("Warning: NaNs found in training categorical features")
            # Fill NaNs with a special value
            train_sector_data = np.nan_to_num(train_sector_data, nan=-1)
            train_industry_data = np.nan_to_num(train_industry_data, nan=-1)
        
        # Process test data if provided
        is_test_provided = test_sector_data is not None and test_industry_data is not None
        
        if is_test_provided:
            test_nan_in_sector = np.isnan(test_sector_data).any()
            test_nan_in_industry = np.isnan(test_industry_data).any()
            
            if test_nan_in_sector or test_nan_in_industry:
                print("Warning: NaNs found in testing categorical features")
                # Fill NaNs with a special value
                test_sector_data = np.nan_to_num(test_sector_data, nan=-1)
                test_industry_data = np.nan_to_num(test_industry_data, nan=-1)
        
        # Reshape data for fitting
        train_sector_reshaped = train_sector_data.reshape(-1, 1)
        train_industry_reshaped = train_industry_data.reshape(-1, 1)
        
        if is_test_provided:
            test_sector_reshaped = test_sector_data.reshape(-1, 1)
            test_industry_reshaped = test_industry_data.reshape(-1, 1)
            
            # Combine train and test to get all possible categories for fitting
            all_sector_data = np.vstack([train_sector_reshaped, test_sector_reshaped])
            all_industry_data = np.vstack([train_industry_reshaped, test_industry_reshaped])
            
            # Fit on all data to ensure consistent categories
            print("Fitting encoders on combined train+test data to ensure consistent features")
            self.sector_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            self.industry_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            
            self.sector_encoder.fit(all_sector_data)
            self.industry_encoder.fit(all_industry_data)
            
            # Transform each dataset separately
            train_encoded_sector = self.sector_encoder.transform(train_sector_reshaped)
            train_encoded_industry = self.industry_encoder.transform(train_industry_reshaped)
            test_encoded_sector = self.sector_encoder.transform(test_sector_reshaped)
            test_encoded_industry = self.industry_encoder.transform(test_industry_reshaped)
            
            # Verify shapes match
            print(f"Encoded sector shapes - Train: {train_encoded_sector.shape}, Test: {test_encoded_sector.shape}")
            print(f"Encoded industry shapes - Train: {train_encoded_industry.shape}, Test: {test_encoded_industry.shape}")
            
            return train_encoded_sector, train_encoded_industry, test_encoded_sector, test_encoded_industry
        else:
            # Original behavior when only training data is provided
            self.sector_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            self.industry_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            
            train_encoded_sector = self.sector_encoder.fit_transform(train_sector_reshaped)
            train_encoded_industry = self.industry_encoder.fit_transform(train_industry_reshaped)
            
            return train_encoded_sector, train_encoded_industry
    
    def generate_features(self, window_sizes=[5, 10, 20, 50], encoding_dim=50, output_dir=None):
        """
        Generate features by combining alpha embeddings, technical indicators, and encoded categories.
        With PyTorch implementation for enhanced performance.
        
        Args:
            window_sizes (list): List of window sizes for technical indicators
            encoding_dim (int): Dimension of encoded representation for alpha signals
            output_dir (str): Directory to save/load models
            
        Returns:
            dict: Dictionary containing training and testing datasets with engineered features
        """
        print("\n=== Starting PyTorch Feature Generation Pipeline ===\n")
        
        # Create output directory if it doesn't exist
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Split data into training and testing sets
        print("Splitting data into training and testing sets...")
        train_test_dict, train_indices, test_indices = self.train_test_split()
        
        # Extract alpha embeddings using PyTorch autoencoder
        print("\nExtracting alpha embeddings with PyTorch autoencoder...")
        train_embeddings, test_embeddings = self.extract_alpha_embeddings(
            train_test_dict['train_x'],
            train_test_dict['test_x'],
            encoding_dim=encoding_dim,
            save_path=output_dir,
            load_path=output_dir
        )
        
        # Calculate technical indicators for training set
        print("\nCalculating technical indicators for training set...")
        train_tech_indicators = self.calculate_technical_indicators(
            train_test_dict['train_si'],
            train_test_dict['train_di'],
            train_test_dict['train_close'],
            train_test_dict['train_high'],
            train_test_dict['train_low'],
            train_test_dict['train_volume'],
            window_sizes=window_sizes
        )
        
        # Calculate technical indicators for testing set
        print("\nCalculating technical indicators for testing set...")
        test_tech_indicators = self.calculate_technical_indicators(
            train_test_dict['test_si'],
            train_test_dict['test_di'],
            train_test_dict['test_close'],
            train_test_dict['test_high'],
            train_test_dict['test_low'],
            train_test_dict['test_volume'],
            window_sizes=window_sizes
        )
        
        # Encode categorical features - using the method that ensures consistent features
        print("\nEncoding categorical features...")
        train_encoded_sector, train_encoded_industry, test_encoded_sector, test_encoded_industry = self.encode_categorical_features(
            train_test_dict['train_sector'],
            train_test_dict['train_industry'],
            train_test_dict['test_sector'],
            train_test_dict['test_industry']
        )
        
        # Normalize SPX return and VIX features
        print("\nNormalizing market features...")
        train_spx_vix = np.column_stack((
            train_test_dict['train_ret1_SPX'],
            train_test_dict['train_close_VIX']
        ))
        test_spx_vix = np.column_stack((
            train_test_dict['test_ret1_SPX'],
            train_test_dict['test_close_VIX']
        ))
        
        # Handle NaNs in market data
        train_spx_vix = np.nan_to_num(train_spx_vix, nan=0.0)
        test_spx_vix = np.nan_to_num(test_spx_vix, nan=0.0)
        
        # Fit on combined data to ensure consistent scaling
        all_spx_vix = np.vstack([train_spx_vix, test_spx_vix])
        self.raw_data_scaler.fit(all_spx_vix)
        
        scaled_train_spx_vix = self.raw_data_scaler.transform(train_spx_vix)
        scaled_test_spx_vix = self.raw_data_scaler.transform(test_spx_vix)
        
        # Create announcement features
        print("\nProcessing announcement features...")
        train_ann_features = np.column_stack((
            train_test_dict['train_trading_days_til_next_ann'],
            train_test_dict['train_trading_days_since_last_ann']
        ))
        test_ann_features = np.column_stack((
            train_test_dict['test_trading_days_til_next_ann'],
            train_test_dict['test_trading_days_since_last_ann']
        ))
        
        # Handle NaNs in announcement data
        train_ann_features = np.nan_to_num(train_ann_features, nan=30)  # Use 30 days as default
        test_ann_features = np.nan_to_num(test_ann_features, nan=30)
        
        # Combine all features
        print("\nCombining all features...")
        train_features = np.hstack((
            train_test_dict['train_x'],      # Original alpha signals
            train_embeddings,                # Alpha embeddings
            train_tech_indicators,           # Technical indicators
            train_encoded_sector,            # Encoded sector
            train_encoded_industry,          # Encoded industry
            scaled_train_spx_vix,            # Normalized SPX and VIX
            train_ann_features               # Announcement features
        ))
        
        test_features = np.hstack((
            train_test_dict['test_x'],
            test_embeddings,
            test_tech_indicators,
            test_encoded_sector,
            test_encoded_industry,
            scaled_test_spx_vix,
            test_ann_features
        ))
        
        # Verify feature dimension consistency
        if train_features.shape[1] != test_features.shape[1]:
            print(f"\n!!! WARNING: Feature dimension mismatch !!!")
            print(f"Training features shape: {train_features.shape}")
            print(f"Testing features shape: {test_features.shape}")
            
            # Handle the mismatch - expand the smaller dimension set
            if train_features.shape[1] > test_features.shape[1]:
                diff = train_features.shape[1] - test_features.shape[1]
                print(f"Expanding test features with {diff} zero columns to match training dimensions")
                test_features = np.hstack([test_features, np.zeros((test_features.shape[0], diff))])
            else:
                diff = test_features.shape[1] - train_features.shape[1]
                print(f"Expanding training features with {diff} zero columns to match test dimensions")
                train_features = np.hstack([train_features, np.zeros((train_features.shape[0], diff))])
        
        # Final NaN check
        train_nan_count = np.isnan(train_features).sum()
        test_nan_count = np.isnan(test_features).sum()
        
        if train_nan_count > 0 or test_nan_count > 0:
            print(f"\nWarning: Found {train_nan_count} NaNs in train features and {test_nan_count} in test features")
            print("Replacing remaining NaNs with zeros...")
            train_features = np.nan_to_num(train_features, nan=0.0)
            test_features = np.nan_to_num(test_features, nan=0.0)
        
        # Update dictionary with engineered features
        train_test_dict['train_features'] = train_features
        train_test_dict['test_features'] = test_features
        
        # Add stock and day indices
        train_test_dict['unique_stocks'] = np.unique(self.si)
        train_test_dict['unique_days'] = np.unique(self.di)
        
        print("\n=== PyTorch Feature Generation Complete ===")
        print(f"Training features shape: {train_features.shape}")
        print(f"Testing features shape: {test_features.shape}")
        print(f"Feature consistency verified: {train_features.shape[1] == test_features.shape[1]}")
        
        return train_test_dict