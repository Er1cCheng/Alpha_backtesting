import numpy as np
from sklearn.neighbors import KernelDensity, KNeighborsRegressor, NearestNeighbors
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed

class KernelRegression:
    def __init__(self, bandwidth='auto', kernel='gaussian', n_neighbors=100, n_jobs=-1):
        """
        Initialize the Kernel Regression model with NaN handling and performance optimizations.
        
        Args:
            bandwidth (str or float): Bandwidth for the kernel. If 'auto', it will be selected using cross-validation.
            kernel (str): Kernel type ('gaussian', 'epanechnikov', etc.)
            n_neighbors (int): Maximum number of neighbors to use for prediction (speeds up computation)
            n_jobs (int): Number of jobs for parallel processing (-1 uses all cores)
        """
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.X_imputer = SimpleImputer(strategy='mean')
        self.y_imputer = SimpleImputer(strategy='mean')
        self.best_bandwidth = None
        self.nn_model = None
        self._is_fitted = False
        self.feature_names = None
        self.n_features = None
    
    def _optimize_bandwidth(self, X_train, y_train):
        """
        Find the optimal bandwidth using cross-validation.
        Uses a more efficient approach with KNeighborsRegressor.
        """
        print("Optimizing bandwidth using cross-validation...")
        
        # Define the parameter grid
        param_grid = {'n_neighbors': [5, 10, 20, 50, 100]}
        
        # Define TimeSeriesSplit for time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)  # Reduced splits for speed
        
        try:
            # Use sample of data for faster CV if dataset is large
            sample_size = min(10000, X_train.shape[0])
            if X_train.shape[0] > sample_size:
                # Use a stratified sampling approach if possible
                indices = np.random.choice(X_train.shape[0], sample_size, replace=False)
                X_sample = X_train[indices]
                y_sample = y_train[indices]
                print(f"Using {sample_size} samples for bandwidth optimization")
            else:
                X_sample = X_train
                y_sample = y_train
            
            grid_search = GridSearchCV(
                KNeighborsRegressor(),
                param_grid=param_grid,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=self.n_jobs,
                verbose=1
            )
            
            grid_search.fit(X_sample, y_sample)
            best_n_neighbors = grid_search.best_params_['n_neighbors']
            print(f"Best n_neighbors: {best_n_neighbors}")
        except Exception as e:
            print(f"Error in grid search: {e}")
            print("Using default n_neighbors=50")
            best_n_neighbors = 50
        
        # Rule of thumb: bandwidth is approximately proportional to n_neighbors^(1/5)
        # for a dataset size of n
        n = X_train.shape[0]
        best_bandwidth = best_n_neighbors * (n ** (-1/5))
        
        print(f"Selected bandwidth: {best_bandwidth:.6f}")
        return best_bandwidth
    
    def fit(self, X_train, y_train, feature_names=None):
        """
        Fit the kernel regression model to the data with NaN handling.
        Uses nearest neighbors for faster prediction.
        
        Args:
            X_train: Training features
            y_train: Training targets
            feature_names: Optional list of feature names (helps with dimension consistency)
        """
        print(f"Fitting KernelRegression model...")
        start_time = np.datetime64('now')
        
        # Store feature dimensionality
        self.n_features = X_train.shape[1]
        print(f"Training with {self.n_features} features")
        
        # Store feature names if provided
        if feature_names is not None:
            self.feature_names = feature_names
            print(f"Feature names stored for consistency checks")
        
        # Ensure inputs are numpy arrays
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        
        # Handle NaN values in features and targets
        print(f"Input shapes before imputation - X: {X_train.shape}, y: {y_train.shape}")
        print(f"NaN counts - X: {np.isnan(X_train).sum()}, y: {np.isnan(y_train).sum()}")
        
        # First handle NaN in target variable as we might need to drop rows
        y_nan_mask = np.isnan(y_train)
        if np.any(y_nan_mask):
            X_train = X_train[~y_nan_mask]
            y_train = y_train[~y_nan_mask]
            print(f"Dropped {np.sum(y_nan_mask)} rows with NaN targets")
        
        # Now handle NaNs in features using imputation
        X_train = self.X_imputer.fit_transform(X_train)
        y_train = self.y_imputer.fit_transform(y_train.reshape(-1, 1)).flatten()
        
        # Scale the features and target after imputation
        X_scaled = self.X_scaler.fit_transform(X_train)
        y_scaled = self.y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        
        print(f"Shapes after imputation and scaling - X: {X_scaled.shape}, y: {y_scaled.shape}")
        
        # Optimize bandwidth if 'auto'
        if self.bandwidth == 'auto':
            self.best_bandwidth = self._optimize_bandwidth(X_scaled, y_scaled)
        else:
            self.best_bandwidth = self.bandwidth
        
        # Build nearest neighbors model for efficient lookup
        self.nn_model = NearestNeighbors(
            n_neighbors=min(self.n_neighbors, X_scaled.shape[0]),
            algorithm='ball_tree',
            n_jobs=self.n_jobs
        )
        self.nn_model.fit(X_scaled)
        
        # Store the training data
        self.X_train_scaled = X_scaled
        self.y_train_scaled = y_scaled
        
        self._is_fitted = True
        end_time = np.datetime64('now')
        elapsed = (end_time - start_time) / np.timedelta64(1, 's')
        print(f"KernelRegression training completed in {elapsed:.2f} seconds")
        return self
    
    def _ensure_feature_consistency(self, X):
        """
        Ensures the feature dimensions match what the model was trained on.
        If there's a mismatch, handles it by padding or truncating.
        
        Args:
            X: Input features
            
        Returns:
            X with consistent dimensionality
        """
        if X.shape[1] != self.n_features:
            print(f"WARNING: Feature dimension mismatch! Got {X.shape[1]}, expected {self.n_features}")
            
            if X.shape[1] < self.n_features:
                # Need to add padding columns
                padding_width = self.n_features - X.shape[1]
                print(f"Adding {padding_width} padding columns with zeros")
                padding = np.zeros((X.shape[0], padding_width))
                X_padded = np.hstack((X, padding))
                return X_padded
            else:
                # Need to truncate columns
                print(f"Truncating to first {self.n_features} features")
                return X[:, :self.n_features]
        return X
    
    def _process_batch(self, X_batch, indices, distances):
        """Process a batch of predictions in parallel"""
        predictions = np.zeros(X_batch.shape[0])
        
        for i in range(X_batch.shape[0]):
            # Get the neighbors and distances for this point
            neighbor_indices = indices[i]
            neighbor_distances = distances[i]
            
            # Compute kernel weights
            weights = np.exp(-neighbor_distances / (2 * (self.best_bandwidth ** 2)))
            
            # Normalize weights
            sum_weights = np.sum(weights)
            if sum_weights > 0:
                weights = weights / sum_weights
                # Compute weighted average of target values
                predictions[i] = np.sum(weights * self.y_train_scaled[neighbor_indices])
            else:
                # Fallback to mean if all weights are zero
                predictions[i] = np.mean(self.y_train_scaled)
                
        return predictions
    
    def predict(self, X, batch_size=1000):
        """
        Make predictions using the Nadaraya-Watson kernel regression.
        Optimized with nearest neighbors and batch processing.
        Handles feature dimension mismatches.
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        print(f"Predicting with KernelRegression...")
        start_time = np.datetime64('now')
        
        # Ensure inputs are numpy arrays
        X = np.asarray(X)
        
        # Check feature dimensionality and adjust if needed
        if X.shape[1] != self.n_features:
            X = self._ensure_feature_consistency(X)
        
        # Handle NaN values in test data using the same imputation strategy
        try:
            X_imputed = self.X_imputer.transform(X)
        except ValueError as e:
            print(f"Error in imputer transform: {e}")
            print(f"Input shape: {X.shape}, Expected features: {self.n_features}")
            print(f"Falling back to manual NaN handling")
            # Manual NaN handling as fallback
            X_imputed = np.copy(X)
            for j in range(X.shape[1]):
                col_mask = np.isnan(X[:, j])
                if np.any(col_mask):
                    mean_val = np.nanmean(X[:, j]) if not np.all(col_mask) else 0
                    X_imputed[col_mask, j] = mean_val
        
        # Scale the features
        X_scaled = self.X_scaler.transform(X_imputed)
        n_samples = X_scaled.shape[0]
        
        # Use nearest neighbors for efficient distance computation
        n_neighbors = min(self.n_neighbors, self.X_train_scaled.shape[0])
        distances, indices = self.nn_model.kneighbors(X_scaled)
        
        # Process in batches for better memory management and parallel processing
        n_batches = int(np.ceil(n_samples / batch_size))
        y_pred_scaled = np.zeros(n_samples)
        
        for i in range(n_batches):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, n_samples)
            
            if self.n_jobs != 1:
                # Use parallel processing for large batches
                X_batch = X_scaled[batch_start:batch_end]
                batch_indices = indices[batch_start:batch_end]
                batch_distances = distances[batch_start:batch_end]
                
                y_pred_scaled[batch_start:batch_end] = self._process_batch(
                    X_batch, batch_indices, batch_distances
                )
            else:
                # Sequential processing for smaller batches
                for j in range(batch_start, batch_end):
                    neighbor_indices = indices[j]
                    neighbor_distances = distances[j]
                    
                    # Compute kernel weights
                    weights = np.exp(-neighbor_distances / (2 * (self.best_bandwidth ** 2)))
                    
                    # Normalize weights
                    sum_weights = np.sum(weights)
                    if sum_weights > 0:
                        weights = weights / sum_weights
                        y_pred_scaled[j] = np.sum(weights * self.y_train_scaled[neighbor_indices])
                    else:
                        y_pred_scaled[j] = np.mean(self.y_train_scaled)
        
        # Inverse transform to get predictions in original scale
        y_pred = self.y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        end_time = np.datetime64('now')
        elapsed = (end_time - start_time) / np.timedelta64(1, 's')
        print(f"KernelRegression prediction completed in {elapsed:.2f} seconds for {n_samples} samples")
        
        return y_pred
    
    def evaluate(self, X, y_true):
        """
        Evaluate the model performance.
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
            
        # Make predictions
        y_pred = self.predict(X)
        
        # Filter out NaN values in y_true for evaluation
        mask = ~np.isnan(y_true)
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]
        
        # Calculate metrics
        mse = mean_squared_error(y_true_filtered, y_pred_filtered)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_filtered, y_pred_filtered)
        r2 = r2_score(y_true_filtered, y_pred_filtered)
        
        # Calculate information coefficient (correlation)
        ic = np.corrcoef(y_true_filtered, y_pred_filtered)[0, 1]
        
        # Calculate directional accuracy
        correct_direction = np.sum((y_true_filtered > 0) == (y_pred_filtered > 0)) / len(y_true_filtered)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'ic': ic,
            'directional_accuracy': correct_direction
        }


class StockAwareKernelRegression:
    def __init__(self, bandwidth='auto', kernel='gaussian', n_neighbors=100, n_jobs=-1):
        """
        Initialize the Stock-Aware Kernel Regression model with optimizations.
        This model applies kernel regression separately for each stock.
        """
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
        self.stock_models = {}
        self._is_fitted = False
        self.global_model = None  # Fallback model for stocks without enough data
        self.n_features = None
    
    def fit(self, X_train, y_train, stock_ids, feature_names=None):
        """
        Fit separate kernel regression models for each stock with NaN handling.
        Uses optimized implementation for speed.
        """
        print(f"Fitting StockAwareKernelRegression model...")
        start_time = np.datetime64('now')
        
        # Store feature dimensionality
        self.n_features = X_train.shape[1]
        print(f"Training with {self.n_features} features")
        
        # Convert to numpy arrays if not already
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        stock_ids = np.asarray(stock_ids)
        
        # Filter out rows with NaN in y_train
        y_nan_mask = np.isnan(y_train)
        if np.any(y_nan_mask):
            X_train = X_train[~y_nan_mask]
            stock_ids = stock_ids[~y_nan_mask]
            y_train = y_train[~y_nan_mask]
            print(f"Dropped {np.sum(y_nan_mask)} rows with NaN targets")
        
        # Get unique stock IDs
        unique_stocks = np.unique(stock_ids)
        print(f"Fitting models for {len(unique_stocks)} unique stocks")
        
        # First, fit a global model as fallback
        print("Fitting global model as fallback...")
        self.global_model = KernelRegression(
            bandwidth=self.bandwidth, 
            kernel=self.kernel,
            n_neighbors=self.n_neighbors,
            n_jobs=self.n_jobs
        )
        self.global_model.fit(X_train, y_train, feature_names)
        
        # Use parallel processing to fit stock-specific models
        def fit_stock_model(stock_id):
            # Get indices for this stock
            idx = stock_ids == stock_id
            stock_X = X_train[idx]
            stock_y = y_train[idx]
            
            # Check if we have enough data points
            if len(stock_X) >= 100:  # Require at least 100 data points
                # Create and fit a kernel regression model
                model = KernelRegression(
                    bandwidth=self.bandwidth, 
                    kernel=self.kernel,
                    n_neighbors=min(self.n_neighbors, len(stock_X)),
                    n_jobs=1  # Use 1 job since we're already parallelizing at the stock level
                )
                try:
                    model.fit(stock_X, stock_y, feature_names)
                    return stock_id, model, len(stock_X)
                except Exception as e:
                    print(f"Error fitting model for stock {stock_id}: {e}")
                    return stock_id, None, len(stock_X)
            else:
                return stock_id, None, len(stock_X)
        
        # Use joblib for parallel processing if n_jobs > 1
        if self.n_jobs != 1:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(fit_stock_model)(stock_id) for stock_id in unique_stocks
            )
        else:
            results = [fit_stock_model(stock_id) for stock_id in unique_stocks]
        
        # Process results
        success_count = 0
        for stock_id, model, data_count in results:
            if model is not None:
                self.stock_models[stock_id] = model
                success_count += 1
        
        print(f"Successfully fitted models for {success_count} out of {len(unique_stocks)} stocks")
        self._is_fitted = True
        
        end_time = np.datetime64('now')
        elapsed = (end_time - start_time) / np.timedelta64(1, 's')
        print(f"StockAwareKernelRegression training completed in {elapsed:.2f} seconds")
        return self
    
    def _ensure_feature_consistency(self, X):
        """
        Ensures the feature dimensions match what the model was trained on.
        If there's a mismatch, handles it by padding or truncating.
        
        Args:
            X: Input features
            
        Returns:
            X with consistent dimensionality
        """
        if X.shape[1] != self.n_features:
            print(f"WARNING: Feature dimension mismatch! Got {X.shape[1]}, expected {self.n_features}")
            
            if X.shape[1] < self.n_features:
                # Need to add padding columns
                padding_width = self.n_features - X.shape[1]
                print(f"Adding {padding_width} padding columns with zeros")
                padding = np.zeros((X.shape[0], padding_width))
                X_padded = np.hstack((X, padding))
                return X_padded
            else:
                # Need to truncate columns
                print(f"Truncating to first {self.n_features} features")
                return X[:, :self.n_features]
        return X
    
    def predict(self, X, stock_ids):
        """
        Make predictions using the stock-specific kernel regression models.
        Uses the global model as fallback.
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        print(f"Predicting with StockAwareKernelRegression...")
        start_time = np.datetime64('now')
            
        # Convert to numpy arrays if not already
        X = np.asarray(X)
        stock_ids = np.asarray(stock_ids)
        
        # Check feature dimensionality and adjust if needed
        if X.shape[1] != self.n_features:
            X = self._ensure_feature_consistency(X)
        
        # Initialize predictions
        y_pred = np.zeros(X.shape[0])
        
        # Track which samples have predictions
        has_prediction = np.zeros(len(stock_ids), dtype=bool)
        
        # Process each stock with a specific model
        unique_pred_stocks = np.unique(stock_ids)
        for stock_id in unique_pred_stocks:
            if stock_id in self.stock_models:
                # Get indices for this stock
                idx = stock_ids == stock_id
                if np.sum(idx) > 0:
                    # Use the stock-specific model to predict
                    y_pred[idx] = self.stock_models[stock_id].predict(X[idx])
                    has_prediction[idx] = True
        
        # For stocks without a specific model, use the global model
        missing_idx = ~has_prediction
        if np.sum(missing_idx) > 0:
            y_pred[missing_idx] = self.global_model.predict(X[missing_idx])
            print(f"Used global model for {np.sum(missing_idx)} samples without stock-specific models")
        
        end_time = np.datetime64('now')
        elapsed = (end_time - start_time) / np.timedelta64(1, 's')
        print(f"StockAwareKernelRegression prediction completed in {elapsed:.2f} seconds for {len(stock_ids)} samples")
        
        return y_pred
    
    def evaluate(self, X, y_true, stock_ids):
        """
        Evaluate the model performance.
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
            
        # Make predictions
        y_pred = self.predict(X, stock_ids)
        
        # Filter out NaN values in y_true for evaluation
        mask = ~np.isnan(y_true)
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]
        stock_ids_filtered = stock_ids[mask]
        
        # Calculate metrics
        mse = mean_squared_error(y_true_filtered, y_pred_filtered)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_filtered, y_pred_filtered)
        r2 = r2_score(y_true_filtered, y_pred_filtered)
        
        # Calculate information coefficient (correlation)
        ic = np.corrcoef(y_true_filtered, y_pred_filtered)[0, 1]
        
        # Calculate directional accuracy
        correct_direction = np.sum((y_true_filtered > 0) == (y_pred_filtered > 0)) / len(y_true_filtered)
        
        # Calculate metrics by stock
        stock_metrics = {}
        for stock_id in np.unique(stock_ids_filtered):
            idx = stock_ids_filtered == stock_id
            if np.sum(idx) > 0:
                stock_mse = mean_squared_error(y_true_filtered[idx], y_pred_filtered[idx])
                stock_rmse = np.sqrt(stock_mse)
                stock_metrics[stock_id] = {
                    'mse': stock_mse,
                    'rmse': stock_rmse,
                    'count': np.sum(idx)
                }
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'ic': ic,
            'directional_accuracy': correct_direction,
            'stock_metrics': stock_metrics
        }


class StockAwareKernelRegression:
    def __init__(self, bandwidth='auto', kernel='gaussian', n_neighbors=100, n_jobs=-1):
        """
        Initialize the Stock-Aware Kernel Regression model with optimizations.
        This model applies kernel regression separately for each stock.
        """
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
        self.stock_models = {}
        self._is_fitted = False
        self.global_model = None  # Fallback model for stocks without enough data
    
    def fit(self, X_train, y_train, stock_ids):
        """
        Fit separate kernel regression models for each stock with NaN handling.
        Uses optimized implementation for speed.
        """
        print(f"Fitting StockAwareKernelRegression model...")
        start_time = np.datetime64('now')
        
        # Convert to numpy arrays if not already
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        stock_ids = np.asarray(stock_ids)
        
        # Filter out rows with NaN in y_train
        y_nan_mask = np.isnan(y_train)
        if np.any(y_nan_mask):
            X_train = X_train[~y_nan_mask]
            stock_ids = stock_ids[~y_nan_mask]
            y_train = y_train[~y_nan_mask]
            print(f"Dropped {np.sum(y_nan_mask)} rows with NaN targets")
        
        # Get unique stock IDs
        unique_stocks = np.unique(stock_ids)
        print(f"Fitting models for {len(unique_stocks)} unique stocks")
        
        # First, fit a global model as fallback
        print("Fitting global model as fallback...")
        self.global_model = KernelRegression(
            bandwidth=self.bandwidth, 
            kernel=self.kernel,
            n_neighbors=self.n_neighbors,
            n_jobs=self.n_jobs
        )
        self.global_model.fit(X_train, y_train)
        
        # Use parallel processing to fit stock-specific models
        def fit_stock_model(stock_id):
            # Get indices for this stock
            idx = stock_ids == stock_id
            stock_X = X_train[idx]
            stock_y = y_train[idx]
            
            # Check if we have enough data points
            if len(stock_X) >= 100:  # Require at least 100 data points
                # Create and fit a kernel regression model
                model = KernelRegression(
                    bandwidth=self.bandwidth, 
                    kernel=self.kernel,
                    n_neighbors=min(self.n_neighbors, len(stock_X)),
                    n_jobs=1  # Use 1 job since we're already parallelizing at the stock level
                )
                try:
                    model.fit(stock_X, stock_y)
                    return stock_id, model, len(stock_X)
                except Exception as e:
                    print(f"Error fitting model for stock {stock_id}: {e}")
                    return stock_id, None, len(stock_X)
            else:
                return stock_id, None, len(stock_X)
        
        # Use joblib for parallel processing if n_jobs > 1
        if self.n_jobs != 1:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(fit_stock_model)(stock_id) for stock_id in unique_stocks
            )
        else:
            results = [fit_stock_model(stock_id) for stock_id in unique_stocks]
        
        # Process results
        success_count = 0
        for stock_id, model, data_count in results:
            if model is not None:
                self.stock_models[stock_id] = model
                success_count += 1
        
        print(f"Successfully fitted models for {success_count} out of {len(unique_stocks)} stocks")
        self._is_fitted = True
        
        end_time = np.datetime64('now')
        elapsed = (end_time - start_time) / np.timedelta64(1, 's')
        print(f"StockAwareKernelRegression training completed in {elapsed:.2f} seconds")
        return self
    
    def predict(self, X, stock_ids):
        """
        Make predictions using the stock-specific kernel regression models.
        Uses the global model as fallback.
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        print(f"Predicting with StockAwareKernelRegression...")
        start_time = np.datetime64('now')
            
        # Convert to numpy arrays if not already
        X = np.asarray(X)
        stock_ids = np.asarray(stock_ids)
        
        # Initialize predictions
        y_pred = np.zeros(X.shape[0])
        
        # Track which samples have predictions
        has_prediction = np.zeros(len(stock_ids), dtype=bool)
        
        # Process each stock with a specific model
        unique_pred_stocks = np.unique(stock_ids)
        for stock_id in unique_pred_stocks:
            if stock_id in self.stock_models:
                # Get indices for this stock
                idx = stock_ids == stock_id
                if np.sum(idx) > 0:
                    # Use the stock-specific model to predict
                    y_pred[idx] = self.stock_models[stock_id].predict(X[idx])
                    has_prediction[idx] = True
        
        # For stocks without a specific model, use the global model
        missing_idx = ~has_prediction
        if np.sum(missing_idx) > 0:
            y_pred[missing_idx] = self.global_model.predict(X[missing_idx])
            print(f"Used global model for {np.sum(missing_idx)} samples without stock-specific models")
        
        end_time = np.datetime64('now')
        elapsed = (end_time - start_time) / np.timedelta64(1, 's')
        print(f"StockAwareKernelRegression prediction completed in {elapsed:.2f} seconds for {len(stock_ids)} samples")
        
        return y_pred
    
    def evaluate(self, X, y_true, stock_ids):
        """
        Evaluate the model performance.
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
            
        # Make predictions
        y_pred = self.predict(X, stock_ids)
        
        # Filter out NaN values in y_true for evaluation
        mask = ~np.isnan(y_true)
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]
        stock_ids_filtered = stock_ids[mask]
        
        # Calculate metrics
        mse = mean_squared_error(y_true_filtered, y_pred_filtered)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_filtered, y_pred_filtered)
        r2 = r2_score(y_true_filtered, y_pred_filtered)
        
        # Calculate information coefficient (correlation)
        ic = np.corrcoef(y_true_filtered, y_pred_filtered)[0, 1]
        
        # Calculate directional accuracy
        correct_direction = np.sum((y_true_filtered > 0) == (y_pred_filtered > 0)) / len(y_true_filtered)
        
        # Calculate metrics by stock
        stock_metrics = {}
        for stock_id in np.unique(stock_ids_filtered):
            idx = stock_ids_filtered == stock_id
            if np.sum(idx) > 0:
                stock_mse = mean_squared_error(y_true_filtered[idx], y_pred_filtered[idx])
                stock_rmse = np.sqrt(stock_mse)
                stock_metrics[stock_id] = {
                    'mse': stock_mse,
                    'rmse': stock_rmse,
                    'count': np.sum(idx)
                }
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'ic': ic,
            'directional_accuracy': correct_direction,
            'stock_metrics': stock_metrics
        }


class StockAwareKernelRegression:
    def __init__(self, bandwidth='auto', kernel='gaussian'):
        """
        Initialize the Stock-Aware Kernel Regression model.
        This model applies kernel regression separately for each stock.
        
        Args:
            bandwidth (str or float): Bandwidth for the kernel. If 'auto', it will be selected using cross-validation.
            kernel (str): Kernel type ('gaussian', 'epanechnikov', etc.)
        """
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.stock_models = {}
        self._is_fitted = False
    
    def fit(self, X_train, y_train, stock_ids):
        """
        Fit separate kernel regression models for each stock.
        
        Args:
            X_train (ndarray): Training features
            y_train (ndarray): Training targets
            stock_ids (ndarray): Stock IDs for each sample
            
        Returns:
            self: The fitted model
        """
        # Get unique stock IDs
        unique_stocks = np.unique(stock_ids)
        
        # Fit a separate model for each stock
        for stock_id in unique_stocks:
            # Get indices for this stock
            idx = stock_ids == stock_id
            
            if np.sum(idx) > 20:  # Only fit if we have enough data points
                # Create and fit a kernel regression model
                model = KernelRegression(bandwidth=self.bandwidth, kernel=self.kernel)
                model.fit(X_train[idx], y_train[idx])
                
                # Store the model
                self.stock_models[stock_id] = model
        
        self._is_fitted = True
        return self
    
    def predict(self, X, stock_ids):
        """
        Make predictions using the stock-specific kernel regression models.
        
        Args:
            X (ndarray): Input features
            stock_ids (ndarray): Stock IDs for each sample
            
        Returns:
            ndarray: Predicted values
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
            
        # Initialize predictions
        y_pred = np.zeros(X.shape[0])
        
        # Make predictions for each stock
        for stock_id in self.stock_models:
            # Get indices for this stock
            idx = stock_ids == stock_id
            
            if np.sum(idx) > 0:
                # Use the stock-specific model to predict
                y_pred[idx] = self.stock_models[stock_id].predict(X[idx])
        
        # For stocks without a model, use the average prediction
        missing_idx = ~np.isin(stock_ids, list(self.stock_models.keys()))
        if np.sum(missing_idx) > 0:
            # Use the average of available predictions
            average_pred = np.mean(y_pred[~missing_idx]) if np.sum(~missing_idx) > 0 else 0
            y_pred[missing_idx] = average_pred
        
        return y_pred
    
    def evaluate(self, X, y_true, stock_ids):
        """
        Evaluate the model performance.
        
        Args:
            X (ndarray): Input features
            y_true (ndarray): True target values
            stock_ids (ndarray): Stock IDs for each sample
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
            
        # Make predictions
        y_pred = self.predict(X, stock_ids)
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate information coefficient (correlation)
        ic = np.corrcoef(y_true, y_pred)[0, 1]
        
        # Calculate directional accuracy
        correct_direction = np.sum((y_true > 0) == (y_pred > 0)) / len(y_true)
        
        # Calculate metrics by stock
        stock_metrics = {}
        for stock_id in np.unique(stock_ids):
            idx = stock_ids == stock_id
            if np.sum(idx) > 0:
                stock_mse = mean_squared_error(y_true[idx], y_pred[idx])
                stock_rmse = np.sqrt(stock_mse)
                stock_metrics[stock_id] = {
                    'mse': stock_mse,
                    'rmse': stock_rmse,
                    'count': np.sum(idx)
                }
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'ic': ic,
            'directional_accuracy': correct_direction,
            'stock_metrics': stock_metrics
        }