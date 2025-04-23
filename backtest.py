import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import time
from datetime import datetime
import torch
from torch.utils.data import DataLoader, TensorDataset
import gc
import os

class PyTorchBacktestFramework:
    def __init__(self, train_test_dict, model_type, output_dir, window_size=20, rebalance_freq=5, stock_count = None):
        """
        Initialize the PyTorch-specific backtest framework.
        
        Args:
            train_test_dict (dict): Dictionary containing training and testing data
            model_type (str): Type of model - 'transformer' or 'regression'
            window_size (int): Lookback window size for time series
            rebalance_freq (int): Rebalance frequency in days
        """
        self.train_test_dict = train_test_dict
        self.window_size = window_size
        self.rebalance_freq = rebalance_freq
        self.output_dir = output_dir
        self.model_type = model_type
        self.stock_count = stock_count if stock_count is not None else np.max(train_test_dict['train_si']) + 1
        
    def prepare_data_for_day(self, day_idx, is_training=True):
        """
        Prepare data for a specific day.
        
        Args:
            day_idx (int): Day index
            is_training (bool): Whether to use training or testing data
            
        Returns:
            tuple: (features, stock_ids, previous_returns)
        """
        # Choose the appropriate dataset
        prefix = 'train_' if is_training else 'test_'
        
        # Get indices for this day
        day_mask = self.train_test_dict[f'{prefix}di'] == day_idx
        
        if np.sum(day_mask) == 0:
            return None, None, None
            
        # Get features and stock IDs for this day
        features = self.train_test_dict[f'{prefix}features'][day_mask]
        stock_ids = self.train_test_dict[f'{prefix}si'][day_mask]
        returns = self.train_test_dict[f'{prefix}y'][day_mask]
        
        return features, stock_ids, returns
    
    def prepare_time_series_batch(self, stock_ids, day_indices, features_map, window_size):
        """
        Efficiently prepare time series data for the transformer model in batches.
        Optimized for PyTorch with reduced memory usage.
        
        Args:
            stock_ids (list): List of stock IDs to prepare data for
            day_indices (list): Current day indices
            features_map (dict): Mapping of (day_idx, stock_id) to features
            window_size (int): Lookback window size
        
        Returns:
            tuple: (sequence_features, stock_id_list)
        """
        valid_sequences = []
        valid_stock_ids = []
        
        for stock_id in stock_ids:
            # For each stock, try to build a valid sequence
            sequence = []
            
            # Look back from current day
            for offset in range(window_size-1, -1, -1):
                # Calculate the previous day index
                prev_day_idx = day_indices - offset - 1
                
                # Check if we have data for this day and stock
                if (prev_day_idx, stock_id) in features_map:
                    sequence.append(features_map[(prev_day_idx, stock_id)])
                else:
                    # No data for this day/stock, can't build a valid sequence
                    break
            
            # Only add if we have a full sequence
            if len(sequence) == window_size:
                valid_sequences.append(np.stack(sequence))
                valid_stock_ids.append(stock_id)
        
        # Convert to tensor if we have valid sequences and free memory
        if valid_sequences:
            sequence_features = torch.tensor(np.stack(valid_sequences), dtype=torch.float32)
            # Free memory
            del valid_sequences
            gc.collect()
            return sequence_features, valid_stock_ids
        else:
            return None, []

    def _prepare_time_series_data_batched(self, stock_ids, day_indices, features, returns, window_size, batch_size=100):
        """
        Optimized implementation to prepare time series data efficiently.
        Processes all data at once without chunking for better performance.
        
        Args:
            stock_ids (ndarray): Stock IDs
            day_indices (ndarray): Day indices
            features (ndarray): Feature matrix
            returns (ndarray): Returns
            window_size (int): Lookback window size
            
        Returns:
            tuple: (X, y) where X is the input sequences and y is the target values
        """
        # Convert inputs to numpy arrays if they aren't already
        stock_ids = np.asarray(stock_ids)
        day_indices = np.asarray(day_indices)
        features = np.asarray(features)
        returns = np.asarray(returns)
        
        # Create dataframe for organization - this doesn't copy the data
        df = pd.DataFrame({
            'stock_id': stock_ids,
            'day_idx': day_indices,
            'return': returns,
            'feature_idx': np.arange(len(stock_ids))  # Index to access features
        })
        
        # Sort by stock_id and day_idx (this creates a view, not a full copy)
        df = df.sort_values(['stock_id', 'day_idx'])

        for stock_id, stock_df in df.groupby('stock_id'):
            L = len(stock_df)
            break

        n_batch = (L - window_size + batch_size - 1) // batch_size

        for batch_idx in range(n_batch):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(stock_df) - window_size - 1)
            print('Now training', batch_start, batch_end)
        
            # Create lists to store sequences and targets
            sequences = []
            targets = []
            
            # Process each stock in one pass
            for stock_id, stock_df in df.groupby('stock_id'):
                # print(stock_id, sequences, len(stock_df), window_size)
                if len(stock_df) <= window_size:
                    continue

                batch_end = min(batch_end, len(stock_df) - window_size)
                
                # Sort by day (this creates a view, not a copy)
                stock_df = stock_df.sort_values('day_idx').reset_index(drop=True)
                
                # Get all feature indices for this stock
                feature_indices = stock_df['feature_idx'].values
                
                # Efficiently create sequences using vectorized operations
                # This creates a sliding window over the feature indices array
                for i in range(batch_start, batch_end):
                    # Extract window indices and target index
                    window_indices = feature_indices[i:i+window_size]
                    target_idx = feature_indices[i+window_size]
                    
                    # Skip invalid indices
                    if np.any(window_indices >= len(features)) or target_idx >= len(returns):
                        continue
                    
                    # Add sequence and target to lists
                    sequence = features[window_indices]
                    target = returns[target_idx]
                    
                    # Only add valid sequences
                    if not np.isnan(target) and not np.any(np.isnan(sequence)):
                        sequences.append(sequence)
                        targets.append(target)
            
            # Convert to arrays all at once (single memory allocation)
            if sequences:
                X = np.array(sequences)
                y = np.array(targets)
                yield X, y
            else:
                yield np.array([]), np.array([])

    def run_backtest(self, model, optimizer, start_day_idx, end_day_idx, retrain_freq=20, disable_retraining=False):
        """
        Run a backtest using PyTorch implementation optimized for TimeSeriesTransformer and regression models.
        Uses helper functions for training data preparation and GPU optimization.
        
        Args:
            model: PyTorch model (TimeSeriesTransformer or Regression model)
            optimizer: PortfolioOptimizer instance
            start_day_idx (int): Starting day index
            end_day_idx (int): Ending day index
            retrain_freq (int): Model retraining frequency in days
            disable_retraining (bool): Whether to disable model retraining
            
        Returns:
            tuple: (portfolio_values, weights_history, metrics_history, directional_accuracy)
        """
        # Initialize results
        portfolio_values = [1.0]  # Start with $1
        weights_history = []
        metrics_history = []

        # Track current portfolio holdings and model training state
        current_weights = None
        last_rebalance_day = start_day_idx - 1
        last_retrain_day = start_day_idx - 1
        model_has_been_trained = False

        # Timing metrics
        initial_training_time = 0
        prediction_times = []
        retraining_times = []
        optimization_times = []
        
        # Directional accuracy tracking
        correct_direction_count = 0
        total_prediction_count = 0

        # Unique days in the test set
        test_days = sorted(np.unique(self.train_test_dict['test_di']))
        test_days = [d for d in test_days if start_day_idx <= d <= end_day_idx]

        # Dictionary to store stock data by day - only keep what's needed
        stock_data_by_day = {}
        features_map = {}  # Map of (day_idx, stock_id) to features

        self.test_date = test_days[0]

        # Pre-load stock data for test period
        print("Preloading stock data...")
        for day_idx in test_days:
            features, stock_ids, returns = self.prepare_data_for_day(day_idx, is_training=False)
            if features is not None:
                stock_data_by_day[day_idx] = (features, stock_ids, returns)
                # Store in the features map for efficient sequence building
                for i, stock_id in enumerate(stock_ids):
                    features_map[(day_idx, stock_id)] = features[i]

        # Initialize and train model before starting backtest if not disabled
        if not disable_retraining:
            print("Performing initial model training...")
            
            # Add timing for initial training
            initial_training_start = time.time()
            
            # Define training window
            train_window_start = np.min(self.train_test_dict['train_di'])
            train_window_end = train_window_start + self.window_size
            # train_window_start = max(start_day_idx - 252, )
            # train_window_end = start_day_idx
            
<<<<<<< HEAD
            while train_window_end < self.test_date:
                # Use helper function to collect and prepare training data for transformer
                model_has_been_trained, model = self.collect_and_prepare_training_data(
                    model,
                    self.model_type,
                    train_window_start, 
                    train_window_end, 
                    features_map, 
                    self.window_size, 
                    is_training=True,
                    device=model.device if hasattr(model, 'device') else 'cuda'
                )
=======
            # while train_window_end < self.test_date:

            model_has_been_trained, model = self.collect_and_prepare_training_data(
                model, 
                self.model_type, 
                train_window_start, 
                self.test_date, 
                features_map, 
                self.window_size, 
                is_training=True,
                device=model.device if hasattr(model, 'device') else 'cuda'
            )
>>>>>>> master

                # if train_window_start % 10 == 0:
                #     print(f"Trained until {train_window_start} to {train_window_end}")

                # train_window_start += 1
                # train_window_end += 1
            
            # Calculate initial training time
            initial_training_time = time.time() - initial_training_start
            # Convert to minutes and seconds
            init_train_mins = int(initial_training_time // 60)
            init_train_secs = int(initial_training_time % 60)
            print(f"Initial training completed in {init_train_mins} min {init_train_secs} sec")

        # Run the backtest
        for i, day_idx in enumerate(test_days):
            print(f"Backtesting day {i+1}/{len(test_days)} (Day index: {day_idx})")
            
            # Check if we need to retrain the model
            if not disable_retraining and day_idx - last_retrain_day >= retrain_freq:
                print(f"Retraining model on day {day_idx}...")
                
                # Add timing for periodic retraining
                retrain_start = time.time()
                
                # Define retraining window
                lim = day_idx - self.window_size
                train_window_start = last_retrain_day - self.window_size

                # while train_window_start < lim:
                retrain_success, model = self.collect_and_prepare_training_data(
                    model, 
                    self.model_type, 
                    train_window_start, 
                    day_idx, 
                    features_map, 
                    self.window_size, 
                    is_training=False,
                    device=model.device if hasattr(model, 'device') else 'cuda'
                )

                    # if train_window_start % 10 == 0:
                    #     print(f"Trained until {train_window_start} to {train_window_end}")
                    
                    # train_window_start += 1
                
                # Calculate retraining time
                retrain_time = time.time() - retrain_start
                retraining_times.append(retrain_time)
                retrain_mins = int(retrain_time // 60)
                retrain_secs = int(retrain_time % 60)
                print(f"Retraining completed in {retrain_mins} min {retrain_secs} sec")
                
                if retrain_success:
                    model_has_been_trained = True
                    
                    # Save model checkpoint (optional)
                    try:
                        if hasattr(model, 'save_model'):
                            checkpoint_path = f'{self.model_type}_checkpoint_day_{day_idx}.pt'
                            checkpoint_path = os.path.join(self.output_dir, checkpoint_path)
                            model.save_model(checkpoint_path)
                            print(f"Saved model checkpoint to {checkpoint_path}")
                    except Exception as e:
                        print(f"Failed to save checkpoint: {e}")
                
                last_retrain_day = day_idx
            
            # Get data for current day and skip if not available
            if day_idx not in stock_data_by_day:
                continue
                
            current_features, current_stock_ids, actual_returns = stock_data_by_day[day_idx]
            
            # Check if we need to rebalance the portfolio
            if day_idx - last_rebalance_day >= self.rebalance_freq:
                print(f"Rebalancing portfolio on day {day_idx}...")
                
                # Get predictions for current day using the appropriate method
                predicted_returns = np.zeros(len(current_stock_ids))
                
                # Add timing for prediction
                predict_start = time.time()
                
                # Use appropriate prediction method based on model type
                if self.model_type == 'transformer':
                    # Use prediction helper function for transformer
                    predicted_returns = self.predict_with_efficient_gpu(
                        current_stock_ids,
                        day_idx,
                        features_map,
                        model,
                        self.window_size, 
                        device=model.device if hasattr(model, 'device') else 'cuda'
                    )
                else:
                    # Use regression prediction method
                    predicted_returns = self.predict_with_regression(
                        model,
                        current_features,
                        current_stock_ids
                    )
                print(type(model))
                
                # Calculate prediction time
                predict_time = time.time() - predict_start
                prediction_times.append(predict_time)
                print(f"Prediction completed in {int(predict_time)} sec")
                
                # Calculate directional accuracy - compare signs of predicted vs actual returns
                for i, stock_id in enumerate(current_stock_ids):
                    predicted = predicted_returns[i]
                    actual = actual_returns[i]
                    
                    # Skip if either is zero (no direction)
                    if predicted != 0 and actual != 0:
                        total_prediction_count += 1
                        # Check if both have the same sign (both positive or both negative)
                        if (predicted > 0 and actual > 0) or (predicted < 0 and actual < 0):
                            correct_direction_count += 1
                
                # Print current directional accuracy
                if total_prediction_count > 0:
                    current_accuracy = correct_direction_count / total_prediction_count * 100
                    print(f"Current directional accuracy: {current_accuracy:.2f}% ({correct_direction_count}/{total_prediction_count})")
                
                # Add timing for portfolio optimization
                optimize_start = time.time()
                
                # Estimate covariance matrix using vectorized operations
                unique_current_stocks = np.unique(current_stock_ids)
                n_stocks = len(unique_current_stocks)
                max_history_length = min(60, day_idx - test_days[0])
                
                # Create stock ID lookup array for quick vectorized mapping
                max_stock_id = np.max(unique_current_stocks)
                stock_row_lookup = np.zeros(self.stock_count, dtype=int) - 1
                stock_row_lookup[unique_current_stocks] = np.arange(n_stocks)
                
                # Pre-allocate historical returns matrix
                historical_returns = np.zeros((n_stocks, max_history_length))
                data_present = np.zeros((n_stocks, max_history_length), dtype=bool)
                
                # Process all historical days at once
                for col_idx, hist_day in enumerate(range(max(day_idx - 60, test_days[0]), day_idx)):
                    if hist_day in stock_data_by_day:
                        hist_stock_ids, hist_returns = stock_data_by_day[hist_day][1:3]
                        
                        # Find stocks in our universe
                        valid_indices = np.where((hist_stock_ids <= max_stock_id) & 
                                              (stock_row_lookup[hist_stock_ids] >= 0))[0]
                        
                        if len(valid_indices) > 0:
                            # Extract valid returns and indices
                            valid_returns = hist_returns[valid_indices]
                            valid_stock_ids = hist_stock_ids[valid_indices]
                            row_indices = stock_row_lookup[valid_stock_ids]
                            
                            # Set returns in matrix
                            historical_returns[row_indices, col_idx] = valid_returns
                            data_present[row_indices, col_idx] = True
                
                # Fill missing data with random noise
                missing_data_mask = ~data_present
                historical_returns[missing_data_mask] = np.random.normal(0, 0.001, np.sum(missing_data_mask))
                
                # Calculate covariance matrix
                cov_matrix = np.cov(historical_returns)
                
                # Free memory
                del historical_returns, data_present, missing_data_mask
                gc.collect()
                
                # Ensure covariance matrix is proper
                cov_matrix = (cov_matrix + cov_matrix.T) / 2
                min_eig = np.min(np.real(np.linalg.eigvals(cov_matrix)))
                if min_eig < 0:
                    cov_matrix -= 1.1 * min_eig * np.eye(len(cov_matrix))
                
                # Map predicted returns to match covariance matrix order
                current_expected_returns = np.zeros(n_stocks)
                
                # Handle potential duplicate stock IDs
                prediction_lookup = np.zeros(max_stock_id + 1)
                counts = np.bincount(current_stock_ids, minlength=max_stock_id + 1)
                sums = np.bincount(current_stock_ids, weights=predicted_returns, minlength=max_stock_id + 1)
                
                # Avoid division by zero
                mask = counts > 0
                prediction_lookup[mask] = sums[mask] / counts[mask]
                
                # Get returns in right order
                current_expected_returns = prediction_lookup[unique_current_stocks]
                
                # Free memory
                del prediction_lookup, counts, sums, mask
                gc.collect()
                
                # Calculate market volatility and adjust risk aversion
                market_vol = 0.02  # Default value
                if 'historical_returns' in locals() and historical_returns.shape[1] > 1:
                    market_vol = np.std(np.mean(historical_returns, axis=0))
                
                risk_aversion = 1.0 + 10.0 * market_vol
                optimizer.risk_aversion = risk_aversion
                
                # Optimize portfolio weights - call appropriate method based on optimizer interface
                if hasattr(optimizer, 'mean_variance_optimization'):
                    optimization_result = optimizer.mean_variance_optimization(
                        current_expected_returns,
                        cov_matrix
                    )
                    optimal_weights = optimization_result['weights']
                elif hasattr(optimizer, 'optimize_large_portfolio'):
                    optimization_result = optimizer.optimize_large_portfolio(
                        current_expected_returns,
                        cov_matrix
                    )
                    optimal_weights = optimization_result['weights']
                else:
                    # Default optimization method - fallback
                    optimization_result = optimizer.optimize(
                        current_expected_returns,
                        cov_matrix
                    )
                    optimal_weights = optimization_result['weights']
                
                # Free optimization memory
                del cov_matrix
                gc.collect()
                
                # Convert optimization results
                current_weights = dict(zip(unique_current_stocks, optimal_weights))
                
                # Calculate portfolio metrics
                print("Computing portfolio metrics...")
                # Account for different compute_portfolio_metrics interfaces
                if hasattr(optimizer, 'compute_portfolio_metrics'):
                    portfolio_metrics = optimizer.compute_portfolio_metrics(
                        optimal_weights,
                        current_expected_returns,
                        np.eye(len(optimal_weights))  # Use identity matrix to avoid copying cov_matrix
                    )
                else:
                    # Basic metrics if no compute_portfolio_metrics method available
                    portfolio_metrics = {
                        'expected_return': np.sum(optimal_weights * current_expected_returns),
                        'volatility': np.sqrt(np.sum(optimal_weights ** 2))  # Simplified using identity matrix
                    }
                
                # Calculate optimization time
                optimize_time = time.time() - optimize_start
                optimization_times.append(optimize_time)
                optimize_mins = int(optimize_time // 60)
                optimize_secs = int(optimize_time % 60)
                print(f"Portfolio optimization completed in {optimize_mins} min {optimize_secs} sec")
                
                # Store metrics and weights
                metrics_history.append({
                    'day_idx': day_idx,
                    'metrics': portfolio_metrics,
                    'risk_aversion': risk_aversion,
                    'market_volatility': market_vol
                })
                
                weights_history.append({
                    'day_idx': day_idx,
                    'weights': current_weights.copy()  # Make copy to ensure we keep history
                })
                
                # Free memory
                del current_expected_returns, optimal_weights
                gc.collect()
                
                last_rebalance_day = day_idx
            
            # Calculate portfolio return for current day
            if current_weights is not None:
                day_return = 0.0
                invested_weight = 0.0
                
                for stock_id, weight in current_weights.items():
                    # Find actual return for this stock
                    stock_mask = current_stock_ids == stock_id
                    if np.sum(stock_mask) > 0:
                        stock_return = actual_returns[stock_mask][0]
                        day_return += weight * stock_return
                        invested_weight += weight
                
                # Adjust for cash
                if invested_weight < 1.0:
                    cash_weight = 1.0 - invested_weight
                    day_return += cash_weight * 0.0  # Zero cash return
                
                # Update portfolio value
                current_value = portfolio_values[-1] * (1 + day_return)
                portfolio_values.append(current_value)
            else:
                # No weights yet
                portfolio_values.append(portfolio_values[-1])
            
            # Force garbage collection
            gc.collect()
        
        # Print timing summary
        print("\nTiming Performance Summary:")
        print(f"Initial Training: {int(initial_training_time // 60)} min {int(initial_training_time % 60)} sec")
        
        if prediction_times:
            avg_prediction_time = sum(prediction_times) / len(prediction_times)
            print(f"Prediction (per day): {int(avg_prediction_time)} sec")
        
        if retraining_times:
            avg_retraining_time = sum(retraining_times) / len(retraining_times)
            retrain_mins = int(avg_retraining_time // 60)
            retrain_secs = int(avg_retraining_time % 60)
            print(f"Periodic Retraining: {retrain_mins} min {retrain_secs} sec")
        
        if optimization_times:
            avg_optimization_time = sum(optimization_times) / len(optimization_times)
            optimize_mins = int(avg_optimization_time // 60)
            optimize_secs = int(avg_optimization_time % 60)
            print(f"Portfolio Optimization: {optimize_mins} min {optimize_secs} sec")
        
        # Print final directional accuracy
        if total_prediction_count > 0:
            final_accuracy = correct_direction_count / total_prediction_count * 100
            print(f"\nModel Performance Summary:")
            print(f"Directional Accuracy: {final_accuracy:.2f}% ({correct_direction_count}/{total_prediction_count})")
        
        # Calculate final directional accuracy percentage
        final_accuracy = None
        if total_prediction_count > 0:
            final_accuracy = correct_direction_count / total_prediction_count * 100
            
        # Plot results with directional accuracy
        self.plot_backtest_results(portfolio_values, weights_history, metrics_history, 
                                  model_name=f"{self.model_type}_{type(model).__name__}", 
                                  directional_accuracy=final_accuracy)
        
        return portfolio_values, weights_history, metrics_history, final_accuracy

    def collect_and_prepare_training_data(self, model, model_type, start_day, end_day, features_map, window_size, is_training=False, device='cuda'):
        """
        Efficient function to collect and prepare training data with GPU optimization.
        Processes all days at once instead of in chunks.
        
        Args:
            model: Model to be trained
            start_day (int): Start day index for training data
            end_day (int): End day index for training data
            features_map (dict): Map of (day_idx, stock_id) to features (will be updated)
            window_size (int): Window size for time series data
            device (str): Device to use ('cuda' or 'cpu')
        
        Returns:
            tuple: (trained_model_flag, model) - Flag if model was trained, and the model
        """
        import contextlib
        
        key = 'train_di' if is_training else 'test_di'
        # Get all training days in the specified window
        # train_days = sorted([d for d in np.unique(self.train_test_dict[key]) 
        #                     if start_day <= d < end_day])
        train_days = [i for i in range(start_day, end_day)]
        
        if len(train_days) == 0:
            print("No training days found in the specified window.")
            return False, model
        
        # Collect all training data at once
        all_features = []
        all_stock_ids = []
        all_day_indices = []
        all_returns = []
        
        # Process all days at once
        for train_day in train_days:
            is_training = train_day < self.test_date
            
            features, stock_ids, returns = self.prepare_data_for_day(train_day, is_training=is_training)
            
            if features is not None and len(features) > 0:
                # Store data references
                all_features.append(features)
                all_stock_ids.append(stock_ids)
                all_day_indices.append(np.ones_like(stock_ids) * train_day)
                all_returns.append(returns)
                
                # Update features map without creating extra copies
                for i, stock_id in enumerate(stock_ids):
                    features_map[(train_day, stock_id)] = features[i]
        
        # Only concatenate if we have data
        if not all_features:
            print("No training data found.")
            return False, model
        
        # Concatenate all data (one-time operation)
        train_features = np.vstack(all_features)
        train_stock_ids = np.concatenate(all_stock_ids)
        train_day_indices = np.concatenate(all_day_indices)
        train_returns = np.concatenate(all_returns)
        
        # Free memory
        del all_features, all_stock_ids, all_day_indices, all_returns
        gc.collect()

        if model_type == 'stock_kernel':
            model.fit(train_features, train_returns, train_stock_ids)
        elif model_type == 'kernel':
            model.fit(train_features, train_returns)
        else:
            # Prepare time series data (single pass)
            for X_train, y_train in self._prepare_time_series_data_batched(
                train_stock_ids, train_day_indices, train_features, train_returns, window_size
            ):
                # Check if we have valid sequences
                if len(X_train) == 0:
                    print("No valid sequences found for training.")
                    return False, model
                
                # Sample if dataset is too large
                if len(X_train) > 10000:
                    print(f"Sampling {10000} sequences from {len(X_train)} for faster training...")
                    sample_indices = np.random.choice(len(X_train), size=10000, replace=False)
                    X_sample = X_train[sample_indices]
                    y_sample = y_train[sample_indices]
                    
                    # Free original data
                    del X_train, y_train
                    gc.collect()
                    
                    # Update training data
                    X_train = X_sample
                    y_train = y_sample
                    
                    # Free sample arrays
                    del X_sample, y_sample
                    gc.collect()
                
                print(f"Training with {len(X_train)} sequences...")
                
                # Build model if needed
                if model.model is None:
                    model.build_model((X_train.shape[1], X_train.shape[2]))
                
                # Check if CUDA is available
                if device == 'cuda' and not torch.cuda.is_available():
                    print("CUDA requested but not available, falling back to CPU.")
                    device = 'cpu'
                
                # Move model to device
                if hasattr(model, 'model'):
                    model.model.to(device)
                
                # Try GPU training if possible
                try:
                    if hasattr(model, 'fit_with_dataloader') and device == 'cuda':
                        # Move data to GPU in a single operation
                        X_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
                        y_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)
                        
                        # Create dataset and dataloader
                        train_dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
                        train_loader = torch.utils.data.DataLoader(
                            train_dataset, batch_size=64, shuffle=True, pin_memory=False
                        )
                        
                        # Train model directly with GPU data
                        model.fit_with_dataloader(train_loader, epochs=50, verbose=1)
                        
                        # Clean up GPU memory
                        del X_tensor, y_tensor, train_dataset, train_loader
                        if device == 'cuda':
                            torch.cuda.empty_cache()
                    else:
                        # Fall back to standard training
                        model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=1)
                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        print(f"GPU memory error: {e}")
                        print("Falling back to CPU training...")
                        
                        # Move model to CPU
                        if hasattr(model, 'model'):
                            model.model.to('cpu')
                        
                        if device == 'cuda':
                            torch.cuda.empty_cache()
                        
                        # Train on CPU
                        model.fit(X_train, y_train, epochs=25, batch_size=64, verbose=1)
                    else:
                        raise e
                
                # Free training data memory
                del X_train, y_train
                gc.collect()
            
            return True, model

    def predict_with_efficient_gpu(self, current_stock_ids, day_idx, features_map, model, window_size, device='cuda'):
        """
        Memory-efficient prediction function for transformer models that operates on GPU with minimal copies.
        
        Args:
            current_stock_ids (array-like): Array of stock IDs for current day
            day_idx (int): Current day index
            features_map (dict): Map of (day_idx, stock_id) to features
            model: PyTorch model to use for prediction
            window_size (int): Window size for time series
            device (str): Device to use ('cuda' or 'cpu')
            
        Returns:
            ndarray: Array of predicted returns for each stock ID
        """
        import contextlib
        
        # Check if CUDA is available if device is 'cuda'
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA requested but not available, falling back to CPU")
            device = 'cpu'
        
        # Initialize predictions array
        predicted_returns = np.zeros(len(current_stock_ids))
        
        # Skip if model hasn't been trained
        if not hasattr(model, 'model') or model.model is None:
            print("Warning: Model has not been built yet. Returning zero predictions.")
            return predicted_returns
        
        # Create sequence features using existing method
        sequence_features, valid_stock_ids = self.prepare_time_series_batch(
            current_stock_ids, day_idx, features_map, window_size
        )
        
        # Make predictions if we have valid sequences
        if sequence_features is not None and len(valid_stock_ids) > 0:
            # Move model to device if needed
            model.model.to(device)
            
            # Make predictions using no_grad context
            with torch.no_grad():
                try:
                    # Move data to device
                    sequence_features = sequence_features.to(device)
                    
                    # Get predictions
                    batch_predictions = model.model(sequence_features).cpu().numpy()
                    
                    # Free GPU memory
                    sequence_features = sequence_features.cpu()
                    del sequence_features
                    if device == 'cuda':
                        torch.cuda.empty_cache()
                    
                    # Map predictions to stock IDs
                    stock_predictions = dict(zip(valid_stock_ids, batch_predictions))
                    
                    # Fill prediction array
                    for i, stock_id in enumerate(current_stock_ids):
                        if stock_id in stock_predictions:
                            predicted_returns[i] = stock_predictions[stock_id]
                    
                    # Free memory
                    del stock_predictions, valid_stock_ids
                    gc.collect()
                    
                except Exception as e:
                    print(f"Error during prediction: {e}")
                    # Keep zero predictions on error
        else:
            print("Warning: No valid sequences found for prediction.")
        
        return predicted_returns
    
    def predict_with_regression(self, model, features, stock_ids=None):
        """
        Prediction function for regression models.
        
        Args:
            model: Regression model
            features: Feature matrix
            stock_ids: Stock IDs (for stock-aware models)
            
        Returns:
            ndarray: Array of predicted returns
        """
        try:
            # Check if it's a stock-aware model
            if self.model_type == 'stock_kernel':
                print('here', model, model.predict)
                return model.predict(features, stock_ids)
            else:
                return model.predict(features)
        except Exception as e:
            print(f"Error during regression prediction: {e}")
            return np.zeros(len(features))
    
    def plot_backtest_results(self, portfolio_values, weights_history, metrics_history, model_name, directional_accuracy=None):
        """
        Plot backtest results with enhanced visualizations.
        
        Args:
            portfolio_values (list): Portfolio values over time
            weights_history (list): Portfolio weights history
            metrics_history (list): Portfolio metrics history
            model_name (str): Name of the model used
            directional_accuracy (float): Directional accuracy of the model
        """
        # Set up plot style
        plt.style.use('seaborn-v0_8-darkgrid')
        fig = plt.figure(figsize=(18, 12))
        
        # Plot portfolio value
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(portfolio_values, linewidth=2)
        ax1.set_title(f'Portfolio Value ({model_name})')
        ax1.set_xlabel('Trading Days')
        ax1.set_ylabel('Portfolio Value ($)')
        
        # Calculate and display performance metrics
        returns = np.array(portfolio_values[1:]) / np.array(portfolio_values[:-1]) - 1
        total_return = portfolio_values[-1] / portfolio_values[0] - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        max_drawdown = np.max(np.maximum.accumulate(portfolio_values) - portfolio_values) / np.max(portfolio_values)
        
        metrics_text = (
            f'Total Return: {total_return:.2%}\n'
            f'Annual Return: {annual_return:.2%}\n'
            f'Annual Volatility: {volatility:.2%}\n'
            f'Sharpe Ratio: {sharpe_ratio:.2f}\n'
            f'Max Drawdown: {max_drawdown:.2%}'
        )
        
        # Add directional accuracy if provided
        if directional_accuracy is not None:
            metrics_text += f'\nDirectional Accuracy: {directional_accuracy:.2f}%'
        
        plt.figtext(0.15, 0.01, metrics_text, fontsize=12, ha='left', bbox=dict(facecolor='white', alpha=0.7))
        
        # Plot weight distribution over time (top 10 stocks)
        if weights_history:
            ax2 = plt.subplot(2, 2, 2)
            
            # Get top 10 stocks by average weight
            all_stocks = set()
            for wh in weights_history:
                all_stocks.update(wh['weights'].keys())
                
            avg_weights = {}
            for stock in all_stocks:
                weights = [wh['weights'].get(stock, 0) for wh in weights_history]
                avg_weights[stock] = np.mean(weights)
            
            top_stocks = sorted(avg_weights.keys(), key=lambda x: avg_weights[x], reverse=True)[:10]
            
            # Plot weights for top stocks
            days = [wh['day_idx'] for wh in weights_history]
            
            for stock in top_stocks:
                weights = [wh['weights'].get(stock, 0) for wh in weights_history]
                ax2.plot(days, weights, label=f'Stock {stock}')
            
            ax2.set_title('Portfolio Weights for Top 10 Stocks')
            ax2.set_xlabel('Day Index')
            ax2.set_ylabel('Weight')
            ax2.legend(loc='upper right', bbox_to_anchor=(1.1, 1.0))
            
            # Plot risk aversion and market volatility
            if metrics_history:
                ax3 = plt.subplot(2, 2, 3)
                
                # Extract data from metrics history
                risk_aversions = [m['risk_aversion'] for m in metrics_history]
                market_vols = [m['market_volatility'] for m in metrics_history]
                metric_days = [m['day_idx'] for m in metrics_history]
                
                # Ensure all arrays have the same length
                if len(metric_days) == len(risk_aversions) == len(market_vols):
                    ax3.plot(metric_days, risk_aversions, label='Risk Aversion', linewidth=2)
                    # Scale market volatility for better visualization
                    ax3.plot(metric_days, np.array(market_vols) * 20, label='Market Volatility (scaled)', linewidth=2, linestyle='--')
                    ax3.set_title('Risk Aversion and Market Volatility')
                    ax3.set_xlabel('Day Index')
                    ax3.set_ylabel('Value')
                    ax3.legend()
                else:
                    print(f"Warning: Dimension mismatch in metrics arrays")
            
            # Plot expected return vs. realized return
            if len(metrics_history) > 1:
                ax4 = plt.subplot(2, 2, 4)
                
                try:
                    # Calculate realized returns
                    realized_returns = []
                    for i in range(1, len(portfolio_values)):
                        realized_returns.append(portfolio_values[i] / portfolio_values[i-1] - 1)
                    
                    # Extract expected returns from metrics
                    expected_returns = []
                    for m in metrics_history:
                        if 'expected_return' in m['metrics']:
                            expected_returns.append(m['metrics']['expected_return'])
                        else:
                            # If expected_return not in metrics, use portfolio expected return
                            weights = list(m['weights'].values()) if 'weights' in m else [1.0]
                            expected_returns.append(np.mean(weights))
                            
                    metric_days = [m['day_idx'] for m in metrics_history]
                    
                    # Align the series (expected returns are at rebalance days)
                    aligned_realized = []
                    for i in range(len(metric_days) - 1):
                        start_idx = metric_days[i] - metric_days[0]
                        end_idx = metric_days[i+1] - metric_days[0]
                        
                        # Handle out of bounds indices
                        start_idx = max(0, min(start_idx, len(realized_returns) - 1))
                        end_idx = max(0, min(end_idx, len(realized_returns)))
                        
                        if start_idx < end_idx:
                            period_returns = realized_returns[start_idx:end_idx]
                            aligned_realized.append(np.mean(period_returns) if len(period_returns) > 0 else 0)
                        else:
                            aligned_realized.append(0)
                    
                    # Add the last period if necessary
                    if len(metric_days) > 0:
                        last_start = metric_days[-1] - metric_days[0]
                        last_start = max(0, min(last_start, len(realized_returns) - 1))
                        if last_start < len(realized_returns):
                            last_returns = realized_returns[last_start:]
                            aligned_realized.append(np.mean(last_returns) if len(last_returns) > 0 else 0)
                    
                    # Make sure arrays have the same length for comparison
                    min_length = min(len(expected_returns), len(aligned_realized))
                    if min_length > 1:
                        expected_returns = expected_returns[:min_length]
                        aligned_realized = aligned_realized[:min_length]
                        
                        # Plot comparison (scatter)
                        ax4.scatter(expected_returns, aligned_realized, alpha=0.7)
                        
                        # Add regression line
                        m, b = np.polyfit(expected_returns, aligned_realized, 1)
                        x_range = np.linspace(min(expected_returns), max(expected_returns), 100)
                        ax4.plot(x_range, m * x_range + b, 'r-', linewidth=2)
                        
                        ax4.set_title('Expected Return vs. Realized Return')
                        ax4.set_xlabel('Expected Return')
                        ax4.set_ylabel('Realized Return')
                        
                        # Calculate and display correlation
                        correlation = np.corrcoef(expected_returns, aligned_realized)[0, 1]
                        ax4.text(0.05, 0.95, f'Correlation: {correlation:.2f}', transform=ax4.transAxes,
                                bbox=dict(facecolor='white', alpha=0.7))
                except Exception as e:
                    print(f"Error plotting expected vs realized returns: {e}")
                    # Leave this subplot empty
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 
                    f'backtest_results_{model_name}.png'), dpi=300)
        plt.close(fig)