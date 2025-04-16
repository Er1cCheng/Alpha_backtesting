import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import argparse
import time
from datetime import datetime
import pickle

from feature_engineering import FeatureEngineering
from transformer import TimeSeriesTransformer
from nonparametric_regression import KernelRegression, StockAwareKernelRegression
from portfolio_optimizer import *

class BacktestFramework:
    def __init__(self, train_test_dict, window_size=20, rebalance_freq=5):
        """
        Initialize the backtest framework.
        
        Args:
            train_test_dict (dict): Dictionary containing training and testing data
            window_size (int): Lookback window size for time series
            rebalance_freq (int): Rebalance frequency in days
        """
        self.train_test_dict = train_test_dict
        self.window_size = window_size
        self.rebalance_freq = rebalance_freq
        
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
    
    def run_backtest(self, model, optimizer, start_day_idx, end_day_idx, retrain_freq=20, disable_retraining=False):
        """
        Run a backtest using the specified model and optimizer.
        
        Args:
            model: Either KernelRegression or TimeSeriesTransformer model
            optimizer: PortfolioOptimizer instance
            start_day_idx (int): Starting day index
            end_day_idx (int): Ending day index
            retrain_freq (int): Model retraining frequency in days
            disable_retraining (bool): Whether to disable model retraining
            
        Returns:
            tuple: (portfolio_values, weights_history, metrics_history)
        """
        # Initialize results
        portfolio_values = [1.0]  # Start with $1
        weights_history = []
        metrics_history = []

        # Track current portfolio holdings and model training state
        current_weights = None
        last_rebalance_day = start_day_idx - 1
        last_retrain_day = start_day_idx - 1
        model_has_been_trained = False  # Track if model has been trained at least once

        # Unique days in the test set
        test_days = sorted(np.unique(self.train_test_dict['test_di']))
        test_days = [d for d in test_days if start_day_idx <= d <= end_day_idx]

        # Dictionary to store stock data by day
        stock_data_by_day = {}

        # Pre-load stock data for all days in the test period
        for day_idx in test_days:
            features, stock_ids, returns = self.prepare_data_for_day(day_idx, is_training=False)
            if features is not None:
                stock_data_by_day[day_idx] = (features, stock_ids, returns)

        # Initialize and train model before starting backtest
        print("Performing initial model training...")
        
        # Use a window of historical data for initial training
        train_window_start = max(start_day_idx - 252, np.min(self.train_test_dict['train_di']))
        train_window_end = start_day_idx

        # Get training days in the window
        train_days = sorted([d for d in np.unique(self.train_test_dict['train_di']) 
                            if train_window_start <= d < train_window_end])

        # For TimeSeriesTransformer: Collect all data first, then prepare time series data
        if isinstance(model, TimeSeriesTransformer):
            all_train_features = []
            all_train_stock_ids = []
            all_train_day_indices = []
            all_train_returns = []
            
            for train_day in train_days:
                features, stock_ids, returns = self.prepare_data_for_day(train_day, is_training=True)
                if features is not None and len(features) > 0:
                    all_train_features.append(features)
                    all_train_stock_ids.append(stock_ids)
                    all_train_day_indices.append(np.ones_like(stock_ids) * train_day)
                    all_train_returns.append(returns)
            
            if len(all_train_features) > 0:
                # Concatenate all training data
                train_features = np.vstack(all_train_features)
                train_stock_ids = np.concatenate(all_train_stock_ids)
                train_day_indices = np.concatenate(all_train_day_indices)
                train_returns = np.concatenate(all_train_returns)
                
                # Prepare time series data
                X_train, y_train = model.prepare_time_series_data(
                    train_stock_ids,
                    train_day_indices,
                    train_features,
                    train_returns,
                    window_size=self.window_size
                )
                
                if len(X_train) > 0:
                    # Build model if not already built
                    if model.model is None:
                        model.build_model((X_train.shape[1], X_train.shape[2]))
                    
                    print(f"Initial training of TimeSeriesTransformer model with {len(X_train)} sequences...")
                    model.fit(X_train, y_train, epochs=30, batch_size=64, verbose=1)
                    model_has_been_trained = True
                else:
                    print("Warning: No valid sequences found for initial TimeSeriesTransformer training.")
        else:
            # For KernelRegression models
            train_features = []
            train_returns = []
            train_stock_ids = []
            
            for train_day in train_days:
                features, stock_ids, returns = self.prepare_data_for_day(train_day, is_training=True)
                if features is not None and len(features) > 0:
                    train_features.append(features)
                    train_returns.append(returns)
                    train_stock_ids.append(stock_ids)
            
            if len(train_features) > 0:
                # Concatenate training data
                train_features = np.vstack(train_features)
                train_returns = np.concatenate(train_returns)
                train_stock_ids = np.concatenate(train_stock_ids)
                
                # Train the model based on its type
                if isinstance(model, StockAwareKernelRegression):
                    print(f"Initial training of StockAwareKernelRegression model...")
                    model.fit(train_features, train_returns, train_stock_ids)
                    model_has_been_trained = True
                elif isinstance(model, KernelRegression):
                    print(f"Initial training of KernelRegression model...")
                    model.fit(train_features, train_returns)
                    model_has_been_trained = True
            else:
                print("Warning: No training data available for initial model training.")
        
        # Run the backtest
        for i, day_idx in enumerate(test_days):
            print(f"Backtesting day {i+1}/{len(test_days)} (Day index: {day_idx})")
            
            # Check if we need to retrain the model
            if not disable_retraining and day_idx - last_retrain_day >= retrain_freq:
                print(f"Retraining model on day {day_idx}...")
                
                # Use a window of historical data for training
                train_window_start = max(day_idx - 252, test_days[0])  # Use up to 1 year of data
                
                if isinstance(model, TimeSeriesTransformer):
                    print(f"Preparing to retrain TimeSeriesTransformer model for day {day_idx}...")
                    
                    # For TimeSeriesTransformer: Collect all data for time series preparation
                    all_train_features = []
                    all_train_stock_ids = []
                    all_train_day_indices = []
                    all_train_returns = []
                    
                    # Gather base training data with diagnostic information
                    days_with_data = 0
                    total_stocks_found = 0
                    
                    for train_day in range(train_window_start, day_idx):
                        if train_day in stock_data_by_day:
                            day_features, day_stock_ids, day_returns = stock_data_by_day[train_day]
                            if len(day_stock_ids) > 0:
                                days_with_data += 1
                                total_stocks_found += len(day_stock_ids)
                                all_train_features.append(day_features)
                                all_train_stock_ids.append(day_stock_ids)
                                all_train_day_indices.append(np.ones_like(day_stock_ids) * train_day)
                                all_train_returns.append(day_returns)
                    
                    print(f"Found data for {days_with_data} days with {total_stocks_found} total stock observations")
                    
                    if len(all_train_features) > 0:
                        # Concatenate all training data
                        train_features = np.vstack(all_train_features)
                        train_stock_ids = np.concatenate(all_train_stock_ids)
                        train_day_indices = np.concatenate(all_train_day_indices)
                        train_returns = np.concatenate(all_train_returns)
                        
                        print(f"Preparing time series sequences from {len(train_stock_ids)} observations...")
                        
                        # Try the standard sequence preparation approach first
                        X_train, y_train = model.prepare_time_series_data(
                            train_stock_ids,
                            train_day_indices,
                            train_features,
                            train_returns,
                            window_size=self.window_size
                        )
                        
                        # If standard approach yields too few sequences, try direct stock-based approach
                        if len(X_train) < 100:  # Threshold for "too few" sequences
                            print(f"Standard approach only found {len(X_train)} sequences. Trying stock-based approach...")
                            
                            # Dictionary to map stock IDs to all their data points
                            stock_history = {}
                            
                            # Organize data by stock
                            for i in range(len(train_stock_ids)):
                                stock_id = train_stock_ids[i]
                                day_idx = train_day_indices[i]
                                features = train_features[i]
                                returns = train_returns[i]
                                
                                if stock_id not in stock_history:
                                    stock_history[stock_id] = {'days': [], 'features': [], 'returns': []}
                                
                                stock_history[stock_id]['days'].append(day_idx)
                                stock_history[stock_id]['features'].append(features)
                                stock_history[stock_id]['returns'].append(returns)
                            
                            # Generate sequences manually for each stock
                            manual_X = []
                            manual_y = []
                            
                            for stock_id, data in stock_history.items():
                                # Convert lists to arrays
                                days = np.array(data['days'])
                                features = np.array(data['features'])
                                returns = np.array(data['returns'])
                                
                                # Sort by day
                                sort_indices = np.argsort(days)
                                days = days[sort_indices]
                                features = features[sort_indices]
                                returns = returns[sort_indices]
                                
                                # Check if we have enough data points
                                if len(days) > self.window_size:
                                    # Generate sequences
                                    for i in range(len(days) - self.window_size):
                                        X_seq = features[i:i+self.window_size]
                                        y_val = returns[i+self.window_size]
                                        manual_X.append(X_seq)
                                        manual_y.append(y_val)
                            
                            # Convert to arrays if we found any sequences
                            if len(manual_X) > 0:
                                X_train = np.array(manual_X)
                                y_train = np.array(manual_y)
                                print(f"Stock-based approach found {len(X_train)} sequences")
                            else:
                                print("Stock-based approach also failed to find sequences")
                        
                        # Proceed with training if we have sequences
                        if len(X_train) > 0:
                            # Build model if not already built
                            if model.model is None:
                                model.build_model((X_train.shape[1], X_train.shape[2]))
                            
                            # If we have many sequences, sample to speed up training
                            if len(X_train) > 10000:
                                print(f"Sampling {10000} sequences from {len(X_train)} for faster retraining...")
                                sample_indices = np.random.choice(len(X_train), size=10000, replace=False)
                                X_sample = X_train[sample_indices]
                                y_sample = y_train[sample_indices]
                                print(f"Retraining TimeSeriesTransformer model with {len(X_sample)} sampled sequences...")
                                model.fit(X_sample, y_sample, epochs=30, batch_size=128, verbose=1)
                            else:
                                print(f"Retraining TimeSeriesTransformer model with {len(X_train)} sequences...")
                                model.fit(X_train, y_train, epochs=30, batch_size=128, verbose=1)
                                
                            model_has_been_trained = True
                            
                            # Save model checkpoint
                            try:
                                checkpoint_path = f'transformer_checkpoint_day_{day_idx}.keras'
                                model.model.save(checkpoint_path)
                                print(f"Saved model checkpoint to {checkpoint_path}")
                            except Exception as e:
                                print(f"Failed to save checkpoint: {e}")
                        else:
                            print("Warning: No valid sequences found for TimeSeriesTransformer retraining.")
                            # If we can't retrain, check if the model was previously trained
                            if model.model is not None:
                                print("Using existing model without retraining.")
                                model_has_been_trained = True
                            else:
                                print("No existing model found. Unable to make predictions.")
                    else:
                        print("Warning: No training data available for model retraining.")
                else:
                    # For KernelRegression models
                    train_features = []
                    train_returns = []
                    train_stock_ids = []
                    
                    for train_day in range(train_window_start, day_idx):
                        if train_day in stock_data_by_day:
                            day_features, day_stock_ids, day_returns = stock_data_by_day[train_day]
                            train_features.append(day_features)
                            train_returns.append(day_returns)
                            train_stock_ids.append(day_stock_ids)
                    
                    if len(train_features) > 0:
                        # Concatenate training data
                        train_features = np.vstack(train_features)
                        train_returns = np.concatenate(train_returns)
                        train_stock_ids = np.concatenate(train_stock_ids)
                        
                        # Retrain model
                        if isinstance(model, StockAwareKernelRegression):
                            model.fit(train_features, train_returns, train_stock_ids)
                        else:
                            model.fit(train_features, train_returns)
                
                last_retrain_day = day_idx
            
            # Get data for current day
            if day_idx not in stock_data_by_day:
                continue
                
            current_features, current_stock_ids, actual_returns = stock_data_by_day[day_idx]
            
            # Check if we need to rebalance the portfolio
            if day_idx - last_rebalance_day >= self.rebalance_freq:
                print(f"Rebalancing portfolio on day {day_idx}...")
                
                # Get predictions for current day
                if isinstance(model, (KernelRegression, StockAwareKernelRegression)):
                    if isinstance(model, StockAwareKernelRegression):
                        predicted_returns = model.predict(current_features, current_stock_ids)
                    else:
                        predicted_returns = model.predict(current_features)
                elif isinstance(model, TimeSeriesTransformer):
                    if not model_has_been_trained:
                        print("Warning: Model has not been trained yet. Using zero predictions.")
                        predicted_returns = np.zeros(len(current_stock_ids))
                    else:
                        # Create a consistent time series for prediction
                        look_back_days = list(range(day_idx - self.window_size, day_idx))
                        
                        # Collect historical data for all stocks in current universe
                        all_pred_features = []
                        all_pred_stock_ids = []
                        all_pred_day_indices = []
                        all_pred_returns = []
                        
                        for hist_day in look_back_days:
                            if hist_day in stock_data_by_day:
                                day_features, day_stock_ids, day_returns = stock_data_by_day[hist_day]
                                
                                # Keep only stocks that are in the current universe
                                mask = np.isin(day_stock_ids, current_stock_ids)
                                if np.any(mask):
                                    all_pred_features.append(day_features[mask])
                                    all_pred_stock_ids.append(day_stock_ids[mask])
                                    all_pred_day_indices.append(np.ones_like(day_stock_ids[mask]) * hist_day)
                                    all_pred_returns.append(day_returns[mask])
                        
                        # Process if we have data
                        if len(all_pred_features) > 0:
                            # Prepare data for prediction
                            pred_features = np.vstack(all_pred_features)
                            pred_stock_ids = np.concatenate(all_pred_stock_ids)
                            pred_day_indices = np.concatenate(all_pred_day_indices)
                            pred_returns = np.concatenate(all_pred_returns)
                            
                            # Use the same preparation method as training
                            X_pred, _ = model.prepare_time_series_data(
                                pred_stock_ids,
                                pred_day_indices,
                                pred_features,
                                pred_returns,
                                window_size=self.window_size
                            )
                            
                            if len(X_pred) > 0:
                                # Get predictions
                                try:
                                    batch_predictions = model.predict(X_pred)
                                    
                                    # Map predictions back to current universe stocks
                                    # First create a mapping from sequence to stock ID
                                    # (The last stock ID in each sequence is the one we're predicting for)
                                    seq_to_stock_map = {}
                                    
                                    # Use the same logic as in prepare_time_series_data to find sequence endpoints
                                    df = pd.DataFrame({
                                        'stock_id': pred_stock_ids,
                                        'day_idx': pred_day_indices,
                                        'row_idx': np.arange(len(pred_stock_ids))
                                    })
                                    
                                    # Sort the dataframe
                                    df = df.sort_values(['stock_id', 'day_idx']).reset_index(drop=True)
                                    
                                    # Find consecutive sequences
                                    df['group_change'] = (df['stock_id'] != df['stock_id'].shift(1)).astype(int)
                                    df['group_id'] = df['group_change'].cumsum()
                                    
                                    # Create sequence index within each group
                                    df['seq_idx'] = df.groupby('group_id').cumcount()
                                    
                                    # Find valid starting points
                                    valid_starts = df[df['seq_idx'] <= df.groupby('group_id')['seq_idx'].transform('max') - self.window_size]
                                    
                                    # For each valid sequence, get the corresponding stock_id
                                    for idx, row in valid_starts.iterrows():
                                        group_id = row['group_id']
                                        start_idx = row['seq_idx']
                                        
                                        # Find the last stock in this sequence (the one we're predicting for)
                                        end_seq = df[(df['group_id'] == group_id) & (df['seq_idx'] == start_idx + self.window_size - 1)]
                                        if len(end_seq) > 0:
                                            target_stock = end_seq.iloc[0]['stock_id']
                                            seq_to_stock_map[idx] = target_stock
                                    
                                    # Now map predictions to current universe stocks
                                    stock_predictions = {}
                                    for seq_idx, pred in enumerate(batch_predictions):
                                        if seq_idx in seq_to_stock_map:
                                            stock_id = seq_to_stock_map[seq_idx]
                                            stock_predictions[stock_id] = pred
                                    
                                    # Create prediction array for current universe
                                    predicted_returns = np.zeros(len(current_stock_ids))
                                    for i, stock_id in enumerate(current_stock_ids):
                                        if stock_id in stock_predictions:
                                            predicted_returns[i] = stock_predictions[stock_id]
                                except Exception as e:
                                    print(f"Prediction error: {e}")
                                    # Fallback to zero predictions
                                    predicted_returns = np.zeros(len(current_stock_ids))
                            else:
                                print("Warning: No valid sequences found for prediction. Using zero predictions.")
                                predicted_returns = np.zeros(len(current_stock_ids))
                        else:
                            print("Warning: No historical data found for current universe. Using zero predictions.")
                            predicted_returns = np.zeros(len(current_stock_ids))
                
                # Estimate covariance matrix using vectorized operations
                unique_current_stocks = np.unique(current_stock_ids)
                n_stocks = len(unique_current_stocks)
                max_history_length = min(60, day_idx - test_days[0])
                
                # Create stock ID lookup array for quick vectorized mapping
                max_stock_id = np.max(unique_current_stocks)
                stock_row_lookup = np.zeros(max_stock_id + 1, dtype=int) - 1
                stock_row_lookup[unique_current_stocks] = np.arange(n_stocks)
                
                # Pre-allocate historical returns matrix
                historical_returns = np.zeros((n_stocks, max_history_length))
                data_present = np.zeros((n_stocks, max_history_length), dtype=bool)
                
                # Process all historical days at once
                for col_idx, hist_day in enumerate(range(max(day_idx - 60, test_days[0]), day_idx)):
                    if hist_day in stock_data_by_day:
                        hist_stock_ids, hist_returns = stock_data_by_day[hist_day][1:3]
                        
                        # Find which of these stocks are in our universe using the lookup array
                        valid_indices = np.where((hist_stock_ids <= max_stock_id) & 
                                              (stock_row_lookup[hist_stock_ids] >= 0))[0]
                        
                        if len(valid_indices) > 0:
                            # Extract the valid returns and their corresponding row indices
                            valid_returns = hist_returns[valid_indices]
                            valid_stock_ids = hist_stock_ids[valid_indices]
                            row_indices = stock_row_lookup[valid_stock_ids]
                            
                            # Set the returns in our matrix
                            historical_returns[row_indices, col_idx] = valid_returns
                            data_present[row_indices, col_idx] = True
                
                # Generate random noise for missing data
                missing_data_mask = ~data_present
                historical_returns[missing_data_mask] = np.random.normal(0, 0.001, np.sum(missing_data_mask))
                
                # Calculate covariance matrix
                cov_matrix = np.cov(historical_returns)
                
                # Ensure covariance matrix is symmetric and positive definite
                cov_matrix = (cov_matrix + cov_matrix.T) / 2
                min_eig = np.min(np.real(np.linalg.eigvals(cov_matrix)))
                if min_eig < 0:
                    cov_matrix -= 1.1 * min_eig * np.eye(len(cov_matrix))
                
                # Map predicted returns to match the order in the covariance matrix
                current_expected_returns = np.zeros(n_stocks)

                # Vectorized alternative using np.bincount:
                prediction_lookup = np.zeros(max_stock_id + 1)
                # Handle potential duplicate stock IDs by averaging their predictions
                counts = np.bincount(current_stock_ids, minlength=max_stock_id + 1)
                sums = np.bincount(current_stock_ids, weights=predicted_returns, minlength=max_stock_id + 1)
                # Avoid division by zero by setting counts of 0 to 1
                mask = counts > 0
                prediction_lookup[mask] = sums[mask] / counts[mask]
                
                # Apply the lookup to get predictions in the right order
                current_expected_returns = prediction_lookup[unique_current_stocks]
                
                # Calculate market volatility and adjust risk aversion
                market_vol = np.std(np.mean(historical_returns, axis=0))
                risk_aversion = 1.0 + 10.0 * market_vol
                optimizer.risk_aversion = risk_aversion
                
                # Optimize portfolio weights
                optimal_weights = optimizer.optimize_large_portfolio(
                    current_expected_returns,
                    cov_matrix
                )['weights']
                
                # Convert optimization results to the needed formats
                current_weights = dict(zip(unique_current_stocks, optimal_weights))

                print("Computing portfolio metrics...")
                # Calculate portfolio metrics
                portfolio_metrics = optimizer.compute_portfolio_metrics(
                    optimal_weights,
                    current_expected_returns,
                    cov_matrix
                )
                
                # Store metrics and weights
                metrics_history.append({
                    'day_idx': day_idx,
                    'metrics': portfolio_metrics,
                    'risk_aversion': risk_aversion,
                    'market_volatility': market_vol
                })
                
                weights_history.append({
                    'day_idx': day_idx,
                    'weights': current_weights
                })
                
                last_rebalance_day = day_idx
            
            # Calculate portfolio return for current day
            if current_weights is not None:
                day_return = 0.0
                invested_weight = 0.0
                
                for stock_id, weight in current_weights.items():
                    # Find the actual return for this stock
                    stock_mask = current_stock_ids == stock_id
                    if np.sum(stock_mask) > 0:
                        stock_return = actual_returns[stock_mask][0]
                        day_return += weight * stock_return
                        invested_weight += weight
                
                # Adjust for cash (uninvested capital)
                if invested_weight < 1.0:
                    # Assume cash return is 0
                    cash_weight = 1.0 - invested_weight
                    day_return += cash_weight * 0.0
                
                # Update portfolio value
                current_value = portfolio_values[-1] * (1 + day_return)
                portfolio_values.append(current_value)
            else:
                # If no weights yet, assume no return
                portfolio_values.append(portfolio_values[-1])
        
        return portfolio_values, weights_history, metrics_history
    
    def plot_backtest_results(self, portfolio_values, weights_history, metrics_history, model_name):
        """
        Plot backtest results.
        
        Args:
            portfolio_values (list): Portfolio values over time
            weights_history (list): Portfolio weights history
            metrics_history (list): Portfolio metrics history
            model_name (str): Name of the model used
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
                    print(f"Warning: Dimension mismatch in metrics arrays: days={len(metric_days)}, " +
                          f"risk_aversions={len(risk_aversions)}, market_vols={len(market_vols)}")
            
            # Plot expected return vs. realized return
            if len(metrics_history) > 1:
                ax4 = plt.subplot(2, 2, 4)
                
                try:
                    # Calculate realized returns
                    realized_returns = []
                    for i in range(1, len(portfolio_values)):
                        realized_returns.append(portfolio_values[i] / portfolio_values[i-1] - 1)
                    
                    # Extract expected returns from metrics
                    expected_returns = [m['metrics']['expected_return'] for m in metrics_history]
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
        plt.savefig(f'backtest_results_{model_name}.png', dpi=300)
        plt.close(fig)