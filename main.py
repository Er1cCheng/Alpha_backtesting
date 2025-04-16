import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

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
from backtest import *
from portfolio_optimizer import *
        
def main():
    """
    Main function to run the quantitative modeling pipeline.
    """
    parser = argparse.ArgumentParser(description='Quantitative Modeling for Portfolio Allocation')
    parser.add_argument('--data_path', type=str, default='market_data.npy', help='Path to the market data file')
    parser.add_argument('--model_type', type=str, default='kernel', choices=['kernel', 'stock_kernel', 'transformer'],
                        help='Type of model to use for prediction')
    parser.add_argument('--risk_aversion', type=float, default=1.0, help='Risk aversion parameter for portfolio optimization')
    parser.add_argument('--max_weight', type=float, default=0.05, help='Maximum weight for a single stock')
    parser.add_argument('--window_size', type=int, default=20, help='Window size for time series data')
    parser.add_argument('--rebalance_freq', type=int, default=5, help='Portfolio rebalance frequency in days')
    parser.add_argument('--retrain_freq', type=int, default=20, help='Model retraining frequency in days')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--load_model', type=str, default=None, help='Path to load a pretrained model')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load data
    print(f"Loading data from {args.data_path}...")
    data_array = np.load(args.data_path, allow_pickle=True)
    data_dict = data_array.item()
    print(len(data_dict["si"]), len(set(data_dict["si"])))
    
    # Feature engineering
    print("Performing feature engineering...")
    feature_eng = FeatureEngineering(data_dict)
    train_test_dict = feature_eng.generate_features(output_dir=args.output_dir)
    
    # Save processed data
    # print("Saving processed data...")
    # with open(os.path.join(args.output_dir, 'processed_data.pickle'), 'wb') as f:
    #     pickle.dump(train_test_dict, f)
    
    # Initialize model based on model_type
    print(f"Initializing {args.model_type} model...")
    if args.model_type == 'kernel':
        model = KernelRegression(bandwidth='auto')
    elif args.model_type == 'stock_kernel':
        model = StockAwareKernelRegression(bandwidth='auto')
    elif args.model_type == 'transformer':
        model = TimeSeriesTransformer(max_seq_length=args.window_size)
    
    # Initialize portfolio optimizer
    optimizer = PortfolioOptimizer(risk_aversion=args.risk_aversion, max_weight=args.max_weight)
    
    # Initialize backtest framework
    backtest = BacktestFramework(train_test_dict, window_size=args.window_size, rebalance_freq=args.rebalance_freq)
    
    # Determine test period
    unique_days = sorted(np.unique(train_test_dict['test_di']))
    start_day_idx = unique_days[0]
    end_day_idx = unique_days[-1]
    
    # After initializing the model but before running the backtest
    model_dir = os.path.join(args.output_dir, 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_filename = f"{args.model_type}_model.pickle"
    model_path = os.path.join(model_dir, model_filename)

    # Check if model exists and load it
    if os.path.exists(model_path):
        print(f"Loading pre-trained {args.model_type} model from {model_path}...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully!")
    else:
        print(f"No pre-trained model found. Will train {args.model_type} model during backtest.")

    # Save model after backtest is complete
    def save_model_callback(model):
        print(f"Saving {args.model_type} model to {model_path}...")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print("Model saved successfully!")

    # Determine if we should disable retraining
    disable_retraining = os.path.exists(model_path)

    # Run backtest
    print(f"Running backtest for {args.model_type} model from day {start_day_idx} to {end_day_idx}...")
    portfolio_values, weights_history, metrics_history = backtest.run_backtest(
        model,
        optimizer,
        start_day_idx,
        end_day_idx,
        retrain_freq=args.retrain_freq,
        disable_retraining=disable_retraining
    )

    # Save the model after backtest
    save_model_callback(model)
    
    # Save backtest results
    results = {
        'portfolio_values': portfolio_values,
        'weights_history': weights_history,
        'metrics_history': metrics_history,
        'model_type': args.model_type,
        'risk_aversion': args.risk_aversion,
        'max_weight': args.max_weight,
        'window_size': args.window_size,
        'rebalance_freq': args.rebalance_freq,
        'retrain_freq': args.retrain_freq,
        'start_day_idx': start_day_idx,
        'end_day_idx': end_day_idx,
        'run_timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
    }
    
    with open(os.path.join(args.output_dir, f'backtest_results_{args.model_type}.pickle'), 'wb') as f:
        pickle.dump(results, f)
    
    # Plot backtest results
    print("Plotting backtest results...")
    backtest.plot_backtest_results(
        portfolio_values,
        weights_history,
        metrics_history,
        args.model_type
    )
    
    # Print final performance
    total_return = portfolio_values[-1] / portfolio_values[0] - 1
    annual_return = (1 + total_return) ** (252 / len(portfolio_values)) - 1
    returns = np.array(portfolio_values[1:]) / np.array(portfolio_values[:-1]) - 1
    volatility = np.std(returns) * np.sqrt(252)
    sharpe_ratio = annual_return / volatility if volatility > 0 else 0
    
    print(f"\nBacktest Results for {args.model_type.upper()} Model:")
    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Return: {annual_return:.2%}")
    print(f"Annualized Volatility: {volatility:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Final Portfolio Value: ${portfolio_values[-1]:.2f}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\nTotal execution time: {(end_time - start_time) / 60:.2f} minutes")