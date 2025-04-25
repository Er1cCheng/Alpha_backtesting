import os
from re import VERBOSE

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import time
from datetime import datetime
import pickle
import warnings
import torch

from feature_engineering import PyTorchFeatureEngineering
from nonparametric_regression import KernelRegression, StockAwareKernelRegression
from backtest import PyTorchBacktestFramework
from portfolio_optimizer import PortfolioOptimizer
from transformer import TimeSeriesTransformer


def save_portfolio_values_to_file(
    portfolio_values, filename="portfolio_values.txt", directory="."
):
    """
    Save all values in the portfolio_values list to a single file.

    Args:
        portfolio_values: List of portfolio values
        filename: Name of the file to save values to
        directory: Directory to save the file in (created if it doesn't exist)
    """
    # Create directory if it doesn't exist
    if not os.path.exists(directory) and directory != ".":
        os.makedirs(directory)
        print(f"Created directory: {directory}")

    # Create the full path
    filepath = os.path.join(directory, filename)

    # Save all values to a single file
    with open(filepath, "w") as file:
        for value in portfolio_values:
            file.write(f"{value}\n")

    print(f"Successfully saved {len(portfolio_values)} values to {filepath}")


def main():
    """
    Main function to run the fully PyTorch-based quantitative modeling pipeline.
    """
    parser = argparse.ArgumentParser(
        description="PyTorch Quantitative Modeling for Portfolio Allocation"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="market_data.npy",
        help="Path to the market data file",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="transformer",
        choices=["kernel", "stock_kernel", "transformer"],
        help="Type of model to use for prediction",
    )
    parser.add_argument(
        "--risk_aversion",
        type=float,
        default=1.0,
        help="Risk aversion parameter for portfolio optimization",
    )
    parser.add_argument(
        "--max_weight",
        type=float,
        default=0.05,
        help="Maximum weight for a single stock",
    )
    parser.add_argument(
        "--window_size", type=int, default=20, help="Window size for time series data"
    )
    parser.add_argument(
        "--rebalance_freq",
        type=int,
        default=5,
        help="Portfolio rebalance frequency in days",
    )
    parser.add_argument(
        "--retrain_freq",
        type=int,
        default=20,
        help="Model retraining frequency in days",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Directory to save results"
    )
    parser.add_argument(
        "--load_model", type=str, default=None, help="Path to load a pretrained model"
    )
    parser.add_argument(
        "--stock_count", type=int, default=None, help="Choose the first k stocks to run"
    )

    # Transformer specific arguments
    parser.add_argument(
        "--d_model", type=int, default=128, help="Dimension of transformer model"
    )
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--d_ff", type=int, default=256, help="Dimension of feed-forward network"
    )
    parser.add_argument(
        "--num_layers", type=int, default=4, help="Number of transformer layers"
    )
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument(
        "--retrain", action="store_true", help="force to train the model"
    )

    # Feature Engineering Control
    parser.add_argument(
        "--encode", action="store_true", help="Whether to use/train an autoencoder"
    )
    parser.add_argument(
        "--sector",
        type=int,
        default=None,
        help="Stock filter for the prediction and portfolio",
    )
    parser.add_argument(
        "--industry",
        type=int,
        default=None,
        help="Stock filter for the prediction and portfolio",
    )

    # Optimizer Control
    parser.add_argument(
        "--optimizer_type", type=str, default=None, help="Directory to save results"
    )

    # PyTorch specific arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for PyTorch models (cuda or cpu)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of epochs for training"
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Print PyTorch device information
    print(f"PyTorch device: {args.device}")
    if args.device == "cuda":
        if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            print(
                f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
            )
        else:
            print("CUDA requested but not available, falling back to CPU")
            args.device = "cpu"

    # Load data
    print(f"Loading data from {args.data_path}...")
    data_array = np.load(args.data_path, allow_pickle=True)
    data_dict = data_array.item()
    print(
        f"Loaded data with {len(data_dict['si'])} entries and {len(set(data_dict['si']))} unique stocks"
    )

    # Feature engineering with PyTorch
    print("Performing PyTorch feature engineering...")
    feature_eng = PyTorchFeatureEngineering(
        data_dict,
        args.encode,
        device=args.device,
        stock_count=args.stock_count,
        sector=args.sector,
        industry=args.industry,
    )
    train_test_dict = feature_eng.generate_features(output_dir=args.output_dir)

    # Initialize model based on model_type
    print(f"\nInitializing {args.model_type} model...")

    if args.model_type == "kernel":
        model = KernelRegression(bandwidth="auto", verbose=1)
    elif args.model_type == "stock_kernel":
        model = StockAwareKernelRegression(bandwidth="auto")
    else:  # transformer
        model = TimeSeriesTransformer(
            max_seq_length=args.window_size,
            d_model=args.d_model,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            num_layers=args.num_layers,
            dropout_rate=args.dropout_rate,
            device=args.device,
        )

    # Initialize portfolio optimizer
    optimizer = PortfolioOptimizer(
        risk_aversion=args.risk_aversion,
        max_weight=args.max_weight,
        gpu_library=args.optimizer_type,
    )

    # Initialize backtest framework - use PyTorch version
    backtest = PyTorchBacktestFramework(
        train_test_dict,
        args.model_type,
        args.output_dir,
        window_size=args.window_size,
        rebalance_freq=args.rebalance_freq,
        stock_count=args.stock_count,
    )

    # Determine test period
    unique_days = sorted(np.unique(train_test_dict["test_di"]))
    start_day_idx = unique_days[0]
    end_day_idx = unique_days[-1]

    # Check if a pre-trained model should be loaded
    model_dir = os.path.join(args.output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    # Define model filename
    model_filename = f"{args.model_type}_model.pt"
    model_path = os.path.join(model_dir, model_filename)

    # Load model if available
    if not args.retrain and (args.load_model or os.path.exists(model_path)):
        load_path = args.load_model if args.load_model else model_path
        print(f"Loading pre-trained model from {load_path}...")
        try:
            if args.model_type == "transformer":
                model.load_model(load_path)
            else:
                with open(load_path, "rb") as f:
                    model = pickle.load(f)
            print("Model loaded successfully!")
            disable_retraining = True
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Will train a new model during backtest.")
            disable_retraining = False
    else:
        print(
            f"No pre-trained model found. Will train {args.model_type} model during backtest."
        )
        disable_retraining = False
    model.verbose = 0

    # Run backtest
    print(
        f"Running backtest for {args.model_type} model from day {start_day_idx} to {end_day_idx}..."
    )
    portfolio_values, weights_history, metrics_history, _ = backtest.run_backtest(
        model,
        optimizer,
        start_day_idx,
        end_day_idx,
        retrain_freq=args.retrain_freq,
        disable_retraining=disable_retraining,
    )

    # Save the model after backtest
    print(f"Saving {args.model_type} model to {model_path}...")
    try:
        if args.model_type == "transformer":
            model.save_model(model_path)
        else:
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
        print("Model saved successfully!")
    except Exception as e:
        print(f"Error saving model: {e}")

    # Model identifier for output files
    model_id = args.model_type

    # Save backtest results
    results = {
        "portfolio_values": portfolio_values,
        "weights_history": weights_history,
        "metrics_history": metrics_history,
        "model_type": args.model_type,
        "risk_aversion": args.risk_aversion,
        "max_weight": args.max_weight,
        "window_size": args.window_size,
        "rebalance_freq": args.rebalance_freq,
        "retrain_freq": args.retrain_freq,
        "start_day_idx": start_day_idx,
        "end_day_idx": end_day_idx,
        "run_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    # For transformer models, also save hyperparameters
    if args.model_type == "transformer":
        results.update(
            {
                "d_model": args.d_model,
                "num_heads": args.num_heads,
                "d_ff": args.d_ff,
                "num_layers": args.num_layers,
                "dropout_rate": args.dropout_rate,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "device": args.device,
            }
        )

    # Save results to file
    results_path = os.path.join(args.output_dir, f"backtest_results_{model_id}.pickle")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)

    # Plot backtest results
    print("Plotting backtest results...")
    backtest.plot_backtest_results(
        portfolio_values, weights_history, metrics_history, model_id
    )

    save_portfolio_values_to_file(portfolio_values)

    # Print final performance
    total_return = portfolio_values[-1] / portfolio_values[0] - 1
    annual_return = (1 + total_return) ** (252 / len(portfolio_values)) - 1
    returns = np.array(portfolio_values[1:]) / np.array(portfolio_values[:-1]) - 1
    volatility = np.std(returns) * np.sqrt(252)
    sharpe_ratio = annual_return / volatility if volatility > 0 else 0

    print(f"\nBacktest Results for {model_id.upper()} Model:")
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
