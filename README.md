# PyTorch Quantitative Modeling for Portfolio Allocation

A comprehensive framework for quantitative finance modeling using PyTorch, featuring time series prediction models, portfolio optimization, and backtesting capabilities.

## Overview

This project implements a complete quantitative modeling pipeline for portfolio allocation, including:

- Feature engineering for financial time series data
- Multiple prediction models (Kernel Regression, Stock-Aware Kernel Regression, Transformer)
- Portfolio optimization with risk constraints
- Backtesting framework to evaluate strategy performance

## Features

- **PyTorch-based Implementation**: Leverages GPU acceleration for model training and inference
- **Multiple Model Options**:
  - Kernel Regression
  - Stock-Aware Kernel Regression
  - Time Series Transformer (with configurable architecture)
- **Feature Engineering**: Automated feature generation for financial time series
- **Portfolio Optimization**: Mean-variance optimization with customizable risk aversion
- **Comprehensive Backtesting**: Evaluate strategies with realistic trading constraints
- **Flexible Configuration**: Command-line arguments for easy experimentation

## Requirements

- Python 3.8+
- PyTorch 1.10+
- NumPy
- Pandas
- Matplotlib
- Seaborn

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pytorch-quant-portfolio.git
cd pytorch-quant-portfolio

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python main.py --data_path market_data.npy --model_type transformer
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data_path` | Path to the market data file | market_data.npy |
| `--model_type` | Model type (`kernel`, `stock_kernel`, `transformer`) | transformer |
| `--risk_aversion` | Risk aversion parameter | 1.0 |
| `--max_weight` | Maximum weight for a single stock | 0.05 |
| `--window_size` | Window size for time series data | 20 |
| `--rebalance_freq` | Portfolio rebalance frequency in days | 5 |
| `--retrain_freq` | Model retraining frequency in days | 20 |
| `--output_dir` | Directory to save results | results |
| `--load_model` | Path to load a pretrained model | None |
| `--stock_count` | Choose the first k stocks to run | None |

### Transformer-Specific Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--d_model` | Dimension of transformer model | 128 |
| `--num_heads` | Number of attention heads | 8 |
| `--d_ff` | Dimension of feed-forward network | 256 |
| `--num_layers` | Number of transformer layers | 4 |
| `--dropout_rate` | Dropout rate | 0.1 |
| `--learning_rate` | Learning rate | 1e-3 |
| `--retrain` | Force retraining of the model | False |

### Feature Engineering Control

| Argument | Description | Default |
|----------|-------------|---------|
| `--encode` | Whether to use/train an autoencoder | False |
| `--sector` | Stock filter for the prediction and portfolio by sector | None |
| `--industry` | Stock filter for the prediction and portfolio by industry | None |

### PyTorch-Specific Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--device` | Device to use for PyTorch models (`cuda` or `cpu`) | auto-detected |
| `--batch_size` | Batch size for training | 64 |
| `--epochs` | Number of epochs for training | 30 |

## Example Workflows

### Training a Transformer Model

```bash
python main.py --data_path market_data.npy --model_type transformer --d_model 256 --num_heads 8 --num_layers 6 --output_dir results/transformer_exp1
```

### Using a Pre-trained Model

```bash
python main.py --data_path market_data.npy --model_type transformer --load_model results/transformer_exp1/models/transformer_model.pt
```

### Sector-Specific Portfolio

```bash
python main.py --data_path market_data.npy --model_type transformer --sector 3 --risk_aversion 2.0
```

## Output

The backtest results will include:

- Portfolio values over time
- Position weights history
- Performance metrics (returns, volatility, Sharpe ratio)
- Visualizations of portfolio performance
- Saved model checkpoints

## Project Structure

- `main.py`: Entry point for the application
- `feature_engineering.py`: Implementation of PyTorch feature engineering
- `nonparametric_regression.py`: Kernel regression models
- `transformer.py`: Time Series Transformer implementation
- `portfolio_optimizer.py`: Mean-variance portfolio optimization
- `backtest.py`: Backtesting framework

## License

[Add your license information here]

## Contributing

[Add your contribution guidelines here]
