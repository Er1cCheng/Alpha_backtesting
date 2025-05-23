import numpy as np
import pandas as pd
import time
from datetime import datetime
import argparse
import pickle
from scipy.optimize import minimize

# GPU libraries
try:
    import cupy as cp  # For CUDA NumPy operations

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    print("CuPy not found. Using NumPy instead.")

try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit

    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    print("JAX not found. JAX acceleration unavailable.")

# For PyTorch implementation
try:
    import torch
    from torch.optim import LBFGS, Adam

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not found. PyTorch acceleration unavailable.")


class PortfolioOptimizer:
    def __init__(
        self,
        risk_aversion=1.0,
        max_weight=0.05,
        solver="SLSQP",
        use_gpu=True,
        gpu_library="d",
        device="cuda:0",
    ):
        """
        Initialize the GPU-accelerated portfolio optimizer.

        Args:
            risk_aversion (float): Risk aversion parameter (lambda)
            max_weight (float): Maximum weight for a single stock
            solver (str): Optimization solver to use ('SLSQP', 'Adam', 'LBFGS', etc.)
            use_gpu (bool): Whether to use GPU acceleration
            gpu_library (str): Which GPU library to use ('cupy', 'jax', 'pytorch')
            device (str): GPU device to use (for PyTorch)
        """
        self.risk_aversion = risk_aversion
        self.max_weight = max_weight
        self.solver = solver
        self.use_gpu = use_gpu
        self.gpu_library = gpu_library
        self.device = device

        # Check if requested GPU library is available
        if use_gpu:
            if gpu_library == "jax" and not HAS_JAX:
                print("JAX requested but not available. Falling back to NumPy.")
                self.use_gpu = False
            elif gpu_library == "pytorch" and not HAS_TORCH:
                print("PyTorch requested but not available. Falling back to NumPy.")
                self.use_gpu = False

    def _to_device(self, data):
        """Helper to move data to the appropriate device/library"""
        if not self.use_gpu:
            return np.asarray(data)

        if self.gpu_library == "jax":
            return jnp.asarray(data)
        elif self.gpu_library == "pytorch":
            return torch.tensor(data, device=self.device, dtype=torch.float64)

        return np.asarray(data)  # Fallback

    def _from_device(self, data):
        """Helper to move data back to CPU numpy arrays"""
        if not self.use_gpu:
            return data

        if self.gpu_library == "jax":
            return np.asarray(data)
        elif self.gpu_library == "pytorch":
            return data.cpu().numpy()

        return data  # Fallback

    def orig_mean_variance_optimization(
        self, expected_returns, cov_matrix, use_jac=True, tol=1e-8, max_iter=1000
    ):
        """
        Perform mean-variance optimization with performance enhancements.

        Args:
            expected_returns (ndarray): Expected returns for each stock
            cov_matrix (ndarray): Covariance matrix of returns
            use_jac (bool): Whether to use analytical Jacobian for faster convergence
            tol (float): Tolerance for optimization
            max_iter (int): Maximum iterations

        Returns:
            dict: Results containing weights and optimization stats
        """
        start_time = time.time()
        n_assets = len(expected_returns)

        # Pre-compute some matrices for efficiency
        expected_returns = np.asarray(expected_returns)
        cov_matrix = np.asarray(cov_matrix)

        # Initial guess: equal weight
        initial_weights = np.ones(n_assets) / n_assets

        # Constraints: weights sum to 1
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        if use_jac:
            constraints[0]["jac"] = lambda w: np.ones(n_assets)

        bounds = [(0.0, self.max_weight) for _ in range(n_assets)]

        # Analytical objective function and gradient for speed
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            utility = portfolio_return - (self.risk_aversion / 2) * portfolio_variance
            return -utility

        # Analytical gradient for faster convergence
        def gradient(weights):
            return -(
                expected_returns - self.risk_aversion * np.dot(cov_matrix, weights)
            )

        # Optimize with or without analytical gradient
        options = {"disp": False, "maxiter": max_iter, "ftol": tol}

        if use_jac:
            result = minimize(
                objective,
                initial_weights,
                method=self.solver,
                jac=gradient,
                bounds=bounds,
                constraints=constraints,
                options=options,
            )
        else:
            result = minimize(
                objective,
                initial_weights,
                method=self.solver,
                bounds=bounds,
                constraints=constraints,
                options=options,
            )

        end_time = time.time()

        output = {
            "weights": result.x if result.success else initial_weights,
            "success": result.success,
            "status": result.message,
            "objective_value": result.fun,
            "execution_time": end_time - start_time,
            "iterations": result.nit if hasattr(result, "nit") else None,
        }

        return output

    def monte_carlo_presampling(
        self, expected_returns, cov_matrix, num_samples=10000, top_n=10
    ):
        """
        Use Monte Carlo presampling to reduce the problem dimension.
        Generate random portfolios and select the top performing ones as a starting point.

        Args:
            expected_returns (ndarray): Expected returns for each stock
            cov_matrix (ndarray): Covariance matrix of returns
            num_samples (int): Number of random portfolios to generate
            top_n (int): Number of top portfolios to use for further optimization

        Returns:
            ndarray: Weights from the best portfolio found
        """
        n_assets = len(expected_returns)
        best_utility = -np.inf
        best_weights = np.ones(n_assets) / n_assets

        print(f"Generating {num_samples} random portfolios...")

        # Generate random weights that sum to 1 and respect max_weight
        for _ in range(num_samples):
            # Generate random weights
            weights = np.random.uniform(0, self.max_weight, n_assets)
            # Normalize to sum to 1
            weights = weights / np.sum(weights)

            # Ensure max weight constraint
            while np.any(weights > self.max_weight):
                excess = weights - self.max_weight
                excess[excess < 0] = 0
                weights = np.minimum(weights, self.max_weight)
                # Redistribute excess
                weights = weights + (np.sum(excess) / n_assets)
                # Ensure again
                weights = np.minimum(weights, self.max_weight)
                # Normalize again
                weights = weights / np.sum(weights)

            # Calculate utility
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            utility = portfolio_return - (self.risk_aversion / 2) * portfolio_variance

            if utility > best_utility:
                best_utility = utility
                best_weights = weights

        return best_weights

    def optimize_large_portfolio(
        self,
        expected_returns,
        cov_matrix,
        use_monte_carlo=True,
        num_samples=10000,
        use_jac=True,
    ):
        """
        Main method to optimize large portfolios efficiently.

        Args:
            expected_returns (ndarray): Expected returns for each stock
            cov_matrix (ndarray): Covariance matrix of returns
            use_monte_carlo (bool): Whether to use Monte Carlo presampling
            num_samples (int): Number of random portfolios for Monte Carlo
            use_jac (bool): Whether to use analytical Jacobian

        Returns:
            dict: Optimization results
        """
        n_assets = len(expected_returns)
        print(f"Optimizing portfolio with {n_assets} assets...")

        # For very large portfolios, use Monte Carlo presampling to get a good initial guess
        if use_monte_carlo and n_assets > 100:
            initial_weights = self.monte_carlo_presampling(
                expected_returns, cov_matrix, num_samples=num_samples
            )
        else:
            initial_weights = np.ones(n_assets) / n_assets

        # Run the optimization with the enhanced initial guess
        return self.orig_mean_variance_optimization(
            expected_returns, cov_matrix, use_jac=use_jac
        )

    def optimize_with_jax(
        self, portfolio_value, expected_returns, cov_matrix, tol=1e-6, max_iter=1000
    ):
        """Portfolio optimization using JAX with custom optimizer"""
        n_assets = len(expected_returns)

        # Move data to JAX arrays
        expected_returns_jax = jnp.asarray(expected_returns)
        cov_matrix_jax = jnp.asarray(cov_matrix)

        # Define objective function using JAX
        @jit
        def objective(weights):
            portfolio_return = jnp.dot(weights, expected_returns_jax)
            portfolio_variance = jnp.dot(weights.T, jnp.dot(cov_matrix_jax, weights))
            utility = portfolio_return - (self.risk_aversion / 2) * portfolio_variance

            # Add penalty for constraint violation (weights sum to 1)
            sum_constraint = jnp.square(jnp.sum(weights) - 1.0) * 1000

            # Add penalty for weight bounds
            bound_violations = jnp.sum(
                jnp.maximum(0, weights - self.max_weight)
            ) + jnp.sum(jnp.maximum(0, -weights))
            # bound_violations = jnp.sum(jnp.maximum(0, weights - self.max_weight)) + \
            #           jnp.sum(jnp.maximum(0, -weights - self.max_weight))
            bound_penalty = bound_violations * 1000

            return -utility + sum_constraint + bound_penalty

        # Get gradient function (JAX auto-differentiation)
        objective_grad = jit(grad(objective))

        # Initial guess: equal weight, projected to satisfy constraints
        weights = jnp.ones(n_assets) / n_assets

        # Simple projected gradient descent
        step_size = 0.01
        prev_loss = float("inf")

        for i in range(max_iter):
            # Calculate gradient
            g = objective_grad(weights)

            # Update weights
            weights = weights - step_size * g

            # Project to satisfy constraints (approximately)
            # 1. Clip to bounds
            weights = jnp.clip(weights, 0, self.max_weight)

            # 2. Normalize to sum to 1 (approximately)
            weights = weights / jnp.sum(weights)

            # Check convergence
            loss = objective(weights)
            if jnp.abs(prev_loss - loss) < tol:
                break
            prev_loss = loss

        # Calculate final objective value (remove penalties)
        portfolio_return = jnp.dot(weights, expected_returns_jax)
        portfolio_variance = jnp.dot(weights.T, jnp.dot(cov_matrix_jax, weights))
        utility = portfolio_return - (self.risk_aversion / 2) * portfolio_variance

        # Move results back to CPU
        output = {
            "weights": np.asarray(weights),
            "success": True,
            "status": f"Converged after {i+1} iterations",
            "objective_value": float(-utility),
            "execution_time": None,  # Will be set by caller
            "iterations": i + 1,
        }

        return output

    def optimize_with_pytorch(
        self, expected_returns, cov_matrix, tol=1e-6, max_iter=1000
    ):
        """Portfolio optimization using PyTorch with auto-differentiation"""
        n_assets = len(expected_returns)

        # Move data to GPU tensors
        expected_returns_gpu = torch.tensor(
            expected_returns, device=self.device, dtype=torch.float64
        )
        cov_matrix_gpu = torch.tensor(
            cov_matrix, device=self.device, dtype=torch.float64
        )

        # Create weights as trainable parameters
        weights = (
            torch.ones(n_assets, device=self.device, dtype=torch.float64) / n_assets
        )
        weights.requires_grad = True

        # Set up optimizer
        if self.solver.upper() == "LBFGS":
            optimizer = LBFGS(
                [weights], lr=0.1, max_iter=20, line_search_fn="strong_wolfe"
            )
        else:  # Default to Adam
            optimizer = Adam([weights], lr=0.01)

        # Define closure for optimization
        def closure():
            optimizer.zero_grad()

            # Project weights to satisfy constraints
            w = weights.clamp(0, self.max_weight)
            w = w / w.sum()

            # Calculate utility
            portfolio_return = torch.dot(w, expected_returns_gpu)
            portfolio_variance = torch.dot(w, torch.mv(cov_matrix_gpu, w))
            utility = portfolio_return - (self.risk_aversion / 2) * portfolio_variance

            # We want to maximize utility, so we minimize negative utility
            loss = -utility
            loss.backward()
            return loss

        # Run optimization
        prev_loss = float("inf")
        for i in range(max_iter):
            loss = optimizer.step(closure)

            # Check convergence
            if abs(prev_loss - loss.item()) < tol:
                break
            prev_loss = loss.item()

        # Get final projected weights
        final_weights = weights.detach().clamp(0, self.max_weight)
        final_weights = final_weights / final_weights.sum()

        # Calculate final utility
        portfolio_return = torch.dot(final_weights, expected_returns_gpu)
        portfolio_variance = torch.dot(
            final_weights, torch.mv(cov_matrix_gpu, final_weights)
        )
        utility = portfolio_return - (self.risk_aversion / 2) * portfolio_variance

        # Move results back to CPU
        output = {
            "weights": final_weights.cpu().numpy(),
            "success": True,
            "status": f"Converged after {i+1} iterations",
            "objective_value": -utility.item(),
            "execution_time": None,  # Will be set by caller
            "iterations": i + 1,
        }

        return output

    def mean_variance_optimization(
        self, portfolio_value, expected_returns, cov_matrix, tol=1e-8, max_iter=1000
    ):
        """
        Perform GPU-accelerated mean-variance optimization.

        Args:
            expected_returns (ndarray): Expected returns for each stock
            cov_matrix (ndarray): Covariance matrix of returns
            tol (float): Tolerance for optimization
            max_iter (int): Maximum iterations

        Returns:
            dict: Results containing weights and optimization stats
        """
        start_time = time.time()

        if not self.use_gpu:
            # Fall back to standard SciPy optimization on CPU
            from scipy.optimize import minimize

            n_assets = len(expected_returns)
            expected_returns = np.asarray(expected_returns)
            cov_matrix = np.asarray(cov_matrix)
            initial_weights = np.ones(n_assets) / n_assets

            # Define objective function and gradient
            def objective(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
                utility = (
                    portfolio_return - (self.risk_aversion / 2) * portfolio_variance
                )
                return -utility

            def gradient(weights):
                return -(
                    expected_returns - self.risk_aversion * np.dot(cov_matrix, weights)
                )

            # Constraints: weights sum to 1
            constraints = [
                {
                    "type": "eq",
                    "fun": lambda w: np.sum(w) - 1.0,
                    "jac": lambda w: np.ones(n_assets),
                }
            ]

            bounds = [(0.0, self.max_weight) for _ in range(n_assets)]

            # Optimize
            result = minimize(
                objective,
                initial_weights,
                method=self.solver,
                jac=gradient,
                bounds=bounds,
                constraints=constraints,
                options={"disp": False, "maxiter": max_iter, "ftol": tol},
            )

            output = {
                "weights": result.x if result.success else initial_weights,
                "success": result.success,
                "status": result.message,
                "objective_value": result.fun,
                "execution_time": None,  # Will be set later
                "iterations": result.nit if hasattr(result, "nit") else None,
            }

        else:
            # Use GPU optimization based on selected library
            if self.gpu_library == "jax":
                output = self.optimize_with_jax(
                    portfolio_value, expected_returns, cov_matrix, tol, max_iter
                )
            elif self.gpu_library == "pytorch":
                output = self.optimize_with_pytorch(
                    portfolio_value, expected_returns, cov_matrix, tol, max_iter
                )
            else:
                output = self.optimize_large_portfolio(
                    portfolio_value, expected_returns, cov_matrix
                )

        end_time = time.time()
        output["execution_time"] = end_time - start_time

        return output

    def compute_portfolio_metrics(self, weights, returns, cov_matrix):
        """
        Compute portfolio metrics on CPU.

        Args:
            weights (ndarray): Portfolio weights
            returns (ndarray): Expected returns for each stock
            cov_matrix (ndarray): Covariance matrix of returns

        Returns:
            dict: Portfolio metrics
        """
        # Convert everything to NumPy for consistency
        weights = np.asarray(weights)
        returns = np.asarray(returns)
        cov_matrix = np.asarray(cov_matrix)

        # Calculate portfolio expected return
        portfolio_return = np.dot(weights, returns)

        # Calculate portfolio risk (standard deviation)
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_risk = np.sqrt(portfolio_variance)

        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0

        # Calculate concentration metrics
        top5_concentration = np.sum(np.sort(weights)[-5:])
        herfindahl_index = np.sum(weights**2)

        return {
            "expected_return": portfolio_return,
            "risk": portfolio_risk,
            "sharpe_ratio": sharpe_ratio,
            "top5_concentration": top5_concentration,
            "herfindahl_index": herfindahl_index,
        }
