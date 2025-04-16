import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import argparse
import time
from datetime import datetime
import pickle

class PortfolioOptimizer:
    def __init__(self, risk_aversion=1.0, max_weight=0.05, solver='SLSQP'):
        """
        Initialize the portfolio optimizer with performance enhancements.
        
        Args:
            risk_aversion (float): Risk aversion parameter (lambda)
            max_weight (float): Maximum weight for a single stock
            solver (str): Optimization solver to use ('SLSQP', 'trust-constr', etc.)
        """
        self.risk_aversion = risk_aversion
        self.max_weight = max_weight
        self.solver = solver
    
    def mean_variance_optimization(self, expected_returns, cov_matrix, use_jac=True, tol=1e-8, max_iter=1000):
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
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        
        if use_jac:
            constraints[0]['jac'] = lambda w: np.ones(n_assets)
        
        bounds = [(0.0, self.max_weight) for _ in range(n_assets)]
        
        # Analytical objective function and gradient for speed
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            utility = portfolio_return - (self.risk_aversion / 2) * portfolio_variance
            return -utility
        
        # Analytical gradient for faster convergence
        def gradient(weights):
            return -(expected_returns - self.risk_aversion * np.dot(cov_matrix, weights))
        
        # Optimize with or without analytical gradient
        options = {'disp': False, 'maxiter': max_iter, 'ftol': tol}
        
        if use_jac:
            result = minimize(
                objective,
                initial_weights,
                method=self.solver,
                jac=gradient,
                bounds=bounds,
                constraints=constraints,
                options=options
            )
        else:
            result = minimize(
                objective,
                initial_weights,
                method=self.solver,
                bounds=bounds,
                constraints=constraints,
                options=options
            )
        
        end_time = time.time()
        
        output = {
            'weights': result.x if result.success else initial_weights,
            'success': result.success,
            'status': result.message,
            'objective_value': result.fun,
            'execution_time': end_time - start_time,
            'iterations': result.nit if hasattr(result, 'nit') else None
        }
        
        return output
    
    def monte_carlo_presampling(self, expected_returns, cov_matrix, num_samples=10000, top_n=10):
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
    
    def optimize_large_portfolio(self, expected_returns, cov_matrix, use_monte_carlo=True, 
                                num_samples=10000, use_jac=True):
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
        return self.mean_variance_optimization(
            expected_returns, cov_matrix, use_jac=use_jac
        )

    def compute_portfolio_metrics(self, weights, returns, cov_matrix):
        """
        Compute portfolio metrics including return, risk, Sharpe ratio and concentration measures.
        
        Args:
            weights (ndarray): Portfolio weights
            returns (ndarray): Expected returns for each stock
            cov_matrix (ndarray): Covariance matrix of returns
            
        Returns:
            dict: Portfolio metrics
        """
        # Calculate portfolio expected return
        portfolio_return = np.dot(weights, returns)
        
        # Calculate portfolio risk (standard deviation)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
        
        # Calculate concentration metrics
        top5_concentration = np.sum(np.sort(weights)[-5:])
        herfindahl_index = np.sum(weights**2)
        
        return {
            'expected_return': portfolio_return,
            'risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio,
            'top5_concentration': top5_concentration,
            'herfindahl_index': herfindahl_index
        }