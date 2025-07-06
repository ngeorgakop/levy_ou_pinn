import torch
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    k, theta, sigma, lambda_jump, jump_std,
    tmax, K_threshold, xmin, xmax,
    HIDDEN_LAYERS, NEURONS_PER_LAYER, device, DTYPE,
    MODEL_PATH
)
from model import OU_PINN

def simulate_levy_ou_path(t0, x0, t_max, dt, k, theta, sigma, lambda_jump, jump_std):
    """
    Simulates a single path of the Lévy-driven Ornstein-Uhlenbeck process.
    """
    t = t0
    x = x0
    
    while t < t_max:
        # Standard OU components
        dW = np.random.normal(0, np.sqrt(dt))
        dx = k * (theta - x) * dt + sigma * dW
        x += dx

        # Jump component (Compound Poisson Process)
        if np.random.uniform(0, 1) < lambda_jump * dt:
            jump = np.random.normal(0, jump_std)
            x += jump
        
        t += dt
        
    return x

def monte_carlo_simulation(t0, x0, n_paths, t_max, dt, k, theta, sigma, lambda_jump, jump_std, K_thresh):
    """
    Runs the Monte Carlo simulation to estimate P(X_T <= K | X_t0 = x0).
    """
    paths_below_K = 0
    for i in range(n_paths):
        final_x = simulate_levy_ou_path(t0, x0, t_max, dt, k, theta, sigma, lambda_jump, jump_std)
        if final_x <= K_thresh:
            paths_below_K += 1
            
    return paths_below_K / n_paths

def compare_pinn_with_monte_carlo_plot():
    """
    Compares the PINN output with a Monte Carlo simulation across a range of x values and plots the result.
    """
    # Parameters
    t0 = 0.5
    n_points = 50
    n_paths = 5000  # Reduced for faster plotting
    dt = 0.001
    
    print("="*50)
    print("Starting PINN vs. Monte Carlo Plot Generation")
    print(f"Parameters: t0={t0}, n_points={n_points}, n_paths={n_paths}, dt={dt}")
    print("="*50)
    
    # Load trained model
    print(f"\nLoading model from {MODEL_PATH}...")
    model = OU_PINN(hidden_layers=HIDDEN_LAYERS, neurons_per_layer=NEURONS_PER_LAYER)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {MODEL_PATH}. Please train the model first by running main.py.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        sys.exit(1)
        
    model.to(device)
    model.eval()

    # Generate points for comparison
    x_values = np.linspace(0, xmax, n_points)
    pinn_predictions = []
    mc_estimates = []

    print(f"\nCalculating predictions for {n_points} points...")
    for i, x0 in enumerate(x_values):
        # Get PINN prediction
        with torch.no_grad():
            input_tensor = torch.tensor([[t0, x0]], dtype=DTYPE, device=device)
            pinn_prediction = model(input_tensor).item()
            pinn_predictions.append(pinn_prediction)

        # Get Monte Carlo estimation
        mc_estimate = monte_carlo_simulation(
            t0, x0, n_paths, tmax, dt, 
            k, theta, sigma, lambda_jump, jump_std, 
            K_threshold
        )
        mc_estimates.append(mc_estimate)
        print(f"  ({i+1}/{n_points}) x0={x0:.2f}: PINN={pinn_prediction:.4f}, MC={mc_estimate:.4f}")

    # Plotting the results
    print("\nPlotting results...")
    plt.figure(figsize=(12, 7))
    plt.plot(x_values, pinn_predictions, label='PINN Prediction', color='blue', linewidth=2)
    plt.plot(x_values, mc_estimates, 'o', label='Monte Carlo Estimate', color='red', markersize=5)
    plt.title(f'PINN vs. Monte Carlo Comparison at t={t0}')
    plt.xlabel('$x_0$ (Initial State)')
    plt.ylabel('$P(X_{t_{max}} \leq K)$')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save the plot
    plot_filename = 'pinn_vs_mc_comparison.png'
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    plt.close()

if __name__ == "__main__":
    compare_pinn_with_monte_carlo_plot() 