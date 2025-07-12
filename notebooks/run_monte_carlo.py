import torch
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    k, theta, sigma, lambda_jump, jump_std,
    tmin, tmax, K_threshold, xmin, xmax,
    device, DTYPE
)
from integration import simulate_ou_paths


def run_monte_carlo_for_interior_points():
    """
    Runs a Monte Carlo simulation for 500 randomly sampled interior points.
    """
    num_points = 500
    num_sims_per_point = 1000  # Number of MC paths for each point
    num_steps_per_path = 100

    print("=" * 50)
    print("Starting Monte Carlo Simulation for Interior Points")
    print(f"Number of interior points: {num_points}")
    print(f"Simulations per point: {num_sims_per_point}")
    print("=" * 50)

    # Generate 500 random interior points (t, x)
    # t ~ U(tmin, tmax)
    # x ~ U(xmin, xmax)
    t_interior = torch.rand(num_points, 1, device=device, dtype=DTYPE) * (tmax - tmin) + tmin
    x_interior = torch.rand(num_points, 1, device=device, dtype=DTYPE) * (xmax - xmin) + xmin

    print("\nGenerated 500 random interior points.")

    # Run the simulation
    print("Running Monte Carlo simulation...")
    default_probabilities = simulate_ou_paths(
        start_x=x_interior,
        start_t=t_interior,
        k=k,
        theta=theta,
        sigma=sigma,
        K_threshold=K_threshold,
        t_max=tmax,
        num_sims=num_sims_per_point,
        num_steps=num_steps_per_path
    )
    print("Simulation complete.")

    # Calculate and display the average probability of default
    average_prob_of_default = torch.mean(default_probabilities)

    print("\n--- Results ---")
    print(f"Average probability of default across {num_points} interior points: {average_prob_of_default.item():.6f}")
    print("---------------")


if __name__ == "__main__":
    run_monte_carlo_for_interior_points() 