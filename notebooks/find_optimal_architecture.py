import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
from model import OU_PINN
from data_generation import get_training_data
from training import train
from integration import simulate_ou_paths
from config import device, k, theta, sigma, lambda_jump, jump_std, LEARNING_RATE, K_threshold

# --- Configuration for the architecture search ---

# 1. Define the architectures to test: (hidden_layers, neurons_per_layer)
ARCHITECTURES_TO_TEST = [
    (2, 20),   # Small
    (3, 20),
    (4, 20),
    (3, 30),   # Medium
    (4, 30),
    (6, 30),   # Larger (like the current default)
    (4, 50),   # Wider
    (8, 30)    # Deeper
]

# 2. Number of epochs to train each architecture
# This should be low enough for a quick test.
EPOCHS_PER_RUN = 2000 # You can adjust this value

# --- Main execution ---

def find_best_architecture():
    """
    Trains multiple network architectures and reports the one with the lowest final loss.
    """
    print("Starting architecture search...")
    print("=" * 60)
    print(f"Each architecture will be trained for {EPOCHS_PER_RUN} epochs.")

    # Generate a single, consistent dataset for all runs
    print("\\nGenerating a consistent training dataset...")
    X_r, X_data, u_data, X_mc = get_training_data(use_saved_data=False, save_generated_data=False)
    print("Dataset generated successfully.")

    results = []

    # Loop over all defined architectures
    for i, (layers, neurons) in enumerate(ARCHITECTURES_TO_TEST):
        print("\\n" + "=" * 60)
        print(f"Testing architecture {i+1}/{len(ARCHITECTURES_TO_TEST)}: {layers} hidden layers, {neurons} neurons/layer")

        # Initialize the model with the current architecture
        model = OU_PINN(hidden_layers=layers, neurons_per_layer=neurons).to(device)

        # Display model parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total model parameters: {total_params:,}")

        start_time = time.time()

        # Pre-calculate Monte Carlo targets for the anchor points
        u_mc_targets = simulate_ou_paths(
            start_x=X_mc[:, 1:2],
            start_t=X_mc[:, 0:1],
            k=k, theta=theta, sigma=sigma,
            K_threshold=K_threshold
        )

        # Train the model
        # Note: The 'train' function prints its own progress.
        training_history = train(
            model, X_r, X_data, u_data, X_mc, u_mc_targets,
            k, theta, sigma, lambda_jump, jump_std, K_threshold,
            epochs=EPOCHS_PER_RUN, lr=LEARNING_RATE
        )

        end_time = time.time()
        training_duration = end_time - start_time

        # Get the final loss from the training history
        final_loss = training_history[-1]['total_loss']
        print(f"Architecture training completed in {training_duration:.2f}s. Final loss: {final_loss:.6f}")

        # Store results for this architecture
        results.append({
            'layers': layers,
            'neurons': neurons,
            'params': total_params,
            'final_loss': final_loss,
            'duration': training_duration
        })

    # --- Report the final results ---
    print("\\n" + "=" * 60)
    print("Architecture Search Complete. Results:")
    print("-" * 60)
    print(f"{'Layers':<10} {'Neurons':<10} {'Parameters':<15} {'Final Loss':<15} {'Duration (s)':<15}")
    print("-" * 60)

    # Sort results by final loss to find the best
    sorted_results = sorted(results, key=lambda x: x['final_loss'])

    for res in sorted_results:
        params_str = f"{res['params']:,}"
        loss_str = f"{res['final_loss']:.6f}"
        duration_str = f"{res['duration']:.2f}"
        print(f"{res['layers']:<10} {res['neurons']:<10} {params_str:<15} {loss_str:<15} {duration_str}")

    print("-" * 60)

    best_architecture = sorted_results[0]
    print(f"\\nBest performing architecture found:")
    print(f"  - Layers: {best_architecture['layers']}")
    print(f"  - Neurons per layer: {best_architecture['neurons']}")
    print(f"  - Final Loss: {best_architecture['final_loss']:.6f}")

if __name__ == "__main__":
  find_best_architecture() 