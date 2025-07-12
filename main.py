import torch
# from . import __version__
# print(f"Using Lévy-driven OU Process PINN version: {__version__}")
import matplotlib.pyplot as plt
import numpy as np
import csv
from config import (
    device, k, theta, sigma, lambda_jump, jump_std,
    EPOCHS, LEARNING_RATE, HIDDEN_LAYERS, NEURONS_PER_LAYER,
    MODEL_PATH, N_r, K_threshold, lb, ub, N_mc, MC_RECALC_INTERVAL,
    EARLY_STOP_THRESHOLD
)
from model import OU_PINN
from data_generation import generate_training_data
from training import train
from integration import simulate_ou_paths


def main():
    """
    Main execution function for training the Lévy-driven OU process PINN.
    """
    print("Starting Lévy-driven OU Process PINN Training")
    print("=" * 50)
    
    # Generate training data
    print("\nGenerating training data...")
    X_r, X_data, u_data, X_mc = generate_training_data()

    # Pre-calculate Monte Carlo targets for the anchor points
    print("\nPre-calculating Monte Carlo targets for anchor points...")
    u_mc_targets = simulate_ou_paths(
        start_x=X_mc[:, 1:2],
        start_t=X_mc[:, 0:1],
        k=k, theta=theta, sigma=sigma,
        K_threshold=K_threshold
    )
    print("Monte Carlo simulation complete.")
    
    # Initialize model
    print(f"\nInitializing model with {HIDDEN_LAYERS} hidden layers and {NEURONS_PER_LAYER} neurons per layer...")
    model = OU_PINN(hidden_layers=HIDDEN_LAYERS, neurons_per_layer=NEURONS_PER_LAYER).to(device)
    
    # Display model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")
    
    # Train the model
    print(f"\nStarting training for {EPOCHS} epochs...")
    training_history = train(
        model, X_r, X_data, u_data, X_mc, u_mc_targets,
        k, theta, sigma, lambda_jump, jump_std, 
        K_threshold=K_threshold,
        epochs=EPOCHS, 
        lr=LEARNING_RATE,
        mc_recalc_interval=MC_RECALC_INTERVAL,
        early_stop_threshold=EARLY_STOP_THRESHOLD
    )
    
    print("Training completed!")
    
    # Save the model
    print(f"\nSaving model to {MODEL_PATH}...")
    try:
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model state_dict saved successfully to {MODEL_PATH}")
    except Exception as e:
        print(f"Error saving model: {e}")
    
    # Extract loss histories for plotting
    total_loss_history = [d['total_loss'] for d in training_history]
    loss_sum_history = [d['loss_sum'] for d in training_history]
    epochs = [d['epoch'] for d in training_history]

    # Plot and save training loss
    print("\nPlotting and saving training loss...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot total loss
    ax1.plot(epochs, total_loss_history, 'b-', linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Total Loss")
    ax1.set_title("Total Loss During Training")
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Plot loss sum with threshold line
    ax2.plot(epochs, loss_sum_history, 'r-', linewidth=2, label='Loss Sum')
    ax2.axhline(y=EARLY_STOP_THRESHOLD, color='g', linestyle='--', linewidth=2, 
                label=f'Early Stop Threshold ({EARLY_STOP_THRESHOLD})')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss Sum")
    ax2.set_title("Loss Sum (All Components)")
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('loss_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Enhanced loss plot saved to loss_plot.png")

    # Save detailed training history to CSV
    print("\nSaving detailed training history to CSV...")
    csv_file = 'training_log.csv'
    
    if not training_history:
        print("Warning: Training history is empty. Cannot save CSV.")
        return model, training_history

    # Dynamically get headers from the first history entry
    csv_columns = training_history[0].keys()
    
    try:
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            writer.writerows(training_history)
        print(f"Training history saved to {csv_file}")
    except IOError as e:
        print(f"I/O error while writing to {csv_file}: {e}")

    # Final loss reporting
    if training_history:
        final_total_loss = training_history[-1]['total_loss']
        final_loss_sum = training_history[-1]['loss_sum']
        print(f"\nFinal training metrics:")
        print(f"  Final total loss: {final_total_loss:.6f}")
        print(f"  Final loss sum: {final_loss_sum:.6f}")
        if final_loss_sum < EARLY_STOP_THRESHOLD:
            print(f"  ✅ Target achieved! Loss sum below threshold ({EARLY_STOP_THRESHOLD})")
        else:
            print(f"  ⚠️  Target not reached. Threshold is {EARLY_STOP_THRESHOLD}")
    
    return model, training_history


if __name__ == "__main__":
    main() 