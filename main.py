import torch
# from . import __version__
# print(f"Using Lévy-driven OU Process PINN version: {__version__}")
import matplotlib.pyplot as plt
import numpy as np
import csv
from config import (
    device, k, theta, sigma, lambda_jump, jump_std,
    EPOCHS, LEARNING_RATE, HIDDEN_LAYERS, NEURONS_PER_LAYER,
    MODEL_PATH
)
from model import OU_PINN
from data_generation import generate_training_data
from training import train


def main():
    """
    Main execution function for training the Lévy-driven OU process PINN.
    """
    print("Starting Lévy-driven OU Process PINN Training")
    print("=" * 50)
    
    # Generate training data
    print("\nGenerating training data...")
    X_r, X_data, u_data = generate_training_data()
    
    # Initialize model
    print(f"\nInitializing model with {HIDDEN_LAYERS} hidden layers and {NEURONS_PER_LAYER} neurons per layer...")
    model = OU_PINN(hidden_layers=HIDDEN_LAYERS, neurons_per_layer=NEURONS_PER_LAYER).to(device)
    
    # Display model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")
    
    # Train the model
    print(f"\nStarting training for {EPOCHS} epochs...")
    training_history = train(
        model, X_r, X_data, u_data, 
        k, theta, sigma, lambda_jump, jump_std, 
        epochs=EPOCHS, lr=LEARNING_RATE
    )
    
    print("Training completed!")
    
    # Save the model
    print(f"\nSaving model to {MODEL_PATH}...")
    try:
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model state_dict saved successfully to {MODEL_PATH}")
    except Exception as e:
        print(f"Error saving model: {e}")
    
    # Extract total loss for plotting
    total_loss_history = [d['total_loss'] for d in training_history]

    # Plot and save training loss
    print("\nPlotting and saving training loss...")
    plt.figure(figsize=(10, 6))
    plt.plot(total_loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss for Lévy-Driven OU Process (PINN)")
    plt.yscale('log')
    plt.grid(True)
    plt.savefig('loss_plot.png')
    plt.close()
    print("Loss plot saved to loss_plot.png")

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

    # Final loss
    if total_loss_history:
        final_loss = total_loss_history[-1]
        print(f"\nFinal training loss: {final_loss:.6f}")
    
    return model, training_history


if __name__ == "__main__":
    main() 