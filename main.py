import torch
# from . import __version__
# print(f"Using Lévy-driven OU Process PINN version: {__version__}")
import matplotlib.pyplot as plt
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
    loss_history = train(
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
    
    # Plot training loss
    print("\nPlotting training loss...")
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss for Lévy-Driven OU Process (PINN)")
    plt.yscale('log')  # Log scale helps visualize loss progression
    plt.grid(True)
    plt.show()
    
    # Final loss
    final_loss = loss_history[-1]
    print(f"\nFinal training loss: {final_loss:.6f}")
    
    return model, loss_history


if __name__ == "__main__":
    model, loss_history = main() 