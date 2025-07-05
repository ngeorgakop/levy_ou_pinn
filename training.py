import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from loss import compute_loss
from config import STEP_SIZE, GAMMA


def train(model, X_r, X_data, u_data, k, theta, sigma, lambda_jump, jump_std, epochs=1000, lr=0.001):
    """
    Train the PINN model.
    
    Args:
        model: Neural network model
        X_r: Residual/collocation points
        X_data: List of boundary/terminal condition points
        u_data: List of target values for boundary/terminal conditions
        k, theta, sigma: OU process parameters
        lambda_jump, jump_std: Jump parameters
        epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        list: Training loss history
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    loss_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Compute loss
        loss = compute_loss(model, X_r, X_data, u_data, k, theta, sigma, lambda_jump, jump_std)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Record loss
        loss_history.append(loss.item())
        
        # Print progress
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}, LR = {scheduler.get_last_lr()[0]:.6f}")

    return loss_history 