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
    training_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Compute loss
        total_loss, loss_interior, loss_terminal, loss_boundary, loss_boundary_mc = compute_loss(model, X_r, X_data, u_data, k, theta, sigma, lambda_jump, jump_std)
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # Record loss
        lr_current = scheduler.get_last_lr()[0]
        training_history.append({
            'epoch': epoch,
            'total_loss': total_loss.item(),
            'loss_interior': loss_interior.item(),
            'loss_terminal': loss_terminal.item(),
            'loss_boundary': loss_boundary.item(),
            'loss_boundary_mc': loss_boundary_mc.item(),
            'lr': lr_current
        })
        
        # Print progress
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: Loss = {total_loss.item():.6f}, LR = {lr_current:.6f}")

    return training_history 