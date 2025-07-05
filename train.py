import torch.optim as optim
from loss import compute_loss
from config import SCHEDULER_CONFIG, TRAINING_CONFIG

def train(model, X_r, X_data, u_data, k, theta, sigma, lambda_jump, jump_std, epochs=None, lr=None):
    """
    Train the PINN model.

    Args:
        model: Neural network model
        X_r: Residual/collocation points
        X_data: List of boundary/terminal condition points
        u_data: List of target values for boundary/terminal conditions
        k, theta, sigma: OU process parameters
        lambda_jump, jump_std: Jump parameters
        epochs: Number of training epochs for the Adam optimizer
        lr: Initial learning rate for the Adam optimizer

    Returns:
        list: Training loss history
    """
    # Use config defaults if not provided
    epochs = epochs or TRAINING_CONFIG['epochs']
    lr = lr or TRAINING_CONFIG['lr']
    
    loss_history = []
    
    # --- Phase 1: Adam Optimizer ---
    print("--- Starting Adam Optimization ---")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # A more robust learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=SCHEDULER_CONFIG['step_size'], 
        gamma=SCHEDULER_CONFIG['gamma']
    )

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = compute_loss(model, X_r, X_data, u_data, k, theta, sigma, lambda_jump, jump_std)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        loss_history.append(loss.item())

        if epoch % TRAINING_CONFIG['print_interval'] == 0 or epoch == epochs - 1:
            print(f"Adam Epoch {epoch}: Loss = {loss.item():.6f}, LR = {scheduler.get_last_lr()[0]:.6f}")

    return loss_history