import torch
import torch.optim as optim
from loss import compute_loss
from integration import simulate_ou_paths


def train(model, X_r, X_data, u_data, X_mc, u_mc_targets, k, theta, sigma, lambda_jump, jump_std, K_threshold, epochs=1000, lr=0.001, mc_recalc_interval=None, early_stop_threshold=0.000099):
    """
    Train the PINN model with early stopping.
    
    Args:
        model: Neural network model
        X_r: Residual/collocation points
        X_data: List of boundary/terminal condition points
        u_data: List of target values for boundary/terminal conditions
        X_mc: Monte Carlo anchor points
        u_mc_targets: Pre-calculated Monte Carlo targets
        k, theta, sigma: OU process parameters
        lambda_jump, jump_std: Jump parameters
        K_threshold (float): Default threshold for the simulation.
        epochs: Number of training epochs
        lr: Learning rate
        mc_recalc_interval (int, optional): Interval to recalculate MC targets. Defaults to None.
        early_stop_threshold (float): Stop training when loss sum goes below this value.
        
    Returns:
        list: Training loss history
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    training_history = []

    print(f"Starting training...")
    print(f"Early stopping threshold (loss sum): {early_stop_threshold}")

    for epoch in range(epochs):
        # Periodically recalculate Monte Carlo targets
        if mc_recalc_interval and epoch > 0 and epoch % mc_recalc_interval == 0:
            print(f"\n--- Epoch {epoch}: Recalculating Monte Carlo targets... ---")
            u_mc_targets = simulate_ou_paths(
                start_x=X_mc[:, 1:2],
                start_t=X_mc[:, 0:1],
                k=k, theta=theta, sigma=sigma,
                K_threshold=K_threshold
            )
            print("--- Monte Carlo targets recalculated. Resuming training... ---\n")

        optimizer.zero_grad()
        
        # Compute loss
        total_loss, loss_interior, loss_terminal, loss_boundary, loss_mc = compute_loss(
            model, X_r, X_data, u_data, X_mc, u_mc_targets, k, theta, sigma, lambda_jump, jump_std,
            epoch=epoch
        )
        
        # Calculate sum of all loss components
        loss_sum = (loss_interior.item() + loss_terminal.item() + 
                   loss_boundary.item() + loss_mc.item())
        
        # Backward pass
        total_loss.backward()
        optimizer.step()

        # Record loss
        training_history.append({
            'epoch': epoch,
            'total_loss': total_loss.item(),
            'loss_interior': loss_interior.item(),
            'loss_terminal': loss_terminal.item(),
            'loss_boundary': loss_boundary.item(),
            'loss_mc': loss_mc.item(),
            'loss_sum': loss_sum,
            'lr': lr
        })
        
        # Check for early stopping
        if loss_sum < early_stop_threshold:
            print(f"\n🎉 Early stopping at epoch {epoch}!")
            print(f"Loss sum ({loss_sum:.6f}) below threshold ({early_stop_threshold})")
            break
        
        # Print progress with loss component breakdown
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: Total Loss = {total_loss.item():.6f} | "
                  f"Interior = {loss_interior.item():.6f} | "
                  f"Terminal = {loss_terminal.item():.6f} | "
                  f"Boundary = {loss_boundary.item():.6f} | "
                  f"MC = {loss_mc.item():.6f} | "
                  f"Loss Sum = {loss_sum:.6f}")

    print(f"\nTraining completed! Final metrics:")
    print(f"Final loss sum: {loss_sum:.6f}")
    
    return training_history 