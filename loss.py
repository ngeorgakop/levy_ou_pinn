import torch
from residual import (get_r)


def compute_loss(model, X_r, X_data, u_data, k, theta, sigma, lambda_jump, jump_std):
    """
    Compute the total loss for training the PINN.
    
    Args:
        model: Neural network model
        X_r: Residual/collocation points
        X_data: List of boundary/terminal condition points
        u_data: List of target values for boundary/terminal conditions
        k, theta, sigma: OU process parameters
        lambda_jump, jump_std: Jump parameters
        
    Returns:
        torch.Tensor: Total loss
    """
    # Extract interior points for PDE residual
    t_r_interior, x_r_interior = X_r[:, 0:1], X_r[:, 1:2]

    # Compute PDE residual loss
    pde_residual = get_r(model, t_r_interior, x_r_interior, k, theta, sigma, lambda_jump, jump_std)
    loss_interior = torch.mean(pde_residual**2)

    # Terminal condition loss
    X_terminal_cond_points = X_data[0]
    u_terminal_target_values = u_data[0]
    u_terminal_pred = model(X_terminal_cond_points)
    loss_terminal = torch.mean((u_terminal_target_values - u_terminal_pred) ** 2)

    # Boundary condition loss
    X_boundary_cond_points = X_data[1]
    u_boundary_target_values = u_data[1]
    u_boundary_pred = model(X_boundary_cond_points)
    loss_boundary = torch.mean((u_boundary_target_values - u_boundary_pred) ** 2)

    # Total loss
    total_loss = loss_interior + loss_terminal + loss_boundary

    return total_loss 