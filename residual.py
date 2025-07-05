import torch
from integration import compute_levy_integral_trapz
from config import INTEGRATION_NUM_STEPS, INTEGRATION_WIDTH


def fun_r(t, x, k, theta, sigma, model, phi_t, phi_x, phi_xx, lambda_jump, jump_std):
    """
    Compute the PDE residual function.
    
    Args:
        t, x: Time and space coordinates
        k, theta, sigma: OU process parameters
        model: Neural network model
        phi_t, phi_x, phi_xx: Partial derivatives
        lambda_jump, jump_std: Jump parameters
        
    Returns:
        torch.Tensor: PDE residual
    """
    # Compute integral term using trapezoidal rule
    integral_term = compute_levy_integral_trapz(
        model, x, t, 
        lambda_jump=lambda_jump, 
        jump_std=jump_std, 
        num_steps=INTEGRATION_NUM_STEPS, 
        integration_width=INTEGRATION_WIDTH
    )

    # Calculate the PDE residual
    residual = phi_t + k * (theta - x) * phi_x + 0.5 * sigma**2 * phi_xx + integral_term
    return residual


def get_r(model, t, x, k, theta, sigma, lambda_jump, jump_std):
    """
    Compute PDE residuals with automatic differentiation.
    
    Args:
        model: Neural network model
        t, x: Time and space coordinates (require gradients)
        k, theta, sigma: OU process parameters
        lambda_jump, jump_std: Jump parameters
        
    Returns:
        torch.Tensor: PDE residual
    """
    # Ensure inputs require gradients
    t.requires_grad_(True)
    x.requires_grad_(True)

    # Concatenate inputs for the model
    tx = torch.cat([t, x], dim=1)
    phi = model(tx)  # Forward pass to get phi for derivative calculation

    # Compute gradients using autograd
    phi_t = torch.autograd.grad(
        phi, t, 
        grad_outputs=torch.ones_like(phi), 
        create_graph=True
    )[0]
    
    phi_x = torch.autograd.grad(
        phi, x, 
        grad_outputs=torch.ones_like(phi), 
        create_graph=True
    )[0]
    
    phi_xx = torch.autograd.grad(
        phi_x, x, 
        grad_outputs=torch.ones_like(phi_x), 
        create_graph=True
    )[0]

    # Calculate residual using the PDE function
    return fun_r(t, x, k, theta, sigma, model, phi_t, phi_x, phi_xx, lambda_jump, jump_std) 