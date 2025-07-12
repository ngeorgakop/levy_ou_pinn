import torch
import math
from config import DTYPE, device, lambda_jump, jump_std


def compute_levy_integral_mc(model, x, t, lambda_jump=1.0, jump_std=0.2, num_samples=1000):
    """
    Approximates the Levy integral using Monte Carlo within PyTorch.
    Integral: lambda * E[phi(t, x+Z) - phi(t, x)] where Z ~ N(0, jump_std^2)

    Args:
        model: The neural network model (takes concatenated [t, x] tensor).
        x (torch.Tensor): Input x coordinates (shape [N, 1]).
        t (torch.Tensor): Input t coordinates (shape [N, 1]).
        lambda_jump (float): Jump rate.
        jump_std (float): Standard deviation of Normal jumps.
        num_samples (int): Number of Monte Carlo samples per (x, t) point.

    Returns:
        torch.Tensor: Approximation of the integral term, shape (N, 1).
    """
    batch_size = x.shape[0]
    if batch_size == 0:  # Handle empty batch case
        return torch.zeros_like(x)

    # Generate random jump sizes Z ~ N(0, jump_std^2) directly with torch
    Z = torch.randn(batch_size, num_samples, 1, dtype=x.dtype, device=x.device) * jump_std

    # Prepare tiled/repeated x and t for batch evaluation
    x_rep = x.unsqueeze(1).expand(-1, num_samples, -1)
    t_rep = t.unsqueeze(1).expand(-1, num_samples, -1)

    # Coordinates after jump: x + Z
    x_plus_z = x_rep + Z

    # Prepare inputs for the model (concatenate t and x)
    input_post_jump = torch.cat((t_rep.reshape(-1, 1), x_plus_z.reshape(-1, 1)), dim=1)
    input_pre_jump = torch.cat((t, x), dim=1)  # Original points, shape [batch_size, 2]

    # Evaluate the model post-jump
    # model expects input shape [batch * num_samples, 2]
    phi_x_z_flat = model(input_post_jump)  # Use model directly
    # Reshape back: (batch_size, num_samples, 1)
    phi_x_z = phi_x_z_flat.reshape(batch_size, num_samples, 1)

    # Evaluate model at original points
    # model expects input shape [batch_size, 2]
    phi_x_eval = model(input_pre_jump)  # Use model directly
    # Expand shape: (batch_size, num_samples, 1) for broadcasting subtraction
    phi_x_expanded = phi_x_eval.unsqueeze(1).expand(-1, num_samples, -1)

    # Calculate the difference for each sample
    delta_phi = phi_x_z - phi_x_expanded

    # Average over the samples and multiply by jump rate lambda
    integral = lambda_jump * torch.mean(delta_phi, dim=1)

    return integral


def compute_levy_integral_trapz(model, x, t, lambda_jump=1.0, jump_std=0.2, num_steps=1000, integration_width=5.0):
    """
    Approximates the Levy integral using the trapezoidal rule (torch.trapz).
    Integral: lambda * ∫ [phi(t, x+z) - phi(t, x)] * p(z) dz
    where p(z) is the PDF of N(0, jump_std^2).

    Args:
        model: The neural network model (takes concatenated [t, x] tensor).
        x (torch.Tensor): Input x coordinates (shape [N, 1]).
        t (torch.Tensor): Input t coordinates (shape [N, 1]).
        lambda_jump (float): Jump rate.
        jump_std (float): Standard deviation of Normal jumps.
        num_steps (int): Number of points for the trapezoidal rule grid.
        integration_width (float): How many standard deviations to integrate over.
                                   The range will be [-width*std, width*std].

    Returns:
        torch.Tensor: Approximation of the integral term, shape (N, 1).
    """
    batch_size = x.shape[0]
    if batch_size == 0:
        return torch.zeros_like(x)

    # 1. Define the integration domain for z. Since z is normally distributed,
    # we can integrate over a finite interval (e.g., +/- 5 standard deviations)
    # instead of (-inf, inf) to capture most of the probability mass.
    z_limit = integration_width * jump_std
    z = torch.linspace(-z_limit, z_limit, steps=num_steps, dtype=x.dtype, device=x.device)  # Shape: [num_steps]

    # 2. Calculate the Normal distribution's Probability Density Function (PDF) for each z.
    pdf_z = (1 / (jump_std * math.sqrt(2 * math.pi))) * torch.exp(-0.5 * (z / jump_std)**2)

    # 3. Prepare tensors for batch evaluation by broadcasting.
    # We need to evaluate the model for each point in the batch (x, t) at each point z in our grid.
    x_rep = x.unsqueeze(1)          # Shape: [batch_size, 1, 1]
    t_rep = t.unsqueeze(1)          # Shape: [batch_size, 1, 1]
    z_rep = z.view(1, -1, 1)        # Shape: [1, num_steps, 1]

    # 4. Evaluate phi(t, x + z) for all combinations.
    x_plus_z = x_rep + z_rep        # Broadcasting results in shape: [batch_size, num_steps, 1]
    t_expanded = t_rep.expand(-1, num_steps, -1)  # Shape: [batch_size, num_steps, 1]

    # Concatenate for model input and flatten for batch processing
    input_post_jump = torch.cat((t_expanded, x_plus_z), dim=2)
    input_post_jump_flat = input_post_jump.view(-1, 2)  # Shape: [batch_size * num_steps, 2]

    phi_x_z_flat = model(input_post_jump_flat)
    phi_x_z = phi_x_z_flat.view(batch_size, num_steps, 1)  # Reshape back: [batch_size, num_steps, 1]

    # 5. Evaluate phi(t, x) at the original points.
    input_pre_jump = torch.cat((t, x), dim=1)
    phi_x = model(input_pre_jump)  # Shape: [batch_size, 1]
    phi_x_expanded = phi_x.unsqueeze(1)  # Shape: [batch_size, 1, 1] for broadcasting

    # 6. Construct the complete integrand: [phi(t, x+z) - phi(t, x)] * p(z)
    delta_phi = phi_x_z - phi_x_expanded  # Broadcasting subtraction

    # Reshape pdf_z for broadcasting with delta_phi
    integrand = delta_phi * pdf_z.view(1, -1, 1)  # Shape: [batch_size, num_steps, 1]
    integrand_squeezed = integrand.squeeze(-1)     # Shape: [batch_size, num_steps]

    # 7. Perform the integration using the trapezoidal rule along the z-axis (dim=1).
    integral_val = torch.trapz(integrand_squeezed, z, dim=1)  # Result shape: [batch_size]

    # 8. Multiply by lambda and ensure correct output shape.
    result = lambda_jump * integral_val.unsqueeze(-1)  # Shape: [batch_size, 1]

    return result


def simulate_ou_paths(start_x, start_t, k, theta, sigma, K_threshold,
                        t_max=1.0, num_sims=5000, num_steps=100):
    """
    Simulates Ornstein-Uhlenbeck paths with jumps to estimate terminal probability.

    Estimates P(X_T <= K | X_t = x) via Monte Carlo for a jump-diffusion process.
    The process is dX_t = k * (theta - X_t)dt + sigma * dW_t + dJ_t,
    where J_t is a compound Poisson process with intensity lambda_jump and normal jumps N(0, jump_std^2).

    This function computes the probability that the process is below the threshold at the FINAL time T,
    not the first passage time probability.

    Args:
        start_x (torch.Tensor): Starting x positions. Shape [N, 1].
        start_t (torch.Tensor): Starting t positions. Shape [N, 1].
        k, theta, sigma (float): OU process parameters.
        K_threshold (float): Terminal threshold.
        t_max (float): End time of the simulation.
        num_sims (int): Number of Monte Carlo simulations per starting point.
        num_steps (int): Number of time steps in each simulation path.

    Returns:
        torch.Tensor: Estimated terminal probabilities P(X_T <= K). Shape [N, 1].
    """
    batch_size = start_x.shape[0]
    if batch_size == 0:
        return torch.empty(0, 1, device=start_x.device, dtype=DTYPE)

    # Time steps are different for each starting time
    dt = (t_max - start_t) / num_steps  # Shape: [N, 1]
    dt_expanded = dt.unsqueeze(1).expand(-1, num_sims, -1) # Shape: [N, num_sims, 1]

    # Initialize paths at their starting values
    X_t = start_x.unsqueeze(1).expand(-1, num_sims, -1)  # Shape: [N, num_sims, 1]

    # Euler-Maruyama method for SDE simulation
    for _ in range(num_steps):
        # Generate random noise for this step (Brownian motion)
        dW = torch.randn_like(X_t) * torch.sqrt(dt_expanded)

        # Generate jumps for this step (Compound Poisson process)
        # 1. Number of jumps in dt follows a Poisson distribution
        num_jumps = torch.poisson(lambda_jump * dt_expanded)
        # 2. Total jump size is sum of N(0, jump_std^2) random variables
        jump_sizes = torch.randn_like(X_t) * jump_std * torch.sqrt(num_jumps)

        # Update paths with OU dynamics and jumps
        # dX = k * (theta - X_t) * dt + sigma * dW + dJ
        dX = k * (theta - X_t) * dt_expanded + sigma * dW + jump_sizes
        X_t = X_t + dX

    # Check final state: probability that X_T <= K_threshold
    # Only care about the final value, not whether threshold was hit during simulation
    final_below_threshold = (X_t <= K_threshold)
    terminal_prob = torch.sum(final_below_threshold, dim=1, dtype=DTYPE) / num_sims

    return terminal_prob 