import numpy as np
import sys
import os
import plotly.graph_objects as go

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import k, theta, sigma, tmax, lambda_jump, jump_std


def simulate_levy_ou_path(t0, x0, t_max, dt, k, theta, sigma, lambda_jump, jump_std):
    """
    Simulates a single path of the Lévy-driven Ornstein-Uhlenbeck process.
    Returns the time points and the corresponding positions.
    """
    times = np.arange(t0, t_max + dt, dt)
    positions = np.zeros_like(times)
    positions[0] = x0

    sqrt_dt = np.sqrt(dt)

    for i in range(len(times) - 1):
        # Standard OU components
        dW = np.random.normal(0, sqrt_dt)
        dx = k * (theta - positions[i]) * dt + sigma * dW
        positions[i+1] = positions[i] + dx

        # Jump component (Compound Poisson Process)
        if np.random.uniform(0, 1) < lambda_jump * dt:
            jump = np.random.normal(0, jump_std)
            positions[i+1] += jump

    return times, positions


def plot_many_levy_ou_realizations():
    """
    Plots multiple realizations of the Lévy-driven OU process starting from x0=0 up until t=T=1.
    """
    # Parameters
    t0 = 0.0
    x0 = 0.0  # Starting from x0=0 as requested
    n_paths = 50
    dt = 0.001

    # tmax is imported from config, which should be 1.0

    print("=" * 50)
    print("Starting Lévy-driven OU Process Realizations Plot Generation")
    print(f"Parameters: t0={t0}, x0={x0}, n_paths={n_paths}, t_max={tmax}, dt={dt}")
    print(f"OU Parameters: k={k}, theta={theta}, sigma={sigma}")
    print(f"Jump Parameters: lambda={lambda_jump}, jump_std={jump_std}")
    print("=" * 50)

    fig = go.Figure()

    print(f"\nSimulating and plotting {n_paths} paths...")
    for i in range(n_paths):
        times, positions = simulate_levy_ou_path(t0, x0, tmax, dt, k, theta, sigma, lambda_jump, jump_std)
        fig.add_trace(go.Scatter(x=times, y=positions, mode='lines', line=dict(width=0.8), opacity=0.7, showlegend=False))
        if (i + 1) % 10 == 0:
            print(f"  ... plotted path {i + 1}/{n_paths}")

    # Plotting aesthetics
    fig.update_layout(
        title_text=f'{n_paths} Realizations of the Lévy-driven Ornstein-Uhlenbeck Process (x0={x0})',
        xaxis_title='Time (t)',
        yaxis_title='State ($X_t$)',
        template="plotly_white",
        height=700,
        width=1200
    )

    # Save the plot
    plot_filename = 'levy_ou_process_realizations.html'
    fig.write_html(plot_filename)
    print(f"\nPlot saved to {plot_filename}")


if __name__ == "__main__":
    plot_many_levy_ou_realizations() 