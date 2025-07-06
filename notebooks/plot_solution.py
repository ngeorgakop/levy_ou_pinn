import torch
import numpy as np
import plotly.graph_objects as go
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import OU_PINN
from config import (
    device, DTYPE, xmin, xmax, tmin, tmax,
    HIDDEN_LAYERS, NEURONS_PER_LAYER, MODEL_PATH
)


def plot_3d_solution():
    """
    Loads a trained PINN model and plots its 3D solution surface using Plotly.
    """
    print("Loading model...")
    # Initialize model with the same architecture as during training
    model = OU_PINN(hidden_layers=HIDDEN_LAYERS, neurons_per_layer=NEURONS_PER_LAYER).to(device)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Model loaded successfully from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}.")
        print("Please run main.py to train and save the model first.")
        return
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return

    model.eval()  # Set the model to evaluation mode

    # Create a grid of points for plotting
    print("Generating grid for plotting...")
    num_points_x = 500
    num_points_t = 500
    x_vals = np.linspace(xmin, xmax, num_points_x)
    t_vals = np.linspace(tmin, tmax, num_points_t)
    X, T = np.meshgrid(x_vals, t_vals)

    # Prepare grid points for the model
    # Shape for model input should be [N, 2] where N = num_points_x * num_points_t
    tx_grid = np.stack([T.ravel(), X.ravel()], axis=-1)
    tx_tensor = torch.tensor(tx_grid, dtype=DTYPE, device=device)

    # Get model predictions
    print("Evaluating model on the grid...")
    with torch.no_grad():
        phi_pred_tensor = model(tx_tensor)

    # Reshape predictions back to grid shape
    phi_pred = phi_pred_tensor.cpu().numpy().reshape(X.shape)

    # Create the interactive 3D plot using Plotly
    print("Creating interactive 3D plot...")
    fig = go.Figure(data=[go.Surface(z=phi_pred, x=T, y=X, colorscale='Viridis')])

    fig.update_layout(
        title='Interactive PINN Solution for Default Probability',
        scene=dict(
            xaxis_title='Time (t)',
            yaxis_title='Asset Value (x)',
            zaxis_title='Default Probability (φ)'
        ),
        autosize=False,
        width=800,
        height=800,
        margin=dict(l=65, r=50, b=65, t=90)
    )

    # Save and show the plot
    fig.write_html("pinn_solution_interactive.html")
    print("Interactive plot saved to pinn_solution_interactive.html")
    fig.show()


if __name__ == "__main__":
    plot_3d_solution() 