import torch
import torch.nn as nn
from config import device, DTYPE


class OU_PINN(nn.Module):
    """
    Physics-Informed Neural Network for Ornstein-Uhlenbeck process with Levy jumps.
    """
    
    def __init__(self, hidden_layers=3, neurons_per_layer=20):
        super(OU_PINN, self).__init__()
        layers = []
        
        # Input layer (t, x) -> size 2
        layers.append(nn.Linear(2, neurons_per_layer))
        layers.append(nn.Tanh())
        
        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(nn.Tanh())
        
        # Output layer -> size 1 (phi)
        layers.append(nn.Linear(neurons_per_layer, 1))
        layers.append(nn.Sigmoid())  # Ensures output is (0, 1)
        
        self.model = nn.Sequential(*layers)

    def forward(self, tx):
        """
        Forward pass through the network.
        
        Args:
            tx (torch.Tensor): Concatenated [t, x] tensor of shape [N, 2]
            
        Returns:
            torch.Tensor: Network output of shape [N, 1]
        """
        return self.model(tx) 