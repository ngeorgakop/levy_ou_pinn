import torch
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def mock_compute_loss(model, X_r, X_data, u_data, k, theta, sigma, lambda_jump, jump_std):
    """Mock compute_loss function for testing."""
    # Return a simple deterministic loss for consistent testing
    output = model(X_r)
    return torch.mean(output**2)

def test_training_numerical_consistency():
    """Test that training produces consistent numerical results."""
    # Since training.py now imports STEP_SIZE and GAMMA directly, we must mock them on the config mock.
    mock_config = MagicMock(
        STEP_SIZE=4000, 
        GAMMA=0.5
    )

    # We mock the entire 'loss' and 'config' modules before importing 'training'
    # This ensures 'training' gets our mocked versions.
    with patch.dict('sys.modules', {
        'loss': MagicMock(),
        'config': mock_config
    }):
        # We also patch print to avoid cluttering test output
        with patch('builtins.print'):
            # Import train function from the correct module
            from training import train
            
            # Now, we patch compute_loss inside the just-imported 'training' module
            with patch('training.compute_loss', side_effect=mock_compute_loss):
                # Set random seed for reproducibility
                torch.manual_seed(42)
                np.random.seed(42)
                
                # Create a simple mock model
                model1 = torch.nn.Sequential(
                    torch.nn.Linear(2, 10),
                    torch.nn.Tanh(),
                    torch.nn.Linear(10, 1)
                )
                
                # Save initial state
                initial_state = model1.state_dict()
                
                # Mock training data
                X_r = torch.randn(100, 2, requires_grad=True)
                X_data = [torch.randn(10, 2)]
                u_data = [torch.randn(10, 1)]
                
                # Parameters
                k, theta, sigma = 1.0, 0.0, 0.2
                lambda_jump, jump_std = 0.1, 0.1
                
                # Run training first time
                torch.manual_seed(123)  # Different seed for training randomness
                loss_history_1 = train(model1, X_r, X_data, u_data, k, theta, sigma, lambda_jump, jump_std, epochs=3)
                
                # Reset model to initial state
                model1.load_state_dict(initial_state)
                
                # Run training again with same conditions
                torch.manual_seed(123)  # Same seed for training randomness
                loss_history_2 = train(model1, X_r, X_data, u_data, k, theta, sigma, lambda_jump, jump_std, epochs=3)
                
                # Check that results are consistent
                assert len(loss_history_1) == len(loss_history_2)
                assert len(loss_history_1) == 3, f"Expected 3 epochs, got {len(loss_history_1)}"
                
                # Check that loss values are identical (deterministic mock)
                for i, (l1, l2) in enumerate(zip(loss_history_1, loss_history_2)):
                    assert abs(l1 - l2) < 1e-6, f"Loss values at epoch {i} differ: {l1} vs {l2}"
