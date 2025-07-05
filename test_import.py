#!/usr/bin/env python3
"""
Test script to verify that all modules import correctly
"""

def test_imports():
    """Test that all modules can be imported without errors"""
    try:
        print("Testing imports...")
        
        # Test basic imports
        import torch
        print("✓ PyTorch imported successfully")
        
        # Test config import
        from config import device, DTYPE, k, theta, sigma
        print("✓ Config imported successfully")
        
        # Test model import
        from model import OU_PINN
        print("✓ Model imported successfully")
        
        # Test integration methods
        from integration import compute_levy_integral_mc, compute_levy_integral_trapz
        print("✓ Integration methods imported successfully")
        
        # Test residual computation
        from residual import get_r, fun_r
        print("✓ Residual functions imported successfully")
        
        # Test loss computation
        from loss import compute_loss
        print("✓ Loss function imported successfully")
        
        # Test training
        from training import train
        print("✓ Training function imported successfully")
        
        # Test data generation
        from data_generation import generate_training_data
        print("✓ Data generation function imported successfully")
        
        print("\n✅ All imports successful!")
        
        # Test basic model creation
        model = OU_PINN(hidden_layers=3, neurons_per_layer=10)
        print("✓ Model creation successful")
        
        # Test model forward pass
        test_input = torch.randn(10, 2)
        output = model(test_input)
        print(f"✓ Model forward pass successful. Output shape: {output.shape}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\n🎉 All tests passed! The modular structure is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the imports and dependencies.") 