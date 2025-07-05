import sys
import os
import unittest
from unittest.mock import patch

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import main

class TestEndToEnd(unittest.TestCase):
    @patch('main.plt.show')  # Mock plt.show() to prevent blocking
    @patch('main.EPOCHS', 1)  # Patch EPOCHS where it's used (in main.py)
    @patch('data_generation.N_r', 10)  # Patch data generation params where they are used
    @patch('data_generation.N_b', 10)
    @patch('data_generation.N_T', 10)
    @patch('residual.INTEGRATION_NUM_STEPS', 5)  # Patch integration steps where it is used
    def test_main_runs_without_errors(self, *mocked_args):
        """
        Test that the main function runs to completion without raising errors.
        """
        print("Running end-to-end test...")
        try:
            model, loss_history = main()
            
            # Basic assertions to ensure the process ran
            self.assertIsNotNone(model, "Model should not be None")
            self.assertIsInstance(loss_history, list, "Loss history should be a list")
            self.assertEqual(len(loss_history), 1, "Loss history should have one entry for one epoch")
            print("End-to-end test completed successfully.")
            
        except Exception as e:
            self.fail(f"main() raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
