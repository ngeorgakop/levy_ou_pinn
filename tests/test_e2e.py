import subprocess
import sys
import os

def test_e2e_main_runs():
    # Adjust the path to your main script if needed
    main_script = os.path.join(os.path.dirname(__file__), '..', 'main.py')
    result = subprocess.run([sys.executable, main_script], capture_output=True, text=True)
    assert result.returncode == 0, f"Process failed: {result.stderr}"
    # Optionally, check for expected output
    # assert "Expected output" in result.stdout
