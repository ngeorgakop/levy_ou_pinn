import os
import re
import argparse

def convert_relative_to_absolute(filepath):
    """Convert relative imports to absolute imports in the given file."""
    with open(filepath, 'r') as file:
        content = file.read()
    
    # Pattern to match relative imports
    # This will match "from .<something>" or "from ..<something>" etc.
    pattern = r'from\s+(\.*)([\w\.]+)\s+import'
    
    def replace_import(match):
        dots, module = match.groups()
        if dots:  # If it's a relative import
            return f'from {module} import'
        return match.group(0)  # Return unchanged if not relative
    
    new_content = re.sub(pattern, replace_import, content)
    
    # Write back to the file if changes were made
    if new_content != content:
        print(f"Converting imports in {filepath}")
        with open(filepath, 'w') as file:
            file.write(new_content)

def process_directory(directory_path):
    """Process all Python files in the given directory and its subdirectories."""
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                convert_relative_to_absolute(filepath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert relative imports to absolute imports')
    parser.add_argument('directory', nargs='?', default='.', 
                        help='Directory to process (default: current directory)')
    args = parser.parse_args()
    
    process_directory(args.directory)
    print("Conversion complete")
