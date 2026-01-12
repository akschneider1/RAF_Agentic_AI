
#!/usr/bin/env python3
"""
Setup script for training dependencies
"""

import subprocess
import sys

def install_packages():
    """Install required packages for training"""
    packages = [
        "torch",
        "transformers==4.30.0",
        "datasets",
        "seqeval",
        "accelerate",
        "scikit-learn",
        "matplotlib",
        "seaborn"
    ]
    
    print("Installing training dependencies...")
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("✅ All dependencies installed!")

if __name__ == "__main__":
    install_packages()
