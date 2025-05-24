"""
Main script for running the Datasheet Analyzer locally.
"""
import os
import sys
import torch
import subprocess
import platform
from pathlib import Path

def check_gpu():
    """Check GPU availability and print information."""
    print("Checking system information...")
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"PyTorch version: {torch.__version__}")
    
    if platform.system() == "Darwin":  # macOS
        if torch.backends.mps.is_available():
            print("MPS (Metal Performance Shaders) is available")
            print(f"Using device: {torch.backends.mps.get_device_name()}")
        else:
            print("MPS is not available, using CPU")
    else:  # Linux/Windows
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.is_available()}")
            print(f"Current GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        else:
            print("Warning: CUDA is not available. Running on CPU.")

def setup_environment():
    """Setup the Python environment."""
    # Add src directory to Python path
    src_path = Path(__file__).parent / "src"
    sys.path.append(str(src_path))

def run_streamlit():
    """Run the Streamlit application."""
    print("Starting Streamlit application...")
    streamlit_path = Path(__file__).parent / "src" / "app" / "streamlit_app.py"
    
    # Run Streamlit with specific port
    subprocess.run([
        "streamlit", "run",
        str(streamlit_path),
        "--server.port=8501"
    ])

def main():
    """Main function."""
    # Check GPU
    check_gpu()
    
    # Setup environment
    setup_environment()
    
    # Run Streamlit
    run_streamlit()

if __name__ == "__main__":
    main() 