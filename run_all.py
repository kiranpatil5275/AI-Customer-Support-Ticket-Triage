"""
One-click script to generate data + train models.
Run this first before launching the Streamlit app.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd):
    print(f"\n>>> Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Error while running: {cmd}")
        sys.exit(1)

if __name__ == "__main__":
    base = Path(__file__).parent
    print("="*60)
    print("AI Ticket Triage - Setup Script")
    print("="*60)
    
    # Generate data
    run_command(f"{sys.executable} {base / 'src' / 'generate_data.py'}")
    
    # Train models
    run_command(f"{sys.executable} {base / 'src' / 'train_models.py'}")
    
    print("\n" + "="*60)
    print("✅ Setup complete!")
    print("="*60)
    print("\nNow run the Streamlit app with:")
    print("    streamlit run app.py")
