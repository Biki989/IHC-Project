import subprocess
import sys
import os

scripts = [
    "scripts/train_unet_plusplus.py",
    "scripts/train_attention_unet.py",
    "scripts/train_nnunet.py"
]

def run_scripts():
    for script in scripts:
        print(f"\n{'='*50}\nStarting {script}\n{'='*50}\n")
        try:
            # Run the training script
            subprocess.run([sys.executable, script], check=True)
            print(f"\n{script} completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"\nError running {script}: {e}")
            sys.exit(1)

if __name__ == "__main__":
    run_scripts()
