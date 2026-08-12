import os
import subprocess

def main():
    configs = [
        'configs/unet.yaml',
        'configs/unet_plusplus.yaml',
        'configs/attention_unet.yaml',
        'configs/nnunet.yaml',
        'configs/segformer.yaml',
        'configs/unetformer.yaml',
        'configs/swin_unet.yaml'
    ]
    
    # Ensure we run from Fine Tuning directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)
    
    for config_path in configs:
        if not os.path.exists(config_path):
            print(f"Skipping {config_path} as it does not exist.")
            continue
            
        print(f"\n{'='*50}\nStarting Fine-Tuning with {config_path}\n{'='*50}\n")
        
        # Run finetune.py as a subprocess to keep memory clean between runs
        subprocess.run(["python", "scripts/finetune.py", "--config", config_path])
        
    print("\nCompleted all fine-tuning runs.")

if __name__ == "__main__":
    main()
