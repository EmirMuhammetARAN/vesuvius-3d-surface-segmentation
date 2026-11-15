import os
import sys
from pathlib import Path

def get_target_dir():
    target = Path(__file__).resolve().parent.parent / "dataset"
    target.mkdir(parents=True, exist_ok=True)
    return target

def download():
    import kagglehub
    from kagglehub.config import get_kaggle_credentials
    
    target_dir = get_target_dir()
    creds = get_kaggle_credentials()
    
    if not creds and not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        print("[!] Kaggle credentials not found.")
        print("[!] You can:")
        print("    1. Place 'kaggle.json' inside '%USERPROFILE%\\.kaggle\\kaggle.json'")
        print("    2. Or set KAGGLE_USERNAME and KAGGLE_KEY environment variables.")
        print("    3. Or authenticate via kagglehub.login() below:")
        try:
            kagglehub.login()
        except Exception as e:
            print(f"[!] Authentication skipped or failed: {e}")
            return
            
    print(f"[*] Starting download to: {target_dir}")
    print("[*] Competition: vesuvius-challenge-surface-detection (27.3 GB)")
    try:
        path = kagglehub.competition_download(
            "vesuvius-challenge-surface-detection",
            output_dir=str(target_dir)
        )
        print(f"[+] Download successful! Files saved to: {path}")
    except Exception as e:
        print(f"[X] Download failed: {e}")

if __name__ == "__main__":
    download()
