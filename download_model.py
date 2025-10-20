#!/usr/bin/env python3
"""
Download model weights for Watermark Anything Tool
"""

import os
import sys
import urllib.request
from pathlib import Path


def download_file(url, destination):
    """Download a file with progress bar"""
    def download_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        progress = int(50 * percent / 100)
        sys.stdout.write(f'\r[{"=" * progress}{" " * (50 - progress)}] {percent:.1f}%')
        sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, destination, reporthook=download_progress)
        print()  # New line after progress bar
        return True
    except Exception as e:
        print(f"\nError: {e}")
        return False


def main():
    """Main download function"""
    checkpoint_dir = Path("checkpoints")
    checkpoint_path = checkpoint_dir / "checkpoint.pth"
    
    # Check if already exists
    if checkpoint_path.exists():
        print("✓ Model weights already exist at:", checkpoint_path)
        return
    
    # Create directory if needed
    checkpoint_dir.mkdir(exist_ok=True)
    
    print("Watermark Anything Model Downloader")
    print("=" * 40)
    print("\nChoose which model to download:")
    print("1. MIT Licensed Model (trained on SA-1B dataset) - Recommended")
    print("2. Original Model (CC-BY-NC license, trained on COCO)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        url = "https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth"
        print("\n✓ Downloading MIT licensed model...")
    elif choice == "2":
        url = "https://dl.fbaipublicfiles.com/watermark_anything/wam_coco.pth"
        print("\n✓ Downloading original model (CC-BY-NC license)...")
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)
    
    print(f"URL: {url}")
    print(f"Destination: {checkpoint_path}")
    print("\nDownloading... (this may take a few minutes)")
    
    if download_file(url, checkpoint_path):
        print(f"\n✓ Successfully downloaded model to: {checkpoint_path}")
        
        # Verify file size
        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        print(f"✓ File size: {size_mb:.1f} MB")
        
        print("\nYou can now use the watermark tool!")
        print("Try: python watermark_tool.py --help")
    else:
        print("\n✗ Failed to download model weights")
        print("\nYou can manually download from:")
        print(f"  {url}")
        print(f"And save to: {checkpoint_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()