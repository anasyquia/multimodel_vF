#!/usr/bin/env python3
"""
Explore the Amazon Product Dataset 2020 to see what files are available.
"""

import kagglehub

def explore_dataset():
    """Explore the dataset structure without downloading."""
    
    print("🔍 Exploring Amazon Product Dataset 2020 structure...")
    
    try:
        # Download the dataset files (this will show us what's available)
        path = kagglehub.dataset_download("promptcloud/amazon-product-dataset-2020")
        
        print(f"✅ Dataset downloaded to: {path}")
        
        # List all files in the dataset
        import os
        print(f"\n📋 Files in dataset:")
        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                print(f"  - {file} ({file_size:.1f} MB)")
        
        return path
        
    except Exception as e:
        print(f"❌ Error exploring dataset: {e}")
        return None

if __name__ == "__main__":
    path = explore_dataset()
    if path:
        print(f"\n🎉 Dataset available at: {path}")
        print("Now we can load the specific CSV files!")
    else:
        print("💔 Failed to explore dataset.") 