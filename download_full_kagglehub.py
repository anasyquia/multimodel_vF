#!/usr/bin/env python3
"""
Download the full Amazon Product Dataset 2020 using kagglehub.
Based on professor's provided code.
"""

import os
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

def download_full_amazon_dataset():
    """Download the complete Amazon Product Dataset 2020 using kagglehub."""
    
    print("🔄 Starting download of full Amazon Product Dataset 2020...")
    print("📊 This dataset contains ~700k products and may take several minutes to download.")
    
    try:
        # Set the path to the file you'd like to load (empty string loads all files)
        file_path = ""
        
        print("⬇️ Downloading dataset...")
        
        # Load the latest version of the full dataset
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "promptcloud/amazon-product-dataset-2020",
            file_path
        )
        
        print("✅ Download completed successfully!")
        
        # Display basic information about the dataset
        print(f"\n📊 Dataset Information:")
        print(f"  - Total products: {len(df):,}")
        print(f"  - Columns: {len(df.columns)}")
        print(f"  - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        print(f"\n📋 Columns available:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:2d}. {col}")
        
        print(f"\n🔍 First 5 records:")
        print(df.head())
        
        print(f"\n💾 Saving to CSV for processing...")
        # Save to CSV for your existing pipeline
        output_file = "data_full/amazon_products_full.csv"
        os.makedirs("data_full", exist_ok=True)
        df.to_csv(output_file, index=False)
        
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"✅ Saved to: {output_file} ({file_size_mb:.1f} MB)")
        
        # Show data quality overview
        print(f"\n🔍 Data Quality Overview:")
        for col in df.columns:
            non_null_count = df[col].notna().sum()
            null_percentage = (len(df) - non_null_count) / len(df) * 100
            print(f"  {col}: {non_null_count:,} non-null ({null_percentage:.1f}% missing)")
        
        return df, output_file
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check your internet connection")
        print("2. Make sure you have enough disk space (~2-3 GB)")
        print("3. Try again - sometimes network issues cause timeouts")
        return None, None

if __name__ == "__main__":
    df, file_path = download_full_amazon_dataset()
    
    if df is not None:
        print(f"\n🎉 Success! You now have the full Amazon dataset with {len(df):,} products!")
        print(f"📁 File saved to: {file_path}")
        print(f"\n🚀 Next steps:")
        print("1. Update your setup_data.py to use this new file")
        print("2. Run the preprocessing to generate embeddings")
        print("3. Enjoy your 700k product multimodal RAG system!")
    else:
        print("\n💔 Download failed. Please try again or check the troubleshooting steps above.") 