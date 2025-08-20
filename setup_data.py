#!/usr/bin/env python3
"""
Setup script to pre-process Amazon product data and create FAISS index with CLIP embeddings.
Run this once to prepare your data, then the Streamlit app can load pre-computed artifacts.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

def setup_data(csv_path: str = "real_amazon_data.csv", 
               clip_model: str = "clip-ViT-B-32",
               subsample: Optional[int] = None,
               force_rebuild: bool = False):
    """
    Setup the multimodal RAG data artifacts.
    
    Args:
        csv_path: Path to Amazon products CSV
        clip_model: CLIP model to use for embeddings
        subsample: Optional subsample size for testing
        force_rebuild: Force rebuild even if artifacts exist
    """
    
    print("🛍️ Amazon Product RAG Data Setup")
    print("=" * 50)
    
    # Check if artifacts already exist
    artifacts_dir = Path("artifacts")
    required_files = [
        "products.parquet",
        "text_emb.npy", 
        "image_emb.npy",
        "prod_emb.npy",
        "faiss_prod.index",
        "metadata.json"
    ]
    
    artifacts_exist = all((artifacts_dir / f).exists() for f in required_files)
    
    if artifacts_exist and not force_rebuild:
        print("✅ Artifacts already exist! Use --force to rebuild.")
        print("📁 Found in artifacts/:")
        for f in required_files:
            size = (artifacts_dir / f).stat().st_size / (1024*1024)
            print(f"   - {f} ({size:.1f} MB)")
        return
    
    # Check CSV file
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return
    
    print(f"📊 Processing CSV: {csv_path}")
    
    try:
        # Import here to avoid loading if artifacts exist
        from rag_backend import MultimodalRAG
        
        print("🚀 Initializing RAG system for data processing...")
        print("⚠️  This will take several minutes for:")
        print("   - Data cleaning and processing")
        print("   - Image downloading") 
        print("   - CLIP embedding generation")
        print("   - FAISS index building")
        print()
        
        # Create RAG system which will build all artifacts
        rag = MultimodalRAG(
            csv_path=csv_path,
            clip_model=clip_model,
            llm_model="microsoft/phi-3-mini-4k-instruct",  # Placeholder, won't be used
            subsample=subsample
        )
        
        # Save metadata
        metadata = {
            "csv_path": csv_path,
            "clip_model": clip_model,
            "num_products": len(rag.df),
            "embedding_dim": rag.emb_dim,
            "subsample": subsample,
            "columns_mapped": rag.mapped_cols
        }
        
        with open(artifacts_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print("\n✅ Setup completed successfully!")
        print(f"📦 Processed {len(rag.df)} products")
        print(f"🎯 Embedding dimension: {rag.emb_dim}")
        print(f"💾 Artifacts saved to: {artifacts_dir}")
        print("\n📁 Generated files:")
        
        for f in required_files:
            if (artifacts_dir / f).exists():
                size = (artifacts_dir / f).stat().st_size / (1024*1024)
                print(f"   ✓ {f} ({size:.1f} MB)")
        
        print("\n🚀 Ready to run: streamlit run streamlit_app.py")
        
    except Exception as e:
        print(f"❌ Error during setup: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup Amazon Product RAG data")
    parser.add_argument("--csv", default="real_amazon_data.csv", 
                       help="Path to CSV file (default: real_amazon_data.csv)")
    parser.add_argument("--clip-model", default="clip-ViT-B-32",
                       choices=["clip-ViT-B-32", "clip-ViT-L-14", "clip-ViT-B-16"],
                       help="CLIP model to use")
    parser.add_argument("--subsample", type=int, 
                       help="Subsample size for testing (optional)")
    parser.add_argument("--force", action="store_true",
                       help="Force rebuild even if artifacts exist")
    
    args = parser.parse_args()
    
    setup_data(
        csv_path=args.csv,
        clip_model=args.clip_model,
        subsample=args.subsample,
        force_rebuild=args.force
    )


if __name__ == "__main__":
    main() 