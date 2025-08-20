#!/usr/bin/env python3
"""
Demo script for testing the Multimodal RAG system
"""

import os
import sys
import pandas as pd
from rag_backend import MultimodalRAG

def main():
    print("🛍️ Multimodal Amazon Product RAG Demo")
    print("=" * 50)
    
    # Check if CSV file exists
    csv_path = "real_amazon_data.csv"
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        print("Please ensure you have the Amazon product dataset in the current directory.")
        return
    
    print(f"📊 Found CSV file: {csv_path}")
    
    try:
        # Initialize RAG system
        print("🚀 Initializing RAG system...")
        rag = MultimodalRAG(
            csv_path=csv_path,
            clip_model="clip-ViT-B-32",
            llm_model="microsoft/phi-3-mini-4k-instruct",
            top_k=3,
            subsample=50  # Use small subset for demo
        )
        
        print("✅ RAG system initialized successfully!")
        print(f"📦 Loaded {len(rag.df)} products")
        
        # Demo text query
        print("\n💬 Testing text query...")
        question = "What are the features of Lego products?"
        answer, sources = rag.answer_text_question(question)
        
        print(f"Q: {question}")
        print(f"A: {answer}")
        print("\nTop Sources:")
        for i, source in enumerate(sources[:2], 1):
            print(f"{i}. {source[:200]}...")
        
        print("\n🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during demo: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 