#!/usr/bin/env python3
"""
Launch script for the Multimodal Amazon Product RAG Streamlit app
"""

import subprocess
import sys
import os

def main():
    print("🛍️ Starting Multimodal Amazon Product RAG")
    print("=" * 50)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit found")
    except ImportError:
        print("❌ Streamlit not found. Please install requirements:")
        print("pip install -r requirements.txt")
        return
    
    # Check if main app file exists
    if not os.path.exists("streamlit_app.py"):
        print("❌ streamlit_app.py not found")
        return
    
    print("🚀 Launching Streamlit app...")
    print("The app will be available at: http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Launch streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"❌ Error launching app: {e}")

if __name__ == "__main__":
    main() 