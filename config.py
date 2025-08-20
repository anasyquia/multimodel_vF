import os
import streamlit as st
from typing import Dict, Any

class Config:
    """Configuration class for the Multimodal RAG application."""
    
    # Default model configurations
    DEFAULT_CLIP_MODELS = [
        "clip-ViT-B-32",
        "clip-ViT-L-14", 
        "clip-ViT-B-16"
    ]
    
    DEFAULT_LLM_MODELS = [
        "openai/gpt-4o-mini",
        "openai/gpt-4o", 
        "openai/gpt-3.5-turbo",
        "microsoft/phi-3-mini-4k-instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "google/gemma-2-2b-it"
    ]
    
    # Default search parameters
    DEFAULT_TOP_K = 5
    DEFAULT_MAX_NEW_TOKENS = 300
    DEFAULT_TEMPERATURE = 0.4
    DEFAULT_TOP_P = 0.95
    
    # File paths
    CACHE_DIR = "cache_images"
    ARTIFACTS_DIR = "artifacts"
    
    @staticmethod
    def get_huggingface_token() -> str:
        """
        Get Hugging Face token from environment or Streamlit secrets.
        
        Returns:
            HF token string or empty string if not found
        """
        # First try environment variable
        token = os.getenv("HF_TOKEN", "")
        
        # If running in Streamlit, try secrets
        if not token:
            try:
                token = st.secrets.get("HF_TOKEN", "")
            except:
                pass
        
        return token
    
    @staticmethod
    def get_openai_api_key() -> str:
        """
        Get OpenAI API key from environment or Streamlit secrets.
        
        Returns:
            OpenAI API key string or empty string if not found
        """
        # First try environment variable
        key = os.getenv("OPENAI_API_KEY", "")
        
        # If running in Streamlit, try secrets
        if not key:
            try:
                key = st.secrets.get("OPENAI_API_KEY", "")
            except:
                pass
        
        return key
    
    @staticmethod
    def setup_huggingface_token():
        """Setup Hugging Face token if available."""
        token = Config.get_huggingface_token()
        if token:
            os.environ["HF_TOKEN"] = token
        return token
    
    @staticmethod
    def get_device() -> str:
        """Get the appropriate device (cuda/cpu)."""
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    @staticmethod
    def get_app_config() -> Dict[str, Any]:
        """Get application configuration."""
        return {
            "page_title": "Multimodal Amazon Product RAG",
            "page_icon": "🛍️",
            "layout": "wide",
            "initial_sidebar_state": "expanded"
        }
    
    @staticmethod
    def get_model_config() -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "clip_models": Config.DEFAULT_CLIP_MODELS,
            "llm_models": Config.DEFAULT_LLM_MODELS,
            "default_top_k": Config.DEFAULT_TOP_K,
            "max_new_tokens": Config.DEFAULT_MAX_NEW_TOKENS,
            "temperature": Config.DEFAULT_TEMPERATURE,
            "top_p": Config.DEFAULT_TOP_P
        }
    
    @staticmethod
    def get_system_prompt() -> str:
        """Get the system prompt for the LLM."""
        return """You are a knowledgeable product expert helping customers find and understand products. Use the provided CONTEXT products to give comprehensive, helpful answers.

Guidelines:
- Answer questions using information from the CONTEXT products only
- Present each product clearly with consistent formatting
- For product listings, use this format for each item:
  **Product Name**
  - Price: [price if available, or "Price not available"]
  - Key features: [brief bullet points of main features]
  
- Compare products by highlighting key differences (features, price, brand, etc.)
- For specific product questions, extract relevant details from descriptions
- If asking about a specific product not in context, say "I don't know" and suggest similar alternatives
- For general category questions, use context products as examples and explain common features
- DO NOT repeat product names at the end of descriptions
- Keep responses concise and well-organized
- Only include complete, accurate information - if price is incomplete, say "Price not available"
- Be specific about product names when citing examples"""

    @staticmethod
    def get_css_styles() -> str:
        """Get custom CSS styles for the app."""
        return """
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #ff6b35;
            text-align: center;
            margin-bottom: 2rem;
        }
        .product-card {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            background-color: #f9f9f9;
        }
        .source-section {
            background-color: #f0f2f6;
            border-radius: 8px;
            padding: 1rem;
            margin-top: 1rem;
        }
        .main .block-container {
            max-width: 100%;
            padding-top: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .stExpander > div:first-child {
            width: 100%;
        }
        .stExpander {
            width: 100% !important;
        }
        .stExpander > div {
            width: 100% !important;
        }
        .stExpander [data-testid="stExpanderDetails"] {
            width: 100% !important;
            max-width: 100% !important;
        }
        .stMarkdown {
            max-width: none;
        }
        /* Product source sections specifically */
        .source-section .stExpander {
            width: 100% !important;
            margin: 0.5rem 0;
        }
        .source-section .stExpander > div {
            width: 100% !important;
        }
        /* Answer and content sections */
        .answer-container {
            width: 100% !important;
            max-width: 100% !important;
        }
        .stContainer {
            width: 100% !important;
        }
        .stContainer > div {
            width: 100% !important;
        }
        /* Ensure all text inputs and content use full width */
        .stTextInput > div > div > input {
            width: 100% !important;
        }
        .element-container {
            width: 100% !important;
        }
        .metric-card {
            background-color: #ffffff;
            border: 1px solid #e1e5e9;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        }
        .info-box {
            background-color: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 1rem;
            border-radius: 4px;
            margin: 1rem 0;
        }
        .warning-box {
            background-color: #fff3e0;
            border-left: 4px solid #ff9800;
            padding: 1rem;
            border-radius: 4px;
            margin: 1rem 0;
        }
        .success-box {
            background-color: #e8f5e8;
            border-left: 4px solid #4caf50;
            padding: 1rem;
            border-radius: 4px;
            margin: 1rem 0;
        }
        </style>
        """ 