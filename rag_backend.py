import os
import re
import io
import json
import time
import math
import random
import shutil
import uuid
import hashlib
import textwrap
import urllib.request
from typing import List, Tuple, Union
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import faiss
from config import Config

# OpenAI integration
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

class MultimodalRAG:
    """Multimodal RAG system for Amazon product search using CLIP and language models."""
    
    def __init__(self, 
                 csv_path: str = None,
                 clip_model: str = "clip-ViT-B-32",
                 llm_model: str = "microsoft/phi-3-mini-4k-instruct",
                 top_k: int = 5,
                 subsample: int = None):
        """
        Initialize the multimodal RAG system.
        
        Args:
            csv_path: Path to Amazon products CSV file
            clip_model: CLIP model name for embeddings
            llm_model: Language model for answer generation
            top_k: Number of top results to retrieve
            subsample: Subsample size for faster prototyping (None for all data)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model_name = clip_model
        self.llm_model_name = llm_model
        self.top_k = top_k
        self.use_openai = llm_model.startswith("openai/")
        
        # Create directories
        os.makedirs("cache_images", exist_ok=True)
        os.makedirs("artifacts", exist_ok=True)
        
        print(f"Device: {self.device}")
        
                        # Load and process data
        # First, try to load from existing artifacts
        if self._try_load_artifacts():
            print("✅ Loaded existing artifacts!")
            # Load models for inference
            self._load_clip_model()
            self._load_llm_model()
        elif csv_path:
            print("🔄 Building new artifacts...")
            self.df = self._load_and_process_data(csv_path, subsample)
            self._download_images()
            
            # Load models
            self._load_clip_model()
            self._load_llm_model()
            
            # Generate embeddings and build index
            self._generate_embeddings()
            self._build_faiss_index()
        else:
            raise ValueError("No artifacts found and no CSV path provided. Please run setup_data.py first or provide a CSV path.")
    
    def _load_and_process_data(self, csv_path: str, subsample: int = None) -> pd.DataFrame:
        """Load and preprocess the CSV data."""
        print("Loading CSV data...")
        df_raw = pd.read_csv(csv_path, low_memory=False)
        print(f"Loaded CSV shape: {df_raw.shape}")
        
        # Normalize column names
        def normalize_col(c: str) -> str:
            return re.sub(r"[^a-z0-9_]+", "", c.strip().lower().replace(" ", "_"))
        
        df = df_raw.copy()
        df.columns = [normalize_col(c) for c in df.columns]
        
        # Try to map common column variants
        col_map_candidates = {
            "name": ["product_name", "product", "title", "name"],
            "brand": ["brand", "brand_name"],
            "price": ["selling_price", "price", "sale_price", "list_price"],
            "about": ["about_product", "about_prodcut", "description", "about", "bullet_points"],
            "images": ["images", "image_url", "image_urls", "image"],
            "dimensions": ["product_dimensions", "dimensions", "size"],
            "technical_details": ["technical_details", "tech_details", "specifications"],
            "product_specification": ["product_specification", "product_spec", "spec"],
            "shipping_weight": ["shipping_weight", "weight"]
        }
        
        def find_col(possible_names):
            for n in possible_names:
                if n in df.columns:
                    return n
            return None
        
        self.mapped_cols = {k: find_col(v) for k, v in col_map_candidates.items()}
        print("Mapped columns:", self.mapped_cols)
        
        missing = [k for k, v in self.mapped_cols.items() if v is None]
        if missing:
            raise ValueError(f"Missing expected columns (please adjust mapping): {missing}")
        
        # Basic cleaning
        for key, col in self.mapped_cols.items():
            if key in ["name", "brand", "about", "dimensions", "technical_details", "product_specification", "shipping_weight"]:
                if col:  # Only clean if column exists
                    df[col] = df[col].astype(str).fillna("").str.strip()
        
        # Price: try to coerce to float
        price_col = self.mapped_cols["price"]
        def parse_price(x):
            if pd.isna(x): 
                return np.nan
            s = str(x)
            s = re.sub(r"[^0-9\.\-]", "", s)
            try:
                return float(s) if s else np.nan
            except:
                return np.nan
        
        df[price_col] = df[price_col].apply(parse_price)
        
        # Images: parse URLs; keep only non-empty http(s) links
        img_col = self.mapped_cols["images"]
        def split_urls(s: str) -> List[str]:
            if not isinstance(s, str): 
                return []
            parts = re.split(r"[;,|\s]+", s.strip())
            urls = [p for p in parts if p.startswith("http")]
            return urls
        
        df["image_urls"] = df[img_col].apply(split_urls)
        df["has_image"] = df["image_urls"].apply(lambda x: len(x) > 0)
        
        # Compose a 'full_text' per product
        name_col = self.mapped_cols["name"]
        brand_col = self.mapped_cols["brand"]
        about_col = self.mapped_cols["about"]
        
        def make_full_text(row) -> str:
            name = str(row.get(name_col, "")).strip()
            brand = str(row.get(brand_col, "")).strip()
            price = row.get(price_col, np.nan)
            about = str(row.get(about_col, "")).strip()
            price_txt = f"${price:.2f}" if (pd.notna(price)) else ""
            
            # Include technical details for better searchability
            dimensions = str(row.get(self.mapped_cols["dimensions"], "")).strip() if self.mapped_cols["dimensions"] else ""
            tech_details = str(row.get(self.mapped_cols["technical_details"], "")).strip() if self.mapped_cols["technical_details"] else ""
            product_spec = str(row.get(self.mapped_cols["product_specification"], "")).strip() if self.mapped_cols["product_specification"] else ""
            shipping_weight = str(row.get(self.mapped_cols["shipping_weight"], "")).strip() if self.mapped_cols["shipping_weight"] else ""
            
            pieces = [p for p in [name, brand, price_txt, about, dimensions, tech_details, product_spec, shipping_weight] if p and p != "nan"]
            return " | ".join(pieces)
        
        df["full_text"] = df.apply(make_full_text, axis=1)
        
        # Filter to products that have text + at least one image URL
        df = df[(df["full_text"].str.len() > 0) & (df["has_image"])].copy()
        df = df.drop_duplicates(subset=[name_col, brand_col, about_col], keep="first")
        
        print(f"After cleaning: {df.shape}")
        
        # Optional subsampling
        if subsample and len(df) > subsample:
            df = df.sample(subsample, random_state=42).reset_index(drop=True)
            print(f"Subsampled to: {len(df)} rows")
        
        return df.reset_index(drop=True)
    
    def _try_load_artifacts(self) -> bool:
        """
        Try to load existing artifacts instead of rebuilding.
        
        Returns:
            True if artifacts were loaded successfully, False otherwise
        """
        try:
            import json
            
            # Check if all required files exist
            required_files = [
                "artifacts/products.parquet",
                "artifacts/text_emb.npy", 
                "artifacts/image_emb.npy",
                "artifacts/prod_emb.npy",
                "artifacts/faiss_prod.index",
                "artifacts/metadata.json"
            ]
            
            if not all(os.path.exists(f) for f in required_files):
                return False
            
            print("📁 Loading artifacts...")
            
            # Load metadata
            with open("artifacts/metadata.json", "r") as f:
                metadata = json.load(f)
            
            # Verify CLIP model matches
            if metadata.get("clip_model") != self.clip_model_name:
                print(f"⚠️  CLIP model mismatch: artifacts={metadata.get('clip_model')}, requested={self.clip_model_name}")
                return False
            
            # Load dataframe
            self.df = pd.read_parquet("artifacts/products.parquet")
            self.mapped_cols = metadata["columns_mapped"]
            
            # Load embeddings
            self.text_emb = np.load("artifacts/text_emb.npy")
            self.image_emb = np.load("artifacts/image_emb.npy") 
            self.prod_emb = np.load("artifacts/prod_emb.npy")
            self.emb_dim = self.text_emb.shape[1]
            
            # Load FAISS index
            self.index = faiss.read_index("artifacts/faiss_prod.index")
            
            print(f"✅ Loaded {len(self.df)} products from artifacts")
            print(f"🎯 Embedding dimension: {self.emb_dim}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load artifacts: {str(e)}")
            return False
    
    def _download_images(self):
        """Download and cache first image for each product."""
        print("Downloading product images...")
        
        def url_to_filename(url: str) -> str:
            h = hashlib.md5(url.encode("utf-8")).hexdigest()
            ext = os.path.splitext(url)[1]
            if len(ext) > 5 or len(ext) == 0:
                ext = ".jpg"
            return f"{h}{ext}"
        
        def fetch_first_image(urls: List[str]) -> str:
            if not urls:
                return ""
            url = urls[0]
            fname = url_to_filename(url)
            out_path = os.path.join("cache_images", fname)
            if os.path.exists(out_path):
                return out_path
            try:
                urllib.request.urlretrieve(url, out_path)
                return out_path
            except Exception as e:
                return ""
        
        tqdm.pandas(desc="Downloading images")
        self.df["image_path"] = self.df["image_urls"].progress_apply(fetch_first_image)
        self.df = self.df[self.df["image_path"].str.len() > 0].reset_index(drop=True)
        
        print(f"Usable rows after image fetch: {len(self.df)}")
    
    def _load_clip_model(self):
        """Load the CLIP model for embeddings."""
        print(f"Loading CLIP model: {self.clip_model_name}")
        self.clip_model = SentenceTransformer(self.clip_model_name, device=self.device)
    
    def _load_llm_model(self):
        """Load the language model for answer generation."""
        print(f"Loading LLM: {self.llm_model_name}")
        
        if self.use_openai:
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI package not available. Install with: pip install openai")
            
            # Extract model name (remove "openai/" prefix)
            self.openai_model = self.llm_model_name.split("/", 1)[1]
            
            # Initialize OpenAI client
            api_key = Config.get_openai_api_key()
            if not api_key:
                raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable or provide it in the UI.")
            
            self.openai_client = openai.OpenAI(api_key=api_key)
            print(f"✅ OpenAI client initialized with model: {self.openai_model}")
            
        else:
            # Load local model
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name, trust_remote_code=True)
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            self.llm_model.eval()
    
    def _generate_embeddings(self):
        """Generate CLIP embeddings for text and images."""
        print("Generating embeddings...")
        
        # Text embeddings
        texts = self.df["full_text"].tolist()
        self.text_emb = self.clip_model.encode(
            texts, 
            batch_size=64, 
            convert_to_numpy=True, 
            normalize_embeddings=True, 
            show_progress_bar=True
        )
        
        # Image embeddings
        def load_image(p):
            try:
                img = Image.open(p).convert("RGB")
                return img
            except Exception:
                return None
        
        images = [load_image(p) for p in self.df["image_path"].tolist()]
        valid_mask = [im is not None for im in images]
        valid_mask_array = np.array(valid_mask)
        
        if not all(valid_mask):
            self.df = self.df[valid_mask_array].reset_index(drop=True)
            self.text_emb = self.text_emb[valid_mask_array]
            images = [im for im in images if im is not None]
        
        self.image_emb = self.clip_model.encode(
            images, 
            batch_size=32, 
            convert_to_numpy=True, 
            normalize_embeddings=True, 
            show_progress_bar=True
        )
        
        print(f"Embedding shapes: text={self.text_emb.shape}, image={self.image_emb.shape}")
        assert self.text_emb.shape == self.image_emb.shape
        self.emb_dim = self.text_emb.shape[1]
        
        # Product-level embedding: average of text & image (normalize after avg)
        self.prod_emb = self.text_emb + self.image_emb
        prod_norms = np.linalg.norm(self.prod_emb, axis=1, keepdims=True) + 1e-12
        self.prod_emb = self.prod_emb / prod_norms
        
        # Save artifacts
        np.save("artifacts/text_emb.npy", self.text_emb)
        np.save("artifacts/image_emb.npy", self.image_emb)
        np.save("artifacts/prod_emb.npy", self.prod_emb)
        self.df.to_parquet("artifacts/products.parquet", index=False)
    
    def _build_faiss_index(self):
        """Build FAISS index for product embeddings."""
        print("Building FAISS index...")
        
        d = self.prod_emb.shape[1]
        self.index = faiss.IndexFlatIP(d)  # cosine similarity if normalized
        self.index.add(self.prod_emb.astype(np.float32))
        
        print(f"FAISS index ntotal: {self.index.ntotal}")
        
        # Save index
        faiss.write_index(self.index, "artifacts/faiss_prod.index")
    
    def _search_products(self, query_vec: np.ndarray, top_k: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar products using FAISS."""
        if top_k is None:
            top_k = self.top_k
        
        q = query_vec.reshape(1, -1).astype(np.float32)
        sims, idxs = self.index.search(q, top_k)
        return sims[0], idxs[0]
    
    def _encode_text_query(self, qtext: str) -> np.ndarray:
        """Encode text query to embedding."""
        v = self.clip_model.encode([qtext], convert_to_numpy=True, normalize_embeddings=True)[0]
        return v
    
    def _encode_image_query(self, img_path: str) -> np.ndarray:
        """Encode image query to embedding."""
        img = Image.open(img_path).convert("RGB")
        v = self.clip_model.encode([img], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)[0]
        return v
    
    def _format_product_card(self, row) -> str:
        """Format a product row as a text card."""
        name_col = self.mapped_cols["name"]
        brand_col = self.mapped_cols["brand"]
        price_col = self.mapped_cols["price"]
        about_col = self.mapped_cols["about"]
        
        name = str(row.get(name_col, ""))
        brand = str(row.get(brand_col, ""))
        price = row.get(price_col, np.nan)
        about = str(row.get(about_col, ""))
        image_urls = row.get("image_urls", [])
        img = str(image_urls[0] if len(image_urls) > 0 else "")
        
        # Clean up brand and price formatting
        brand_clean = brand if brand and brand.lower() not in ['nan', 'none', ''] else None
        price_clean = f"${price:.2f}" if pd.notna(price) and price > 0 else None
        
        # Build product info with only available data
        product_info = []
        if brand_clean:
            product_info.append(f"Brand: {brand_clean}")
        if price_clean:
            product_info.append(f"Price: {price_clean}")
        
        # Format the product card
        info_str = f" ({', '.join(product_info)})" if product_info else ""
        img_str = f"\nImage: {img}" if img else ""
        about_str = f"\nAbout: {about[:500]}" if about and about.lower() != 'nan' else ""
        
        return f"- **{name}**{info_str}{img_str}{about_str}"
    
    def _build_prompt(self, user_q: str, context_rows, is_image_question: bool = False) -> str:
        """Build prompt for the LLM with context."""
        context_block = "\n".join([self._format_product_card(r) for r in context_rows])
        
        if is_image_question:
            system_prompt = """You are a helpful product expert. When analyzing an uploaded image, use the provided CONTEXT products (which were retrieved by visual similarity) to identify what type of product is shown.

For image identification questions:
- Look at the similar products in the context to determine the product category and type
- Describe what you can see in the image based on the similar products found
- If the exact product isn't in the context, identify the general category/type and mention key features
- Suggest from the context

For other questions, cite relevant product names and compare them clearly."""
        else:
            system_prompt = Config.get_system_prompt()
        
        prompt = f"""{system_prompt}

CONTEXT (Top results):
{context_block}

User: {user_q}
Assistant:"""
        return prompt
    
    def _generate_answer(self, prompt: str, max_new_tokens=300, temperature=0.4, top_p=0.95) -> str:
        """Generate answer using the LLM (OpenAI or local)."""
        if self.use_openai:
            return self._generate_openai_answer(prompt, max_new_tokens, temperature, top_p)
        else:
            return self._generate_local_answer(prompt, max_new_tokens, temperature, top_p)
    
    def _generate_openai_answer(self, prompt: str, max_new_tokens=300, temperature=0.4, top_p=0.95) -> str:
        """Generate answer using OpenAI API."""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    @torch.inference_mode()
    def _generate_local_answer(self, prompt: str, max_new_tokens=300, temperature=0.4, top_p=0.95) -> str:
        """Generate answer using local LLM."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm_model.device)
        output_ids = self.llm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id
        )
        out = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        if "Assistant:" in out:
            out = out.split("Assistant:")[-1].strip()
        return out
    
    def _get_topk_rows_by_query_vec(self, qv: np.ndarray, top_k: int = None):
        """Get top-k product rows by query vector."""
        if top_k is None:
            top_k = self.top_k
        
        sims, idxs = self._search_products(qv, top_k=top_k)
        rows = [self.df.iloc[i] for i in idxs]
        return rows
    
    def answer_text_question(self, question: str, top_k: int = None) -> Tuple[str, List[str]]:
        """
        Answer a text-based question about products.
        
        Args:
            question: User's text question
            top_k: Number of top results to retrieve
            
        Returns:
            Tuple of (answer, sources)
        """
        if top_k is None:
            top_k = self.top_k
            
        qv = self._encode_text_query(question)
        rows = self._get_topk_rows_by_query_vec(qv, top_k=top_k)
        
        # Filter out low-quality products for consistency with search results
        quality_rows = []
        for row in rows:
            product_dict = {
                'name': str(row.get(self.mapped_cols['name'], '')),
                'brand': str(row.get(self.mapped_cols['brand'], '')),
                'price': str(row.get(self.mapped_cols['price'], '')),
                'about': str(row.get(self.mapped_cols['about'], '')),
                'images': str(row.get(self.mapped_cols['images'], ''))
            }
            if self._is_quality_product(product_dict):
                quality_rows.append(row)
        
        # Use quality-filtered rows for both prompt and sources
        prompt = self._build_prompt(question, quality_rows)
        answer = self._generate_answer(prompt)
        sources = [self._format_product_card(r) for r in quality_rows]
        
        # Also return the raw product data for better display consistency
        product_data = []
        for row in quality_rows:
            product_dict = {
                'name': str(row.get(self.mapped_cols['name'], '')),
                'brand': str(row.get(self.mapped_cols['brand'], '')),
                'price': str(row.get(self.mapped_cols['price'], '')),
                'about': str(row.get(self.mapped_cols['about'], '')),
                'images': str(row.get(self.mapped_cols['images'], '')),
                'full_row': row
            }
            product_data.append(product_dict)
        
        return answer, sources, product_data
    
    def answer_image_question(self, image_path: str, question: str = "Identify and describe this product.", top_k: int = None) -> Tuple[str, List[str]]:
        """
        Answer an image-based question about products.
        
        Args:
            image_path: Path to uploaded image
            question: Question about the image
            top_k: Number of top results to retrieve
            
        Returns:
            Tuple of (answer, sources)
        """
        if top_k is None:
            top_k = self.top_k
            
        qv = self._encode_image_query(image_path)
        rows = self._get_topk_rows_by_query_vec(qv, top_k=top_k)
        
        # Filter out low-quality products for consistency with search results
        quality_rows = []
        for row in rows:
            product_dict = {
                'name': str(row.get(self.mapped_cols['name'], '')),
                'brand': str(row.get(self.mapped_cols['brand'], '')),
                'price': str(row.get(self.mapped_cols['price'], '')),
                'about': str(row.get(self.mapped_cols['about'], '')),
                'images': str(row.get(self.mapped_cols['images'], ''))
            }
            if self._is_quality_product(product_dict):
                quality_rows.append(row)
        
        # Use quality-filtered rows for both prompt and sources
        aug_q = f"{question} Use the context which was retrieved by visual similarity to this image."
        prompt = self._build_prompt(aug_q, quality_rows, is_image_question=True)
        answer = self._generate_answer(prompt)
        sources = [self._format_product_card(r) for r in quality_rows]
        
        # Also return the raw product data for better display consistency
        product_data = []
        for row in quality_rows:
            product_dict = {
                'name': str(row.get(self.mapped_cols['name'], '')),
                'brand': str(row.get(self.mapped_cols['brand'], '')),
                'price': str(row.get(self.mapped_cols['price'], '')),
                'about': str(row.get(self.mapped_cols['about'], '')),
                'images': str(row.get(self.mapped_cols['images'], '')),
                'full_row': row
            }
            product_data.append(product_dict)
        
        return answer, sources, product_data
    
    def evaluate_recall_at_k(self, ks=(1, 5, 10), eval_n=None) -> dict:
        """
        Evaluate retrieval performance using recall@k metrics.
        
        Args:
            ks: List of k values to evaluate
            eval_n: Number of samples to evaluate (None for all)
            
        Returns:
            Dictionary with recall@k scores
        """
        if eval_n is None:
            eval_n = min(1000, len(self.df))
        
        eval_idx = np.random.default_rng(42).choice(len(self.df), size=eval_n, replace=False)
        
        def recall_at_k(query_mat: np.ndarray, target_idx: np.ndarray, ks=(1, 5, 10)) -> dict:
            N = len(target_idx)
            ks = sorted(list(ks))
            hits_at = {k: 0 for k in ks}
            B = 256
            
            for i in range(0, N, B):
                q_batch = query_mat[i:i+B].astype(np.float32)
                sims, idxs = self.index.search(q_batch, max(ks))
                for bi in range(q_batch.shape[0]):
                    truth = target_idx[i+bi]
                    retrieved = idxs[bi].tolist()
                    for k in ks:
                        if truth in retrieved[:k]:
                            hits_at[k] += 1
            
            out = {f"Recall@{k}": hits_at[k] / N for k in ks}
            out["Accuracy@1"] = out["Recall@1"]
            return out
        
        text_q = self.text_emb[eval_idx]
        img_q = self.image_emb[eval_idx]
        truth = eval_idx
        
        text2prod = recall_at_k(text_q, truth, ks=ks)
        img2prod = recall_at_k(img_q, truth, ks=ks)
        
        return {
            "text_to_product": text2prod,
            "image_to_product": img2prod
        } 

    def search_product_by_name(self, product_name: str, fuzzy_match: bool = True) -> List[dict]:
        """
        Search for products by name/title.
        
        Args:
            product_name: Product name to search for
            fuzzy_match: Whether to use fuzzy matching (default True)
            
        Returns:
            List of matching product dictionaries with name, brand, price, image, etc.
        """
        # Get the product name column
        name_col = self.mapped_cols.get('name')
        if not name_col:
            raise ValueError("Product name column not found")
        
        # Clean search term
        search_term = product_name.lower().strip()
        
        if fuzzy_match:
            # Fuzzy search - find products that contain search terms
            mask = self.df[name_col].str.lower().str.contains(search_term, case=False, na=False)
            matches = self.df[mask]
        else:
            # Exact match
            mask = self.df[name_col].str.lower() == search_term
            matches = self.df[mask]
        
        # If no fuzzy matches, try individual words
        if len(matches) == 0 and fuzzy_match:
            words = search_term.split()
            if len(words) > 1:
                # Try matching any of the words
                word_masks = [self.df[name_col].str.lower().str.contains(word, case=False, na=False) for word in words]
                combined_mask = word_masks[0]
                for mask in word_masks[1:]:
                    combined_mask = combined_mask | mask
                matches = self.df[combined_mask]
        
        # Convert to list of dictionaries and check more candidates for quality filtering
        results = []
        for _, row in matches.head(20).iterrows():  # Check more candidates to find quality ones
            result = {
                'name': str(row.get(self.mapped_cols['name'], '')),
                'brand': str(row.get(self.mapped_cols['brand'], '')),
                'price': str(row.get(self.mapped_cols['price'], '')),
                'about': str(row.get(self.mapped_cols['about'], '')),
                'images': str(row.get(self.mapped_cols['images'], '')),
                'full_row': row
            }
            results.append(result)
        
        # Filter out products with bad data
        filtered_results = []
        for result in results:
            if self._is_quality_product(result):
                filtered_results.append(result)
        
        return filtered_results
    
    def _is_quality_product(self, product: dict) -> bool:
        """
        Check if a product has sufficient quality data to be worth showing.
        
        Args:
            product: Product dictionary
            
        Returns:
            True if product has good enough data, False if it should be filtered out
        """
        import re
        
        name = str(product.get('name', '')).strip()
        about = str(product.get('about', '')).strip()
        price = str(product.get('price', '')).strip()
        brand = str(product.get('brand', '')).strip()
        
        # Must have a meaningful name
        if not name or name.lower() in ['nan', 'none', ''] or len(name) < 3:
            return False
        
        # Name shouldn't be mostly numbers/codes
        if len(re.sub(r'[\d\s\-_]', '', name)) < 3:
            return False
        
        # At least one of: meaningful description, valid price, or valid brand
        has_description = about and about.lower() not in ['nan', 'none', ''] and len(about) > 15
        has_price = price and price.lower() not in ['nan', 'none', ''] and price != '0' and price != '0.0'
        has_brand = brand and brand.lower() not in ['nan', 'none', '']
        
        # If description exists, it should be meaningful (not just punctuation)
        if has_description:
            clean_about = re.sub(r'[\s\W]', '', about)
            if len(clean_about) < 5:  # Less than 5 actual characters
                has_description = False
        
        # Must have at least description OR (price AND brand)
        return has_description or (has_price and has_brand) 