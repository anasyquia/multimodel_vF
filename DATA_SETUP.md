# 📊 Data Setup Guide

This multimodal RAG system requires preprocessing Amazon product data into embeddings and FAISS indices. Follow these steps to set up your data:

## 🚀 Quick Start

### Option 1: Use Sample Data (Recommended for Testing)
```bash
# The system works with the included sample dataset
# Just run the Streamlit app directly:
streamlit run streamlit_app_simple.py
```

### Option 2: Set Up Your Own Dataset

#### Step 1: Get Amazon Product Data
```bash
# Download from Kaggle (requires account)
# https://www.kaggle.com/datasets/promptcloud/amazon-product-dataset-2020

# Or use the provided download script:
python download_full_kagglehub.py
```

#### Step 2: Process the Data
```bash
# Generate CLIP embeddings and FAISS index
python setup_data.py --csv your_amazon_data.csv --clip-model clip-ViT-B-32

# This creates:
# - artifacts/faiss_prod.index (FAISS search index)
# - artifacts/text_emb.npy (text embeddings)
# - artifacts/image_emb.npy (image embeddings)
# - artifacts/products.parquet (processed data)
# - artifacts/metadata.json (system metadata)
```

## 📁 Expected Directory Structure

```
multimodal-rag/
├── streamlit_app_simple.py    # Main UI
├── rag_backend.py             # RAG system
├── config.py                  # Configuration
├── setup_data.py              # Data preprocessing
├── requirements.txt           # Dependencies
├── artifacts/                 # Generated data (not in Git)
│   ├── faiss_prod.index      # FAISS search index
│   ├── text_emb.npy          # Text embeddings
│   ├── image_emb.npy         # Image embeddings
│   ├── products.parquet      # Processed products
│   └── metadata.json         # System metadata
└── cache_images/             # Image cache (not in Git)
```

## ⚙️ System Requirements

- **RAM**: 8GB+ (16GB recommended for large datasets)
- **Storage**: 2-5GB for artifacts (depending on dataset size)
- **GPU**: Optional (CLIP works on CPU)

## 🔧 Processing Details

The `setup_data.py` script:
1. **Loads** your CSV data
2. **Generates** CLIP embeddings for text and images
3. **Creates** FAISS index for fast similarity search
4. **Saves** processed artifacts to `artifacts/` directory

## 📊 Data Statistics

With the sample dataset:
- **Products**: ~10,000 items
- **Embeddings**: 512 dimensions (CLIP ViT-B-32)
- **Index Size**: ~20MB
- **Processing Time**: 5-10 minutes

## 🚨 Important Notes

- **Large files** (`artifacts/`, `*.csv`) are excluded from Git
- **First run** takes longer due to embedding generation
- **Subsequent runs** load pre-computed artifacts quickly
- **Model downloads** happen automatically (CLIP, GPT-4o-mini)

## 🆘 Troubleshooting

**Missing artifacts?**
```bash
# Re-run data setup
python setup_data.py --csv your_data.csv
```

**Out of memory?**
```bash
# Use smaller batch size
python setup_data.py --csv your_data.csv --subsample 5000
```

**Can't find dataset?**
- Check the kaggle download scripts
- Ensure CSV has columns: product_name, description, image_urls
- Sample data structure provided in `README.md` 