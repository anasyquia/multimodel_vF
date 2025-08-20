# 🛍️ Multimodal Amazon Product RAG

A sophisticated Retrieval-Augmented Generation (RAG) system for Amazon product search that combines text and image queries using CLIP embeddings and language models.

## 👥 Group Setup (Start Here!)

### Quick Start for Team Members

1. **Clone this repository**
   ```bash
   git clone [YOUR_GITHUB_REPO_URL]
   cd GenAI_Final_2
   ```

2. **Set up Python environment**
   ```bash
   # Create virtual environment (recommended)
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Get OpenAI API Key**
   - Each team member needs their own OpenAI API key
   - Sign up at: https://platform.openai.com/api-keys
   - **DO NOT commit API keys to GitHub!**
   - You'll enter the key in the Streamlit interface

5. **Download Amazon Product Dataset**
   - You'll need an Amazon product CSV file
   - Place it in the project root as `real_amazon_data.csv`
   - OR use the upload feature in the Streamlit interface

6. **Run the application**
   ```bash
   # Run the simplified interface (recommended)
   streamlit run streamlit_app_simple.py
   
   # OR run the full interface
   streamlit run streamlit_app.py
   ```

7. **Access the app**
   - Open your browser to: http://localhost:8501
   - Enter your OpenAI API key in the sidebar
   - Initialize the RAG system
   - Start searching!

### 📋 What Each Team Member Needs
- ✅ Python 3.8+ installed
- ✅ Git installed
- ✅ OpenAI API key (get your own from platform.openai.com)
- ✅ Amazon product CSV dataset

## ✨ Features

- **Multimodal Search**: Query products using both text and images
- **CLIP Embeddings**: Uses state-of-the-art CLIP models for cross-modal understanding
- **Vector Search**: Fast similarity search with FAISS indexing
- **Language Model Integration**: OpenAI GPT-4o-mini for answer generation
- **Interactive UI**: Beautiful Streamlit interface with real-time results
- **Product Context**: Retrieves and displays relevant product information with sources
- **Three Query Types**:
  - 🔍 Text-based product search
  - 🖼️ Image-based product search  
  - 🖼️ Specific product image requests

## 🔧 Detailed Setup

### Prerequisites

- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Installation

1. **Clone the repository**
   ```bash
   git clone [YOUR_REPO_URL]
   cd GenAI_Final_2
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   For GPU support, replace `faiss-cpu` with `faiss-gpu` in requirements.txt:
   ```bash
   pip uninstall faiss-cpu
   pip install faiss-gpu
   ```

3. **Prepare your data**
   
   Place your Amazon product CSV file in the project directory or prepare to upload it through the UI. The CSV should contain columns for:
   - Product name
   - Brand
   - Price
   - Description/About
   - Image URLs

4. **Set up API keys**
   
   For OpenAI models (recommended), you can either:
   - Enter your API key directly in the Streamlit interface (recommended)
   - Or set environment variable:
   ```bash
   export OPENAI_API_KEY=your_openai_key_here
   ```

## 🚀 Usage

### Pre-processing Data (Recommended)

For faster startup, pre-process your data once:

```bash
# Process your CSV and build FAISS index with CLIP embeddings
python setup_data.py

# For custom options:
python setup_data.py --csv your_data.csv --clip-model clip-ViT-L-14 --subsample 1000
```

This creates the `artifacts/` folder with:
- Pre-computed CLIP embeddings
- FAISS vector index  
- Processed product data

### Running the Application

```bash
streamlit run streamlit_app.py
```

The application will be available at `http://localhost:8501`

**Note**: If you don't run setup first, the app will process everything on first launch (slower).

### Using the Interface

1. **Configuration**:
   - Upload your Amazon product CSV file
   - Select CLIP model (default: clip-ViT-B-32)
   - Choose LLM model (default: microsoft/phi-3-mini-4k-instruct)
   - Adjust search parameters

2. **Initialize System**:
   - Click "🚀 Initialize RAG System"
   - Wait for data processing and model loading

3. **Text Queries**:
   - Use the "💬 Text Query" tab
   - Ask questions like "What are the features of the Lego Minecraft Creeper?"
   - View AI-generated answers and source products

4. **Image Queries**:
   - Use the "🖼️ Image Query" tab
   - Upload a product image
   - Ask questions about the image
   - Get visually similar products and descriptions

5. **System Information**:
   - Check the "📊 System Info" tab for dataset and model details

## 🏗️ Architecture

### Components

- **`streamlit_app.py`**: Main Streamlit application interface
- **`rag_backend.py`**: Core RAG system with CLIP embeddings and FAISS search
- **`config.py`**: Configuration management and settings

### Data Flow

1. **Data Processing**: CSV data is cleaned, normalized, and images are downloaded
2. **Embedding Generation**: CLIP model creates embeddings for text and images
3. **Index Building**: FAISS index enables fast similarity search
4. **Query Processing**: User queries are encoded and matched against the index
5. **Answer Generation**: Retrieved context is used to generate responses via LLM

### Models

**Selected Models** (simplified for assignment):
- **CLIP Model**: `clip-ViT-B-32` - Balanced performance for multimodal embeddings
- **Language Model**: `openai/gpt-4o-mini` - Fast, cost-effective OpenAI model for answer generation

*Note: Models are pre-configured to reduce complexity and ensure consistent results.*

## 📊 Data Requirements

### CSV Format

Your Amazon product CSV should include these columns (names will be automatically mapped):

| Column Type | Possible Names | Description |
|-------------|----------------|-------------|
| Product Name | `product_name`, `product`, `title`, `name` | Product title |
| Brand | `brand`, `brand_name` | Brand name |
| Price | `selling_price`, `price`, `sale_price`, `list_price` | Product price |
| Description | `about_product`, `description`, `about`, `bullet_points` | Product details |
| Images | `images`, `image_url`, `image_urls`, `image` | Image URLs (semicolon/comma separated) |

### Example Data

```csv
product_name,brand_name,selling_price,about_product,image
"LEGO Minecraft Creeper BigFig",LEGO,14.99,"Buildable Minecraft figure with 184 pieces","https://images-na.ssl-images-amazon.com/images/I/51LoYG%2BDsLL.jpg"
```

## 🎯 Performance

### Evaluation Metrics

The system includes built-in evaluation using Recall@K metrics:
- **Text→Product**: How well text queries retrieve relevant products
- **Image→Product**: How well image queries find similar products

### Optimization Tips

1. **GPU Usage**: Use CUDA-compatible hardware for faster processing
2. **Data Size**: Subsample large datasets for faster prototyping
3. **Model Choice**: Balance quality vs. speed based on your needs
4. **Batch Size**: Adjust embedding batch sizes based on available memory

## 🔍 Example Queries

### Text Queries
- "What are the features of the Lego Minecraft Creeper?"
- "Show me wireless headphones under $100"
- "Compare different smartphone brands"
- "Find kitchen appliances for small apartments"

### Image Queries
- Upload a product image and ask "Can you identify this product?"
- "What are similar products to this item?"
- "Tell me about the usage of this product"

## 🛠️ Customization

### Adding New Models

1. **CLIP Models**: Add new models to `config.py` in `DEFAULT_CLIP_MODELS`
2. **LLM Models**: Add compatible Hugging Face models to `DEFAULT_LLM_MODELS`

### Modifying System Prompt

Update the system prompt in `config.py` `get_system_prompt()` method to change AI behavior.

### Custom Styling

Modify CSS styles in `config.py` `get_css_styles()` method for UI customization.

## 📁 File Structure

```
multimodal-rag/
├── streamlit_app.py          # Main Streamlit application
├── rag_backend.py            # Core RAG system
├── config.py                 # Configuration management
├── setup_data.py             # Pre-processing script
├── run.py                    # Launch helper
├── demo.py                   # Simple test script
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── real_amazon_data.csv     # Your Amazon product data
├── cache_images/            # Downloaded product images
└── artifacts/               # Generated embeddings and indexes
    ├── text_emb.npy         # Text embeddings
    ├── image_emb.npy        # Image embeddings  
    ├── prod_emb.npy         # Combined product embeddings
    ├── products.parquet     # Processed product data
    ├── faiss_prod.index     # FAISS vector index
    └── metadata.json        # Processing metadata
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for CLIP models
- Sentence Transformers for easy CLIP integration
- Facebook Research for FAISS
- Hugging Face for transformer models
- Streamlit for the amazing web framework

## 📞 Support

For issues and questions:
1. Check the existing issues
2. Create a new issue with detailed description
3. Include system information and error logs 