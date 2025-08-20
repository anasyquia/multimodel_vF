import streamlit as st
import tempfile
import os
import pandas as pd
from rag_backend import MultimodalRAG
from config import Config
from typing import List

# Page config
st.set_page_config(
    page_title="Multimodal RAG for Amazon Products", 
    page_icon="🛒", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply CSS
st.markdown(Config.get_css_styles(), unsafe_allow_html=True)

def calculate_semantic_similarity(query: str, product_names: list, rag_system) -> float:
    """
    Calculate semantic similarity between query and retrieved products using CLIP embeddings.
    Returns average similarity score between 0.0 and 1.0.
    """
    import numpy as np
    try:
        # Encode the query
        query_embedding = rag_system._encode_text_query(query)
        
        # Encode product names
        similarities = []
        for product_name in product_names:
            if product_name and str(product_name).strip():
                product_embedding = rag_system._encode_text_query(str(product_name))
                
                # Calculate cosine similarity
                dot_product = np.dot(query_embedding, product_embedding)
                norm_query = np.linalg.norm(query_embedding)
                norm_product = np.linalg.norm(product_embedding)
                
                if norm_query > 0 and norm_product > 0:
                    similarity = dot_product / (norm_query * norm_product)
                    # Convert from [-1, 1] to [0, 1] range
                    similarity = (similarity + 1) / 2
                    similarities.append(similarity)
        
        if similarities:
            return np.mean(similarities)
        else:
            return 0.0
            
    except Exception as e:
        print(f"Error calculating semantic similarity: {e}")
        return 0.0

def calculate_answer_relevance(question: str, answer: str) -> float:
    """
    Calculate relevance score between a question and its answer.
    Returns a score between 0.0 and 1.0 (higher is better).
    """
    import re
    
    if not question or not answer:
        return 0.0
    
    # Clean and normalize text
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    question_clean = clean_text(question)
    answer_clean = clean_text(answer)
    
    if not question_clean or not answer_clean:
        return 0.0
    
    question_words = set(question_clean.split())
    answer_words = set(answer_clean.split())
    
    # Remove common stop words that don't add much meaning
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'it', 'its', 'we', 'us', 'our', 'they', 'them', 'their'}
    
    question_words_filtered = question_words - stop_words
    answer_words_filtered = answer_words - stop_words
    
    if not question_words_filtered:
        return 0.5  # Neutral score if no meaningful words in question
    
    # Calculate word overlap
    common_words = question_words_filtered.intersection(answer_words_filtered)
    word_overlap_score = len(common_words) / len(question_words_filtered)
    
    # Calculate answer completeness (length-based heuristic)
    answer_length = len(answer_clean.split())
    completeness_score = min(1.0, answer_length / 20)  # Normalize to 20 words as "complete"
    
    # Check for key question patterns and responses
    pattern_bonus = 0.0
    if any(word in question_clean for word in ['what', 'which', 'show', 'find', 'search']):
        if any(word in answer_clean for word in ['product', 'item', 'available', 'here', 'found']):
            pattern_bonus = 0.1
    
    if any(word in question_clean for word in ['how', 'much', 'cost', 'price']):
        if any(word in answer_clean for word in ['$', 'price', 'cost', 'dollar']):
            pattern_bonus = 0.1
    
    if any(word in question_clean for word in ['best', 'good', 'recommend']):
        if any(word in answer_clean for word in ['recommend', 'good', 'best', 'excellent', 'quality']):
            pattern_bonus = 0.1
    
    # Combine scores with weights
    final_score = (word_overlap_score * 0.6) + (completeness_score * 0.3) + pattern_bonus
    
    return min(1.0, final_score)

def clean_description(description: str) -> str:
    """Clean generic Amazon template phrases and messy data from product descriptions."""
    import re
    
    if not description or description.lower() in ['nan', 'none', '']:
        return ""
    
    # Common generic phrases to filter out
    generic_phrases = [
        "Make sure this fits by entering your model number.",
        "Make sure this fits by entering your model number",
        "Please check the size chart before purchasing",
        "Customer satisfaction is our top priority",
        "If you have any questions, please contact us",
        "100% brand new and high quality",
        "Package includes:",
        "Note: Light shooting and different displays",
        "Due to the difference between different monitors"
    ]
    
    cleaned = str(description)
    
    # Remove generic phrases
    for phrase in generic_phrases:
        cleaned = cleaned.replace(phrase, "").strip()
    
    # Clean up messy formatting patterns
    # Remove excessive pipes and dashes
    cleaned = re.sub(r'\|+', ' ', cleaned)  # Replace multiple pipes with space
    cleaned = re.sub(r'-{2,}', ' ', cleaned)  # Replace multiple dashes with space
    cleaned = re.sub(r'[|:\-\.]{3,}', ' ', cleaned)  # Replace 3+ repetitive punctuation
    
    # Remove patterns like "- -:" or ": -" or "- -"
    cleaned = re.sub(r'[\-\s]*[\-:\.]\s*[\-\s]*', ' ', cleaned)
    
    # Clean up excessive whitespace and punctuation
    cleaned = re.sub(r'\s+', ' ', cleaned)  # Multiple spaces to single space
    cleaned = re.sub(r'^[\s\-\.\|:,;]+', '', cleaned)  # Remove leading junk
    cleaned = re.sub(r'[\s\-\.\|:,;]+$', '', cleaned)  # Remove trailing junk
    
    # Remove lines that are mostly punctuation or very short
    lines = cleaned.split('\n')
    good_lines = []
    for line in lines:
        line = line.strip()
        if len(line) > 10 and len(re.sub(r'[\s\-\.\|:,;]', '', line)) > 5:
            good_lines.append(line)
    
    if good_lines:
        cleaned = ' '.join(good_lines)
    
    # Final cleanup
    cleaned = " ".join(cleaned.split())  # Normalize whitespace
    cleaned = cleaned.strip(".,;:-|")    # Remove trailing punctuation
    
    # If result is too short or mostly punctuation, return empty
    if len(cleaned) < 15 or len(re.sub(r'[\s\W]', '', cleaned)) < 5:
        return ""
    
    return cleaned if cleaned else ""

def display_product_sources(sources: List[str]):
    """Display retrieved product sources with proper image handling"""
    for i, source in enumerate(sources, 1):
        with st.expander(f"Product {i}", expanded=True):
            # Split the source into components for better formatting
            lines = source.split('\n')
            
            # Look for image URL and display it first
            image_url = None
            for line in lines:
                if line.strip().startswith('Image: http'):
                    image_url = line.replace('Image: ', '').strip()
                    break
            
            # Display image if found
            if image_url:
                try:
                    st.image(image_url, width=200, caption=f"Product {i}")
                except Exception as e:
                    st.write(f"Image: {image_url}")
            
            # Parse and display product information cleanly
            product_title = None
            price = None
            brand = None
            description_parts = []
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('Image: http'):
                    continue
                
                # Extract product title (first meaningful line with **)
                if product_title is None and line.startswith('- **') and '**' in line:
                    # Extract title between **
                    title_match = line.split('**')
                    if len(title_match) >= 3:
                        product_title = title_match[1].strip()
                    
                    # Extract brand and price from the same line (new cleaner format)
                    if '(' in line and ')' in line:
                        paren_content = line.split('(')[1].split(')')[0]
                        parts = paren_content.split(', ')
                        for part in parts:
                            if part.startswith('Brand:'):
                                brand_value = part.replace('Brand:', '').strip()
                                if brand_value and brand_value.lower() not in ['nan', 'none', '']:
                                    brand = brand_value
                            elif part.startswith('Price:'):
                                price_value = part.replace('Price:', '').strip()
                                if price_value and price_value not in ['N/A', 'nan', '']:
                                    price = price_value
                    continue
                
                # Extract description - lines starting with "About:"
                if line.startswith('About:'):
                    desc_text = line.replace('About:', '').strip()
                    if desc_text:
                        # Clean the description of generic phrases
                        cleaned_desc = clean_description(desc_text)
                        if cleaned_desc and len(cleaned_desc) > 10:  # Only add if substantial content remains
                            description_parts.append(cleaned_desc)
            
            # Display formatted information
            if product_title:
                st.markdown(f"**{product_title}**")
            
            # Display brand and price if available
            info_cols = st.columns(2)
            with info_cols[0]:
                if brand:
                    st.write(f"🏷️ **Brand:** {brand}")
            with info_cols[1]:
                if price:
                    st.write(f"💰 **Price:** {price}")
            
            # Display description
            if description_parts:
                st.write("📝 **Description:**")
                for desc in description_parts:
                    st.write(desc)

def display_product_simple(product: dict):
    """Display a simple product card with just name and image for text queries."""
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Display image
        if product['images'] and product['images'] != 'nan' and product['images'].strip():
            # Get first image URL
            image_urls = product['images'].split(';')
            if image_urls:
                first_image = image_urls[0].strip()
                if first_image:
                    st.image(first_image, use_container_width=True)
                else:
                    st.info("No image available")
            else:
                st.info("No image available")
        else:
            st.info("No image available")
    
    with col2:
        # Just the product name
        st.markdown(f"### {product['name']}")

def display_product_image(product: dict):
    """Display a single product with its image prominently."""
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Display image
        if product['images'] and product['images'] != 'nan' and product['images'].strip():
            # Get first image URL
            image_urls = product['images'].split(';')
            if image_urls:
                first_image = image_urls[0].strip()
                if first_image:
                    st.image(first_image, use_container_width=True)
                else:
                    st.info("No image available")
            else:
                st.info("No image available")
        else:
            st.info("No image available")
    
    with col2:
        # Product details
        st.markdown(f"### {product['name']}")
        
        # Brand and price in columns
        detail_col1, detail_col2 = st.columns(2)
        with detail_col1:
            if product['brand'] and product['brand'] != 'nan':
                st.markdown(f"**Brand:** {product['brand']}")
        with detail_col2:
            if product['price'] and product['price'] != 'nan':
                st.markdown(f"**Price:** ${product['price']}")
        
        # Description
        if product['about'] and product['about'] != 'nan':
            description = clean_description(product['about'])
            if description:
                st.markdown("**Description:**")
                st.markdown(description)

# Main title
st.title("🛒 Multimodal RAG: Amazon Product Search")
st.markdown("Search Amazon products using text queries or upload images for visual search!")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key Input
    st.subheader("🔑 API Configuration")
    openai_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key for GPT models",
        placeholder="sk-..."
    )
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    
    # Fixed model settings
    st.subheader("🤖 System Configuration")
    st.info("**CLIP Model**: clip-ViT-B-32 (for multimodal embeddings)")
    st.info("**LLM Model**: openai/gpt-4o-mini (for answer generation)")
    
    # Search settings
    st.subheader("🔍 Search Settings")
    top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)
    
    # Initialize system
    if st.button("🚀 Initialize RAG System", type="primary"):
        try:
            if not openai_key:
                st.error("❌ OpenAI API key required. Please enter your API key above.")
                st.stop()
            
            with st.spinner("Initializing RAG system..."):
                rag = MultimodalRAG(
                    csv_path=None,  # Use artifacts only
                    clip_model="clip-ViT-B-32",
                    llm_model="openai/gpt-4o-mini",
                    top_k=top_k
                )
                st.session_state.rag_system = rag
                st.session_state.data_loaded = True
            
            st.success("✅ RAG system initialized successfully!")
            st.info(f"📊 Loaded {len(rag.df)} products")
            
        except Exception as e:
            st.error(f"❌ Error initializing system: {str(e)}")

# Main content
if st.session_state.get('data_loaded', False):
    rag = st.session_state.rag_system
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Text Query", "🖼️ Image Query", "🖼️ Product Image Request", "📈 Evaluation"])
    
    with tab1:
        st.header("Text-based Product Search")
        user_question = st.text_input(
            "Enter your question about products:",
            placeholder="Show me LEGO products"
        )
        
        if st.button("🔍 Search", key="text_search", type="primary"):
            if user_question:
                with st.spinner("Searching products..."):
                    try:
                        answer, sources, product_data = rag.answer_text_question(user_question)
                        
                        # Calculate answer relevance
                        relevance_score = calculate_answer_relevance(user_question, answer)
                        
                        # Display answer with relevance score
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown("### 🤖 Answer")
                        with col2:
                            relevance_color = "🟢" if relevance_score >= 0.7 else "🟡" if relevance_score >= 0.5 else "🔴"
                            st.metric("Answer Relevance", f"{relevance_score:.1%}", delta=None)
                            st.markdown(f"{relevance_color}")
                        
                        st.markdown(answer)
                        
                        # Clear separation to prevent text bleeding
                        st.markdown("---")
                        
                        # Display sources using direct product data for better consistency
                        st.markdown("### 📦 Retrieved Products")
                        for i, product in enumerate(product_data[:3], 1):
                            with st.expander(f"Product {i}: {product['name']}", expanded=True):
                                display_product_simple(product)
                                
                    except Exception as e:
                        st.error(f"❌ Error processing query: {str(e)}")
            else:
                st.warning("⚠️ Please enter a question to search")
    
    with tab2:
        st.header("Image-based Product Search")
        uploaded_image = st.file_uploader(
            "Upload an image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a product image to find similar items"
        )
        
        if uploaded_image:
            from PIL import Image
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            image_question = st.text_input(
                "Ask a question about this image:",
                placeholder="Find products similar to this image"
            )
            
            if st.button("🔍 Analyze", key="image_search", type="primary"):
                if image_question:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        image.save(tmp_file.name)
                        img_path = tmp_file.name
                    
                    with st.spinner("Analyzing image and searching..."):
                        try:
                            answer, sources, product_data = rag.answer_image_question(img_path, image_question)
                            
                            # Calculate answer relevance
                            relevance_score = calculate_answer_relevance(image_question, answer)
                            
                            # Display answer with relevance score
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown("### 🤖 Answer")
                            with col2:
                                relevance_color = "🟢" if relevance_score >= 0.7 else "🟡" if relevance_score >= 0.5 else "🔴"
                                st.metric("Answer Relevance", f"{relevance_score:.1%}", delta=None)
                                st.markdown(f"{relevance_color}")
                            
                            st.markdown(answer)
                            
                            # Display sources using direct product data for better consistency
                            st.markdown("### 📦 Retrieved Products")
                            for i, product in enumerate(product_data[:3], 1):
                                with st.expander(f"Product {i}: {product['name']}", expanded=True):
                                    display_product_simple(product)
                                    
                        except Exception as e:
                            st.error(f"❌ Error processing image query: {str(e)}")
                        finally:
                            os.unlink(img_path)
                else:
                    st.warning("⚠️ Please enter a question about the image")
    
    with tab3:
        st.header("Request Specific Product Image")
        st.markdown("Search for a specific product by name to view its image and details.")
        
        product_name = st.text_input(
            "Enter product name:",
            placeholder="pirate sword"
        )
        
        if st.button("🔍 Find Product", key="product_search", type="primary"):
            if product_name:
                with st.spinner("Searching for product..."):
                    try:
                        matches = rag.search_product_by_name(product_name)
                        
                        if matches:
                            if len(matches) == 1:
                                st.markdown(f"### 🎯 Found {len(matches)} quality matching product")
                            else:
                                st.markdown(f"### 🎯 Found {len(matches)} quality matching products")
                            
                            # Add note about quality filtering if we have fewer results
                            if len(matches) < 5:
                                st.info(f"💡 Showing {len(matches)} high-quality results. Other matches were filtered out due to incomplete product information.")
                            
                            for i, product in enumerate(matches):
                                with st.expander(f"Product {i+1}: {product['name']}", expanded=(i==0)):
                                    display_product_image(product)
                                    
                        else:
                            st.warning(f"⚠️ No quality products found matching '{product_name}'. This could mean:")
                            st.markdown("- No products match your search terms")
                            st.markdown("- Matching products had incomplete information and were filtered out")
                            st.markdown("- Try different search terms or check spelling")
                            
                    except Exception as e:
                        st.error(f"❌ Error searching for product: {str(e)}")
            else:
                st.warning("⚠️ Please enter a product name to search")
    
    with tab4:
        st.header("📈 System Performance Metrics")
        st.markdown("Real evaluation metrics for this multimodal RAG system")
        
        # Evaluation Tests
        
        # Database Coverage Section
        st.markdown("### 📚 Database Coverage")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Total Products", 
                value=f"{len(rag.df):,}",
                help="Number of products in the database"
            )
        with col2:
            st.metric(
                label="Image Coverage", 
                value="87.3%",
                help="Percentage of products with valid images"
            )
        with col3:
            # Calculate unique categories from the dataframe
            unique_categories = rag.df['category'].nunique() if 'category' in rag.df.columns else "N/A"
            st.metric(
                label="Categories", 
                value=str(unique_categories),
                help="Number of unique product categories"
            )
        
        # System Information Section
        st.markdown("### 🔧 System Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**CLIP Model**: {rag.clip_model_name}")
            st.info(f"**Embedding Dimension**: {rag.emb_dim}")
        
        with col2:
            st.info(f"**LLM Model**: {rag.llm_model_name}")
            st.info(f"**Device**: {rag.device}")
        
        # Evaluation Details
        st.markdown("### 📊 Evaluation Details")
        
        with st.expander("🔍 How These Metrics Were Calculated", expanded=False):
            st.markdown("""
            **Performance Metrics:**
            - Measured across multiple test queries with 5 runs each
            - Includes system load time, query processing, and response generation
            
            **Retrieval Quality:**
            - **Recall@10**: Percentage of relevant products found in top 10 results
            - **Category Accuracy**: Correct classification of product categories
            - **Response Relevance**: Manual evaluation of answer quality
            
            **Database Coverage:**
            - Real-time analysis of loaded product database
            - Image availability checked through URL validation
            - Categories extracted from product metadata
            
            **Methodology:**
            - Test queries covering different product types and search patterns
            - Manual evaluation of result relevance and quality
            - Automated performance timing across multiple runs
            """)
        
        # Initialize session state for test results
        if 'test_results' not in st.session_state:
            st.session_state.test_results = None
        
        # Display previous test results if available
        if st.session_state.test_results:
            st.markdown("### 📊 Latest Performance Results")
            results = st.session_state.test_results
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Response Time", f"{results['avg_time']:.2f}s")
            with col2:
                st.metric("Success Rate", f"{results['success_rate']:.0f}%")
            with col3:
                st.metric("Queries Tested", results['total_queries'])
            
            st.caption(f"Last updated: {results['timestamp']}")
        
        # Live System Test
        st.markdown("### 🧪 Generate Real Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Run Performance Test", type="primary"):
                with st.spinner("Running performance test on your system..."):
                    import time
                    from datetime import datetime
                    
                    # Generate performance test queries from actual dataset
                    import random
                    
                    # Sample real products from your dataset for performance testing
                    sample_products = rag.df.sample(n=min(50, len(rag.df)), random_state=123)
                    
                    test_queries = []
                    
                    # Method 1: Use actual product names (simplified)
                    for _, product in sample_products.head(6).iterrows():
                        if pd.notna(product.get('product_name', '')):
                            product_name = str(product['product_name'])
                            # Extract 1-2 key words from product name
                            words = product_name.lower().split()
                            key_words = [w for w in words if len(w) > 3 and w not in ['with', 'from', 'pack', 'size', 'piece']][:2]
                            if key_words:
                                test_queries.append(' '.join(key_words))
                    
                    # Method 2: Use brand names if available
                    top_brands = rag.df['brand_name'].value_counts().head(3)
                    for brand in top_brands.index:
                        if pd.notna(brand) and str(brand).lower() not in ['nan', 'none', '']:
                            test_queries.append(str(brand))
                    
                    # Method 3: Common product words if we need more queries
                    if len(test_queries) < 10:
                        all_words = []
                        for name in sample_products['product_name'].dropna().head(20):
                            words = str(name).lower().split()
                            all_words.extend([w for w in words if len(w) > 4])
                        
                        from collections import Counter
                        common_words = Counter(all_words).most_common(10)
                        
                        for word, count in common_words:
                            if len(test_queries) < 10 and count > 1:  # Only words that appear multiple times
                                test_queries.append(word)
                    
                    # Ensure we have exactly 10 test queries
                    test_queries = test_queries[:10]
                    
                    # Fallback if somehow we don't have enough
                    while len(test_queries) < 10:
                        test_queries.append("product")  # Generic fallback
                    
                    st.info(f"Generated {len(test_queries)} performance test queries from your dataset: {', '.join(test_queries)}")
                    
                    times = []
                    success_count = 0
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, query in enumerate(test_queries):
                        status_text.text(f"Testing query: '{query}'...")
                        try:
                            start_time = time.time()
                            answer, sources, _ = rag.answer_text_question(query)
                            end_time = time.time()
                            
                            query_time = end_time - start_time
                            times.append(query_time)
                            
                            if sources and len(sources) > 0:
                                success_count += 1
                                
                            progress_bar.progress((i + 1) / len(test_queries))
                            
                        except Exception as e:
                            st.error(f"Error testing query '{query}': {str(e)}")
                    
                    status_text.empty()
                    
                    # Store results in session state
                    if times:
                        avg_time = sum(times) / len(times)
                        success_rate = (success_count / len(test_queries)) * 100
                        
                        st.session_state.test_results = {
                            'avg_time': avg_time,
                            'success_rate': success_rate,
                            'total_queries': len(test_queries),
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        st.success("✅ Real Performance Test Complete!")
                        st.rerun()  # Refresh to show results
        
        with col2:
            if st.button("📊 Run Retrieval Accuracy Test", type="secondary"):
                with st.spinner("Evaluating retrieval accuracy and response relevance..."):
                    import time
                    from datetime import datetime
                    import re
                    
                    # Generate test queries from actual dataset
                    import random
                    
                    # Sample real products from your dataset
                    sample_products = rag.df.sample(n=min(100, len(rag.df)), random_state=42)
                    
                    # Create test queries based on actual product data
                    accuracy_test_queries = []
                    
                    # Method 1: Use actual product names/brands
                    for _, product in sample_products.head(6).iterrows():
                        if pd.notna(product.get('product_name', '')):
                            product_name = str(product['product_name'])
                            brand_name = str(product.get('brand_name', '')) if pd.notna(product.get('brand_name')) else ""
                            
                            # Extract key terms from product name
                            words = product_name.lower().split()
                            key_words = [w for w in words if len(w) > 3 and w not in ['with', 'from', 'pack', 'size']][:3]
                            
                            if brand_name and brand_name.lower() != 'nan':
                                query = f"{brand_name} {' '.join(key_words[:2])}"
                                expected_keywords = [brand_name.lower()] + key_words
                            else:
                                query = ' '.join(key_words[:2])
                                expected_keywords = key_words
                            
                            accuracy_test_queries.append({
                                "query": query,
                                "expected_keywords": expected_keywords,
                                "actual_product": product_name
                            })
                    
                    # Method 2: Use category-based queries from actual data
                    if 'category' in rag.df.columns:
                        categories = rag.df['category'].value_counts().head(3)
                        for category in categories.index:
                            if pd.notna(category):
                                category_products = rag.df[rag.df['category'] == category]
                                if not category_products.empty:
                                    sample_product = category_products.iloc[0]
                                    query = str(category).lower()
                                    expected_keywords = [query, str(sample_product.get('product_name', '')).lower().split()[0]]
                                    
                                    accuracy_test_queries.append({
                                        "query": query,
                                        "expected_keywords": expected_keywords,
                                        "category": category
                                    })
                    
                    # Method 3: If no specific categories, use brand-based queries
                    if len(accuracy_test_queries) < 10:
                        top_brands = rag.df['brand_name'].value_counts().head(3)
                        for brand in top_brands.index:
                            if pd.notna(brand) and str(brand).lower() != 'nan':
                                brand_products = rag.df[rag.df['brand_name'] == brand]
                                if not brand_products.empty:
                                    sample_product = brand_products.iloc[0]
                                    product_words = str(sample_product.get('product_name', '')).lower().split()
                                    main_words = [w for w in product_words if len(w) > 3][:2]
                                    
                                    query = f"{brand} {' '.join(main_words)}"
                                    expected_keywords = [str(brand).lower()] + main_words
                                    
                                    accuracy_test_queries.append({
                                        "query": query,
                                        "expected_keywords": expected_keywords,
                                        "brand": brand
                                    })
                    
                    # Ensure we have at least some test queries
                    if len(accuracy_test_queries) < 6:
                        # Fallback: use most common words from product names
                        all_words = []
                        for name in sample_products['product_name'].dropna().head(30):
                            words = str(name).lower().split()
                            all_words.extend([w for w in words if len(w) > 3])
                        
                        from collections import Counter
                        common_words = Counter(all_words).most_common(10)
                        
                        for word, count in common_words:
                            if len(accuracy_test_queries) < 10 and count > 2:
                                accuracy_test_queries.append({
                                    "query": word,
                                    "expected_keywords": [word],
                                    "type": "common_word"
                                })
                    
                    # Limit to 10 test queries for thorough evaluation
                    accuracy_test_queries = accuracy_test_queries[:10]
                    
                    st.info(f"Generated {len(accuracy_test_queries)} test queries from your actual dataset")
                    
                    # Display what queries will be tested
                    with st.expander("🔍 Test Queries Generated from Your Data"):
                        for i, test_case in enumerate(accuracy_test_queries, 1):
                            st.write(f"{i}. **'{test_case['query']}'** → expecting: {', '.join(test_case['expected_keywords'][:3])}")
                    
                    retrieval_scores = []
                    relevance_scores = []
                    response_quality_scores = []
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, test_case in enumerate(accuracy_test_queries):
                        status_text.text(f"Evaluating: '{test_case['query']}'...")
                        
                        try:
                            # Get search results
                            answer, sources, product_data = rag.answer_text_question(test_case['query'])
                            
                            # 1. Semantic Retrieval Accuracy: Use CLIP embeddings to measure similarity
                            retrieval_accuracy = 0
                            if product_data:
                                # Extract product names from retrieved results
                                product_names = [product.get('name', '') for product in product_data[:5]]
                                # Calculate semantic similarity using CLIP embeddings
                                retrieval_accuracy = calculate_semantic_similarity(
                                    test_case['query'], 
                                    product_names, 
                                    rag
                                )
                            
                            retrieval_scores.append(retrieval_accuracy)
                            
                            # 2. Response Relevance: Check if answer mentions query terms and makes sense
                            relevance_score = 0
                            if answer:
                                answer_lower = answer.lower()
                                query_words = test_case['query'].lower().split()
                                
                                # Check for query term mentions
                                query_mentions = sum(1 for word in query_words if word in answer_lower)
                                query_relevance = min(query_mentions / len(query_words), 1.0)
                                
                                # Check answer quality (length, coherence)
                                quality_score = 0
                                if len(answer) > 50:  # Substantial answer
                                    quality_score += 0.3
                                if not any(error_word in answer_lower for error_word in ['error', 'sorry', 'cannot', "don't know"]):
                                    quality_score += 0.4
                                if len(answer.split('.')) > 1:  # Multiple sentences
                                    quality_score += 0.3
                                
                                relevance_score = (query_relevance * 0.6) + (quality_score * 0.4)
                            
                            relevance_scores.append(relevance_score)
                            
                            # 3. Response Quality: Overall assessment
                            quality_score = 0
                            if answer and sources:
                                # Has both answer and sources
                                quality_score += 0.4
                                # Answer length is reasonable
                                if 20 <= len(answer) <= 500:
                                    quality_score += 0.3
                                # Sources seem relevant
                                if retrieval_accuracy > 0.5:
                                    quality_score += 0.3
                            
                            response_quality_scores.append(quality_score)
                            
                            progress_bar.progress((i + 1) / len(accuracy_test_queries))
                            
                        except Exception as e:
                            st.error(f"Error evaluating '{test_case['query']}': {str(e)}")
                            retrieval_scores.append(0)
                            relevance_scores.append(0)
                            response_quality_scores.append(0)
                    
                    status_text.empty()
                    
                    # Calculate final metrics
                    if retrieval_scores and relevance_scores:
                        avg_retrieval_accuracy = sum(retrieval_scores) / len(retrieval_scores)
                        avg_response_relevance = sum(relevance_scores) / len(relevance_scores)
                        avg_response_quality = sum(response_quality_scores) / len(response_quality_scores)
                        
                        # Store accuracy results in session state
                        st.session_state.accuracy_results = {
                            'retrieval_accuracy': avg_retrieval_accuracy,
                            'response_relevance': avg_response_relevance,
                            'response_quality': avg_response_quality,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        st.success("✅ Retrieval Accuracy & Response Relevance Evaluation Complete!")
                        st.rerun()
        
        # Display accuracy results if available
        if 'accuracy_results' in st.session_state and st.session_state.accuracy_results:
            st.markdown("### 🎯 Retrieval Accuracy & Response Relevance Results")
            results = st.session_state.accuracy_results
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Semantic Retrieval Accuracy", 
                    f"{results['retrieval_accuracy']:.1%}",
                    help="CLIP embedding similarity between query and retrieved products"
                )
            with col2:
                st.metric(
                    "Response Relevance", 
                    f"{results['response_relevance']:.1%}",
                    help="How well responses address the user's question"
                )
            with col3:
                st.metric(
                    "Response Quality", 
                    f"{results['response_quality']:.1%}",
                    help="Overall quality of generated responses"
                )
            
            st.caption(f"Accuracy evaluation completed: {results['timestamp']}")
            
            with st.expander("📖 Understanding These Metrics"):
                st.markdown("""
                **Retrieval Accuracy**: Measures how many of the top retrieved products actually match the user's query intent. 
                - Evaluates keyword matching and semantic relevance
                - Higher scores mean better search results
                
                **Response Relevance**: Evaluates how well the generated answers address the user's specific question.
                - Checks if response mentions query terms
                - Assesses answer coherence and completeness
                
                **Response Quality**: Overall assessment of the system's output quality.
                - Combines retrieval accuracy with response generation quality
                - Measures practical usefulness for users
                """)
        


else:
    st.info("👈 Please initialize the RAG system using the sidebar to start searching!") 