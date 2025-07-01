import streamlit as st
import sqlite3
import numpy as np
import re
import requests
import faiss
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import logging

# Configuration
DB_PATH = "mosdac_cache.db"
MODEL_NAME = "all-MiniLM-L6-v2"
MOSDAC_BASE_URL = "https://mosdac.gov.in"

# Initialize models
model = SentenceTransformer(MODEL_NAME)

# --------------------------
# Web Crawler & Content Extraction
# --------------------------
def fetch_mosdac_content():
    """Fetch content with enhanced error handling"""
    try:
        # Sample data - replace with actual crawling if needed
        return [
            {
                "url": f"{MOSDAC_BASE_URL}/satellites",
                "text": "MOSDAC operates 4 satellites: Oceansat-2 (ocean color), ScatSat-1 (winds), INSAT-3D and INSAT-3DR (atmospheric monitoring)"
            },
            {
                "url": f"{MOSDAC_BASE_URL}/registration",
                "text": "To access data: 1) Register on MOSDAC portal 2) Use data discovery tool 3) Download datasets in NetCDF or HDF5 formats"
            },
            {
                "url": f"{MOSDAC_BASE_URL}/dataproducts",
                "text": "Available products: Sea Surface Temperature, Chlorophyll-a, Wind Vectors, Ocean Heat Content"
            }
        ]
    except Exception as e:
        logging.error(f"Content fetch failed: {str(e)}")
        return []  # Return empty list to handle gracefully

# --------------------------
# Database Functions
# --------------------------
def initialize_database():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS embeddings
                    (id INTEGER PRIMARY KEY, text TEXT, url TEXT, embedding BLOB)''')
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return False
    finally:
        conn.close()

# --------------------------
# Data Processing
# --------------------------
def process_and_cache_data():
    """Process and cache data with progress indicators"""
    with st.spinner("Loading MOSDAC data..."):
        if not initialize_database():
            return False
        
        docs = fetch_mosdac_content()
        if not docs:
            st.error("No content available")
            return False

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("DELETE FROM embeddings")
        
        for doc in docs:
            try:
                embedding = model.encode(doc["text"]).tobytes()
                c.execute("INSERT INTO embeddings (text, url, embedding) VALUES (?, ?, ?)",
                         (doc["text"], doc["url"], embedding))
            except Exception as e:
                logging.error(f"Error processing document: {str(e)}")
                continue
        
        conn.commit()
        conn.close()
        return True

# --------------------------
# Retrieval System
# --------------------------
class MOSDACRetriever:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH)
        self._initialize_retrieval()
        
    def _initialize_retrieval(self):
        self.cursor = self.conn.cursor()
        self.cursor.execute("SELECT id, text, url FROM embeddings")
        self.data = self.cursor.fetchall()
        self.texts = [row[1] for row in self.data]
        
        # Initialize BM25
        tokenized_corpus = [re.findall(r'\w+', text.lower()) for text in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Initialize FAISS
        self.cursor.execute("SELECT embedding FROM embeddings")
        embeddings = [np.frombuffer(row[0], dtype=np.float32) for row in self.cursor.fetchall()]
        if embeddings:
            self.index = faiss.IndexFlatL2(len(embeddings[0]))
            self.index.add(np.array(embeddings))
        else:
            self.index = None
    
    def retrieve(self, query, k=3):
        try:
            # BM25 retrieval
            tokenized_query = re.findall(r'\w+', query.lower())
            bm25_scores = self.bm25.get_scores(tokenized_query)
            bm25_results = [(i, score) for i, score in enumerate(bm25_scores)]
            
            # Vector retrieval
            if self.index:
                query_embedding = model.encode([query])[0]
                D, I = self.index.search(np.array([query_embedding]).astype('float32'), k)
                vector_results = [(int(i), float(1/(d+1e-9))) for d, i in zip(D[0], I[0])]
            else:
                vector_results = []
            
            # Combine results
            combined = {i: score*0.5 for i, score in bm25_results}
            for i, score in vector_results:
                combined[i] = combined.get(i, 0) + score*0.5
                
            top_indices = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
            return [(self.texts[i], self.data[i][2]) for i, _ in top_indices]
        except Exception as e:
            logging.error(f"Retrieval error: {str(e)}")
            return []

# --------------------------
# Response Generation
# --------------------------
def generate_response(query):
    """Generate response with fallbacks"""
    # Predefined responses
    greetings = ["hi", "hello", "hey"]
    if query.lower().strip() in greetings:
        return "Hello! I'm MOSDAC Help-Bot. Ask about satellites or data access."
    
    common_questions = {
        "satellites": "MOSDAC uses:\n1. Oceansat-2\n2. ScatSat-1\n3. INSAT-3D\n4. INSAT-3DR\n[Learn more](https://mosdac.gov.in/satellites)",
        "access data": "Data access steps:\n1. [Register](https://mosdac.gov.in/registration)\n2. Use discovery tool\n3. Download datasets",
        "products": "Available products:\n- Sea Surface Temp\n- Chlorophyll\n- Wind Vectors\n[Details](https://mosdac.gov.in/dataproducts)"
    }
    
    for key, response in common_questions.items():
        if key in query.lower():
            return response
    
    # Semantic search fallback
    try:
        retriever = st.session_state.retriever
        results = retriever.retrieve(query)
        if results:
            return "From MOSDAC:\n\n" + "\n\n".join([f"- {text}\n  [Source]({url})" for text, url in results])
        return "I couldn't find information on that. Try asking about satellites or data access."
    except Exception as e:
        logging.error(f"Response error: {str(e)}")
        return "Sorry, I encountered an error. Please try again."

# --------------------------
# Streamlit UI
# --------------------------
def main():
    st.set_page_config(
        page_title="MOSDAC Help-Bot",
        page_icon="🛰️",
        layout="centered"
    )
    
    # Initialize system
    if 'retriever' not in st.session_state:
        if process_and_cache_data():
            st.session_state.retriever = MOSDACRetriever()
            st.session_state.messages = [{
                "role": "assistant",
                "content": "✅ MOSDAC Help-Bot ready! Ask about:\n- Satellites\n- Data access\n- Available products"
            }]
        else:
            st.error("Failed to initialize system")
            st.stop()
    
    # Chat interface
    st.title("🛰️ MOSDAC Help-Bot")
    
    # Display messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Handle input
    if prompt := st.chat_input("Ask about MOSDAC..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("Thinking..."):
            response = generate_response(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

if __name__ == "__main__":
    main()