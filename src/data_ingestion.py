import os
import sqlite3
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
import faiss

# Initialize models
model = SentenceTransformer('all-MiniLM-L6-v2')

def download_mosdac_data():
    """Simulate downloading MOSDAC documents"""
    sample_data = [
        {"url": "https://mosdac.gov.in/dataproducts", "text": "MOSDAC provides satellite-derived ocean data including sea surface temperature, chlorophyll concentration, and wind vectors. Data available for Indian Ocean region from 2010-present."},
        {"url": "https://mosdac.gov.in/snow-cover", "text": "Daily snow cover products at 500m resolution for Himalayan region. Derived from MODIS and INSAT-3D satellites. Includes fractional snow cover and snow albedo."},
        {"url": "https://mosdac.gov.in/tropical-cyclones", "text": "Tropical cyclone heat potential products updated daily during cyclone season. Uses multi-satellite data to estimate ocean thermal energy."}
    ]
    return sample_data

def chunk_documents(docs):
    """Split documents with overlapping chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=100,
        length_function=len
    )
    chunks = []
    for doc in docs:
        chunks.extend(splitter.split_text(doc['text']))
    return chunks

def create_vector_index(chunks):
    """Create FAISS vector index"""
    embeddings = model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index

def cache_embeddings(chunks, db_path='mosdac_cache.db'):
    """Cache embeddings in SQLite"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS embeddings
                 (id INTEGER PRIMARY KEY, text TEXT, url TEXT, embedding BLOB)''')
    
    for i, chunk in enumerate(tqdm(chunks)):
        embedding = model.encode([chunk])[0].tobytes()
        c.execute("INSERT INTO embeddings (text, url, embedding) VALUES (?, ?, ?)",
                 (chunk, "https://mosdac.gov.in", embedding))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    print("🚀 Downloading MOSDAC sample data...")
    docs = download_mosdac_data()
    
    print("🔪 Chunking documents...")
    chunks = chunk_documents(docs)
    
    print("📦 Caching embeddings...")
    cache_embeddings(chunks)
    
    print("✅ Data ingestion complete! Saved to mosdac_cache.db")