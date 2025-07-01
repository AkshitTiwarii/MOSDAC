import streamlit as st
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import re
import requests
import os

# Initialize models
model = SentenceTransformer('all-MiniLM-L6-v2')

# Simple retriever class
class HybridRetriever:
    def __init__(self, db_path='mosdac_cache.db'):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("SELECT text FROM embeddings")
        self.texts = [row[0] for row in self.cursor.fetchall()]
        self.bm25 = BM25Okapi([self.tokenize(t) for t in self.texts])
        
    def tokenize(self, text):
        return re.findall(r'\w+', text.lower())
    
    def retrieve(self, query, k=3):
        # Simple retrieval without FAISS for now
        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:k]
        return [self.texts[i] for i in top_indices]

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
    try:
        st.session_state.retriever = HybridRetriever()
    except Exception as e:
        st.error(f"Failed to initialize retriever: {str(e)}")
        st.session_state.retriever = None

# UI Layout
st.title("🌊 MOSDAC Help-Bot")

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    
if prompt := st.chat_input("Ask about satellite data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    if st.session_state.retriever is None:
        response = "System not initialized. Please check logs."
    else:
        try:
            context = "\n".join(st.session_state.retriever.retrieve(prompt))
            
            # Simple response generation without API call
            response = f"I found information about: {context[:200]}..."
        except Exception as e:
            response = f"Error: {str(e)}"
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)

st.info("DeepSeek API integration would be enabled with valid API key")