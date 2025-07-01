import streamlit as st
import sqlite3
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import re
import faiss

# Configuration
DB_PATH = "mosdac_cache.db"
MODEL_NAME = "all-MiniLM-L6-v2"

# Hardcoded MOSDAC knowledge base
MOSDAC_KNOWLEDGE = [
    {"text": "MOSDAC currently utilizes data from 4 primary satellites: Oceansat-2, ScatSat-1, INSAT-3D, and INSAT-3DR.", 
     "url": "https://mosdac.gov.in/satellites"},
    {"text": "Number of satellites: MOSDAC uses 4 operational satellites for ocean and atmospheric monitoring.", 
     "url": "https://mosdac.gov.in/satellites"},
    {"text": "Oceansat-2 provides ocean color data and sea surface temperature with 1km resolution.", 
     "url": "https://mosdac.gov.in/oceansat2"},
    {"text": "ScatSat-1 measures ocean surface winds with 25km resolution.", 
     "url": "https://mosdac.gov.in/scatsat1"},
    {"text": "To access data: 1) Register at MOSDAC portal 2) Use data discovery tool 3) Download datasets", 
     "url": "https://mosdac.gov.in/registration"},
    {"text": "Key data products: Sea surface temperature, chlorophyll concentration, wind vectors", 
     "url": "https://mosdac.gov.in/dataproducts"}
]

def initialize_database():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS embeddings
                 (id INTEGER PRIMARY KEY, text TEXT, url TEXT, embedding BLOB)''')
    conn.commit()
    conn.close()

def cache_embeddings():
    model = SentenceTransformer(MODEL_NAME)
    initialize_database()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM embeddings")
    
    for doc in MOSDAC_KNOWLEDGE:
        embedding = model.encode(doc["text"]).tobytes()
        c.execute("INSERT INTO embeddings (text, url, embedding) VALUES (?, ?, ?)",
                 (doc["text"], doc["url"], embedding))
    conn.commit()
    conn.close()

class MOSDACRetriever:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH)
        self.cursor = self.conn.cursor()
        self.cursor.execute("SELECT text, url FROM embeddings")
        self.docs = self.cursor.fetchall()
        self.texts = [doc[0] for doc in self.docs]
        
        # BM25 setup
        tokenized_corpus = [self.tokenize(text) for text in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # FAISS setup
        self.cursor.execute("SELECT embedding FROM embeddings")
        embeddings = [np.frombuffer(row[0], dtype=np.float32) for row in self.cursor.fetchall()]
        if embeddings:
            self.index = faiss.IndexFlatL2(len(embeddings[0]))
            self.index.add(np.array(embeddings))
    
    def tokenize(self, text):
        return re.findall(r'\w+', text.lower())
    
    def retrieve(self, query, k=3):
        # BM25 search
        tokenized_query = self.tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_results = [(i, score) for i, score in enumerate(bm25_scores)]
        bm25_results.sort(key=lambda x: x[1], reverse=True)
        
        # Vector search
        if hasattr(self, 'index'):
            query_embedding = SentenceTransformer(MODEL_NAME).encode([query])[0]
            distances, indices = self.index.search(np.array([query_embedding]), k)
            vector_results = [(int(i), float(1/(d+1e-9))) for d, i in zip(distances[0], indices[0])]
        else:
            vector_results = []
        
        # Combine results
        combined = {}
        for i, score in bm25_results[:k]:
            combined[i] = combined.get(i, 0) + score * 0.5
        for i, score in vector_results:
            combined[i] = combined.get(i, 0) + score * 0.5
            
        top_indices = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
        return [(self.texts[i], self.docs[i][1]) for i, _ in top_indices]

def main():
    st.set_page_config(page_title="MOSDAC Help-Bot", page_icon="🛰️")
    
    # Initialize
    if 'retriever' not in st.session_state:
        with st.spinner("Loading MOSDAC knowledge..."):
            cache_embeddings()
            st.session_state.retriever = MOSDACRetriever()
            st.session_state.messages = [{"role": "assistant", "content": "✅ MOSDAC Help-Bot ready! Ask me about:" 
                                        "\n- Satellite count\n- Data access\n- Specific satellites"}]
    
    # Chat interface
    st.title("🛰️ MOSDAC Satellite Help")
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    if prompt := st.chat_input("Your question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("Searching MOSDAC resources..."):
            try:
                # Direct answers for common questions
                lower_prompt = prompt.lower()
                if any(g in lower_prompt for g in ["hi", "hello"]):
                    response = "Hello! I'm MOSDAC Help-Bot. Ask about:\n- Satellite count\n- Data access\n- Specific satellites"
                elif "how many satellite" in lower_prompt:
                    response = "MOSDAC uses 4 satellites:\n1. Oceansat-2\n2. ScatSat-1\n3. INSAT-3D\n4. INSAT-3DR\n[Source](https://mosdac.gov.in/satellites)"
                elif "access data" in lower_prompt:
                    response = "Data access steps:\n1. [Register here](https://mosdac.gov.in/registration)\n2. Use data discovery tool\n3. Download datasets"
                else:
                    results = st.session_state.retriever.retrieve(prompt)
                    if results:
                        response = "From MOSDAC resources:\n\n" + "\n\n".join([f"- {text}\n  [Source]({url})" for text, url in results])
                    else:
                        response = "I couldn't find specific information. Try asking about:\n- MOSDAC satellites\n- Data products\n- Registration process"
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
                
            except Exception as e:
                st.error(f"System error: {str(e)}")
                st.stop()

if __name__ == "__main__":
    main()