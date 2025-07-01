import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import re
import requests

model = SentenceTransformer('all-MiniLM-L6-v2')

class HybridRetriever:
    def __init__(self, db_path='mosdac_cache.db'):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("SELECT text FROM embeddings")
        self.texts = [row[0] for row in self.cursor.fetchall()]
        self.bm25 = BM25Okapi([self.tokenize(t) for t in self.texts])
        self.load_faiss_index()
        
    def tokenize(self, text):
        return re.findall(r'\w+', text.lower())
    
    def load_faiss_index(self):
        self.cursor.execute("SELECT embedding FROM embeddings")
        embeddings = [np.frombuffer(row[0], dtype=np.float32) for row in self.cursor.fetchall()]
        dimension = len(embeddings[0])
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
    
    def reciprocal_rank_fusion(self, bm25_scores, vector_scores, k=3, alpha=0.5):
        combined_scores = {}
        for doc_id, score in bm25_scores:
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + alpha * score
            
        for doc_id, score in vector_scores:
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + (1 - alpha) * score
            
        return sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    
    def retrieve(self, query, k=3):
        # BM25 retrieval
        tokenized_query = self.tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_results = [(i, score) for i, score in enumerate(bm25_scores)]
        bm25_results.sort(key=lambda x: x[1], reverse=True)
        bm25_top = bm25_results[:k]
        
        # Vector retrieval
        query_embedding = model.encode([query])[0]
        D, I = self.index.search(np.array([query_embedding]).astype('float32'), k)
        vector_results = [(int(i), float(1/(d+1e-9))) for d, i in zip(D[0], I[0])]  # Convert distance to similarity
        
        # Hybrid fusion
        fused_results = self.reciprocal_rank_fusion(bm25_top, vector_results, k)
        return [self.texts[doc_id] for doc_id, _ in fused_results]

class RAGSystem:
    def __init__(self, retriever):
        self.retriever = retriever
    
    def generate_response(self, query):
        context = "\n".join(self.retriever.retrieve(query))
        
        # DeepSeek API call (OpenAI-compatible)
        api_url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Authorization": "Bearer sk-c479bc29e306477094c92e69cb69a9f8",  # Replace with actual key
            "Content-Type": "application/json"
        }
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are MOSDAC Help-Bot. Answer using provided context. Cite sources."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ]
        }
        
        try:
            response = requests.post(api_url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error: {str(e)}"