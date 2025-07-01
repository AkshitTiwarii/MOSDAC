import streamlit as st
import sqlite3
import numpy as np
import re
import requests
import faiss
import json
import logging
import os
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium, folium_static  # Added folium_static
from PIL import Image
import sys
import subprocess
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Configuration
DB_PATH = "mosdac_cache.db"
MODEL_NAME = "all-MiniLM-L6-v2"
MOSDAC_BASE_URL = "https://mosdac.gov.in"
KNOWLEDGE_GRAPH_FILE = "mosdac_knowledge_graph.json"
LOGO_PATH = "WhatsApp Image 2025-07-02 at 01.26.33_6d36af89.jpg"  # Updated to your filename

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --------------------------
# Model Loading
# --------------------------
def load_embedding_model():
    """Robust model loading using sentence-transformers"""
    try:
        model = SentenceTransformer(MODEL_NAME)
        logging.info(f"Successfully loaded model: {MODEL_NAME}")
        return model
    except Exception as e:
        logging.error(f"Primary load failed: {str(e)}. Trying CPU-only load...")
        try:
            model = SentenceTransformer(MODEL_NAME, device='cpu')
            logging.warning("Loaded model to CPU as fallback")
            return model
        except Exception as fallback_error:
            logging.critical(f"CPU load failed: {str(fallback_error)}")
            st.error("CRITICAL: Model loading failed on all methods")
            st.stop()

# Initialize model
model = load_embedding_model()

# --------------------------
# Text Embedding Function
# --------------------------
def get_embeddings(texts):
    """Generate embeddings using the loaded model"""
    try:
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings
    except Exception as e:
        logging.error(f"Embedding generation error: {str(e)}")
        return np.zeros((len(texts), 384))

geolocator = Nominatim(user_agent="mosdac_bot", timeout=10)

# --------------------------
# Web Crawler & Content Extraction
# --------------------------
def fetch_mosdac_content():
    """Fetch content from MOSDAC with geospatial awareness"""
    try:
        return [
            {
                "url": f"{MOSDAC_BASE_URL}/en/satellite-missions",
                "text": "MOSDAC operates satellites including Oceansat-2 for ocean color monitoring, ScatSat-1 for wind measurements, and INSAT-3D/3DR for atmospheric studies. These satellites provide data over the Indian Ocean region with resolutions up to 1km.",
                "entities": ["Oceansat-2", "ScatSat-1", "INSAT-3D", "INSAT-3DR", "Indian Ocean"],
                "geo_coords": [20.5937, 78.9629]  # India coordinates
            },
            {
                "url": f"{MOSDAC_BASE_URL}/en/user-registration",
                "text": "To access MOSDAC data: Register on the portal, use the data discovery tool, and download datasets in NetCDF or HDF5 formats. Academic users can register for free access.",
                "entities": ["registration", "data discovery", "NetCDF", "HDF5"],
                "geo_coords": None
            },
            {
                "url": f"{MOSDAC_BASE_URL}/en/data-products",
                "text": "MOSDAC offers products like Sea Surface Temperature (SST), Chlorophyll-a concentration, Wind Vectors, and Ocean Heat Content. Data is available from 2010 to present at daily intervals.",
                "entities": ["SST", "Chlorophyll-a", "Wind Vectors", "Ocean Heat Content"],
                "geo_coords": [20.5937, 78.9629]
            },
            {
                "url": f"{MOSDAC_BASE_URL}/en/tropical-cyclone",
                "text": "Cyclone prediction products are updated daily during storm seasons using INSAT-3D data to monitor storm development in Bay of Bengal and Arabian Sea regions.",
                "entities": ["cyclone", "INSAT-3D", "Bay of Bengal", "Arabian Sea"],
                "geo_coords": [13.0827, 80.2707]  # Chennai
            },
            {
                "url": f"{MOSDAC_BASE_URL}/en/data-services",
                "text": "MOSDAC provides data services including satellite imagery visualization, data subsetting, and time-series analysis. APIs are available for programmatic access to oceanographic datasets.",
                "entities": ["data services", "satellite imagery", "data subsetting", "time-series analysis"],
                "geo_coords": None
            }
        ]
    except Exception as e:
        logging.error(f"Content fetch failed: {str(e)}")
        return []

# --------------------------
# Knowledge Graph Construction
# --------------------------
def build_knowledge_graph(docs):
    """Create knowledge graph from documents"""
    knowledge_graph = {
        "entities": {},
        "relationships": []
    }
    
    try:
        for doc in docs:
            for entity in doc.get("entities", []):
                if entity not in knowledge_graph["entities"]:
                    knowledge_graph["entities"][entity] = {
                        "type": "concept",
                        "mentions": [],
                        "geo": doc.get("geo_coords")
                    }
                knowledge_graph["entities"][entity]["mentions"].append({
                    "source": doc["url"],
                    "text": doc["text"]
                })
            
            if len(doc.get("entities", [])) > 1:
                for i in range(len(doc["entities"]) - 1):
                    for j in range(i+1, len(doc["entities"])):
                        relationship = {
                            "source": doc["entities"][i],
                            "target": doc["entities"][j],
                            "type": "related",
                            "source_url": doc["url"]
                        }
                        knowledge_graph["relationships"].append(relationship)
        
        with open(KNOWLEDGE_GRAPH_FILE, 'w') as f:
            json.dump(knowledge_graph, f, indent=2)
            
        return knowledge_graph
    except Exception as e:
        logging.error(f"Knowledge graph error: {str(e)}")
        return None

# --------------------------
# Geospatial Processing
# --------------------------
def extract_geo_entities(query):
    """Extract geographic entities from query"""
    try:
        location = geolocator.geocode(query, exactly_one=True)
        if location:
            return {
                "name": location.address,
                "coords": [location.latitude, location.longitude]
            }
        return None
    except Exception as e:
        logging.error(f"Geolocation error: {str(e)}")
        return None

def generate_folium_map(coords, area_name):
    """Generate interactive Folium map with proper attribution"""
    try:
        m = folium.Map(
            location=coords, 
            zoom_start=7,
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri, Maxar, Earthstar Geographics',
            control_scale=True
        )
        
        # Add satellite imagery layer
        folium.TileLayer(
            tiles='OpenStreetMap',
            attr='OpenStreetMap contributors',
            name='Street Map'
        ).add_to(m)
        
        folium.Marker(
            location=coords,
            popup=f"<b>MOSDAC Coverage:</b> {area_name}",
            tooltip=area_name,
            icon=folium.Icon(color="red", icon="satellite", prefix="fa")
        ).add_to(m)
        
        folium.Circle(
            location=coords,
            radius=200000,  # 200km radius
            color="#3186cc",
            fill=True,
            fill_color="#3186cc",
            fill_opacity=0.2,
            popup=f"Satellite Coverage Area: {area_name}"
        ).add_to(m)
        
        folium.LayerControl().add_to(m)
        return m
    except Exception as e:
        logging.error(f"Map generation error: {str(e)}")
        return None

# --------------------------
# Database Functions
# --------------------------
def initialize_database():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS embeddings
                    (id INTEGER PRIMARY KEY, text TEXT, url TEXT, 
                    entities TEXT, geo_coords TEXT, embedding BLOB)''')
        conn.commit()
        return True
    except Exception as e:
        logging.error(f"Database error: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()

def clear_database():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''DELETE FROM embeddings''')
        conn.commit()
        return True
    except Exception as e:
        logging.error(f"Database clear error: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()

# --------------------------
# Data Processing
# --------------------------
def process_and_cache_data():
    """Process and cache data with knowledge graph"""
    with st.spinner("Building MOSDAC knowledge base..."):
        if not initialize_database():
            return False
        
        clear_database()  # Clear existing data
        
        docs = fetch_mosdac_content()
        if not docs:
            st.error("No content available")
            return False

        build_knowledge_graph(docs)
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        texts = [doc["text"] for doc in docs]
        embeddings = get_embeddings(texts)
        
        for i, doc in enumerate(docs):
            try:
                embedding_bytes = embeddings[i].tobytes()
                entities = json.dumps(doc.get("entities", []))
                geo_coords = json.dumps(doc.get("geo_coords"))
                
                c.execute('''INSERT INTO embeddings 
                            (text, url, entities, geo_coords, embedding) 
                            VALUES (?, ?, ?, ?, ?)''',
                         (doc["text"], doc["url"], entities, geo_coords, embedding_bytes))
            except Exception as e:
                logging.error(f"Error processing document: {str(e)}")
                continue
        
        conn.commit()
        conn.close()
        return True

# --------------------------
# Retrieval System with Geospatial Support
# --------------------------
class MOSDACRetriever:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH)
        self._initialize_retrieval()
        
    def _initialize_retrieval(self):
        self.cursor = self.conn.cursor()
        self.cursor.execute("SELECT id, text, url, entities, geo_coords FROM embeddings")
        self.data = self.cursor.fetchall()
        
        if not self.data:
            self.texts = []
            self.entity_lists = []
            self.geo_coords = []
            self.bm25 = None
            self.index = None
            self.knowledge_graph = None
            return
            
        self.texts = [row[1] for row in self.data]
        self.entity_lists = [json.loads(row[3]) if row[3] else [] for row in self.data]
        self.geo_coords = [json.loads(row[4]) if row[4] else None for row in self.data]
        
        tokenized_corpus = [re.findall(r'\w+', text.lower()) for text in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        self.cursor.execute("SELECT embedding FROM embeddings")
        embeddings = [np.frombuffer(row[0], dtype=np.float32) for row in self.cursor.fetchall()]
        if embeddings:
            self.index = faiss.IndexFlatL2(len(embeddings[0]))
            self.index.add(np.array(embeddings))
        else:
            self.index = None
            
        try:
            if os.path.exists(KNOWLEDGE_GRAPH_FILE):
                with open(KNOWLEDGE_GRAPH_FILE, 'r') as f:
                    self.knowledge_graph = json.load(f)
            else:
                self.knowledge_graph = None
        except Exception as e:
            logging.error(f"Knowledge graph load error: {str(e)}")
            self.knowledge_graph = None
    
    def retrieve(self, query, k=5):
        try:
            geo_entity = extract_geo_entities(query)
            
            tokenized_query = re.findall(r'\w+', query.lower())
            bm25_scores = self.bm25.get_scores(tokenized_query) if self.bm25 else []
            
            vector_results = []
            if self.index:
                # Generate query embedding
                query_embedding = get_embeddings([query])[0]
                D, I = self.index.search(np.array([query_embedding]).astype('float32'), k)
                vector_results = [(int(i), float(1/(d+1e-9))) for d, i in zip(D[0], I[0])]
            
            combined = {}
            for i, score in enumerate(bm25_scores):
                boost = 1.0
                
                # Geo boost
                if geo_entity and self.geo_coords[i]:
                    try:
                        dist = np.linalg.norm(np.array(geo_entity["coords"]) - np.array(self.geo_coords[i]))
                        boost += 3.0 / (dist + 1)  # Higher boost for closer locations
                    except:
                        pass
                
                # Entity boost
                if self.knowledge_graph:
                    for entity in self.entity_lists[i]:
                        if entity.lower() in query.lower():
                            boost += 2.0
                
                combined[i] = score * boost
            
            # Add vector results
            for i, score in vector_results:
                combined[i] = combined.get(i, 0) + score * 0.7  # Higher weight for semantic similarity
                
            top_indices = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
            return [
                {
                    "text": self.texts[i],
                    "url": self.data[i][2],
                    "entities": self.entity_lists[i],
                    "geo_coords": self.geo_coords[i]
                } 
                for i, _ in top_indices
            ]
        except Exception as e:
            logging.error(f"Retrieval error: {str(e)}")
            return []

# --------------------------
# Free AI APIs Integration
# --------------------------
def get_ai_response(query, context):
    """Get response from free AI APIs"""
    # Fallback response
    fallback_response = f"Here's what I found:\n\n{context}"
    
    # Try DeepSeek API first
    try:
        if "DEEPSEEK_API_KEY" in st.secrets:
            headers = {"Authorization": f"Bearer {st.secrets['DEEPSEEK_API_KEY']}"}
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "You are MOSDAC Help-Bot. Answer concisely using only the provided context."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
                ],
                "temperature": 0.2,
                "max_tokens": 300
            }
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=10
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logging.error(f"DeepSeek API error: {str(e)}")
    
    # Try HuggingFace API as fallback
    try:
        API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-xxl"
        headers = {"Authorization": f"Bearer {st.secrets.get('HF_API_TOKEN', '')}"}
        
        payload = {
            "inputs": f"Answer this question based only on context: {query}\n\nContext: {context}",
            "parameters": {
                "max_length": 300,
                "temperature": 0.3,
                "do_sample": True
            }
        }
        
        response = requests.post(API_URL, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            return response.json()[0]['generated_text']
    except Exception as e:
        logging.error(f"HuggingFace API error: {str(e)}")
    
    return fallback_response

# --------------------------
# Response Generation
# --------------------------
def generate_response(query):
    """Generate response with multi-stage processing"""
    # Handle greetings
    greetings = ["hi", "hello", "hey", "greetings", "hola", "namaste"]
    if query.lower().strip() in greetings:
        return "Hello! I'm MOSDAC Help-Bot. Ask about satellites, data access, or geospatial queries."
    
    # Handle common questions
    common_questions = {
        "satellites": "MOSDAC uses several satellites including:\n\n- **Oceansat-2**: Ocean color monitoring\n- **ScatSat-1**: Wind measurements\n- **INSAT-3D/3DR**: Atmospheric studies\n\n[Learn more about satellite missions](https://mosdac.gov.in/en/satellite-missions)",
        "access data": "To access MOSDAC data:\n\n1. [Register on MOSDAC portal](https://mosdac.gov.in/en/user-registration)\n2. Use the data discovery tool\n3. Download datasets in NetCDF or HDF5 formats\n\nAcademic users can register for free access.",
        "products": "Available MOSDAC products include:\n\n- Sea Surface Temperature (SST)\n- Chlorophyll-a concentration\n- Wind Vectors\n- Ocean Heat Content\n\n[Explore all data products](https://mosdac.gov.in/en/data-products)",
        "chennai": "For Chennai region:\n\n- Satellite: INSAT-3D\n- Products: Cyclone prediction, Sea Surface Temperature\n- Resolution: 1km\n[Chennai-specific data](https://mosdac.gov.in/en/tropical-cyclone)",
        "registration": "To register for MOSDAC access:\n\n1. Visit [User Registration page](https://mosdac.gov.in/en/user-registration)\n2. Fill out the required information\n3. Academic users should use their institutional email\n4. You'll receive access credentials within 24 hours"
    }
    
    for key, response in common_questions.items():
        if key in query.lower():
            return response
    
    # Handle geospatial queries
    geo_entity = extract_geo_entities(query)
    if geo_entity:
        try:
            # Store map in session state to prevent disappearing
            if 'map_obj' not in st.session_state or st.session_state.map_entity != geo_entity["name"]:
                st.session_state.map_obj = generate_folium_map(geo_entity["coords"], geo_entity["name"])
                st.session_state.map_entity = geo_entity["name"]
            
            if st.session_state.map_obj:
                with st.expander(f"🗺️ Interactive Satellite Coverage Map: {geo_entity['name']}"):
                    # Use folium_static to prevent map from disappearing
                    folium_static(st.session_state.map_obj, width=700, height=400)
                return f"MOSDAC provides comprehensive satellite data coverage for **{geo_entity['name']}**. Explore the interactive map above."
            else:
                return f"MOSDAC provides satellite data coverage for **{geo_entity['name']}** at coordinates {geo_entity['coords']}."
        except Exception as e:
            logging.error(f"Map display error: {str(e)}")
            return f"MOSDAC provides satellite data coverage for **{geo_entity['name']}** at coordinates {geo_entity['coords']}."
    
    # Handle other queries with retrieval
    try:
        retriever = st.session_state.retriever
        results = retriever.retrieve(query, k=3)
        
        if results:
            context = "\n".join([f"- {res['text']}" for res in results])
            ai_response = get_ai_response(query, context)
            
            # Format sources with titles
            sources = []
            for res in results:
                title = res['url'].split('/')[-1].replace('-', ' ').title()
                sources.append(f"- [{title}]({res['url']})")
            
            return f"{ai_response}\n\n**Sources:**\n" + "\n".join(sources)
        return "I couldn't find information on that. Try asking about satellites, data access, or geospatial queries."
    except Exception as e:
        logging.error(f"Response error: {str(e)}")
        return "Sorry, I encountered an error. Please try again."

# --------------------------
# Streamlit UI with Enhanced Features
# --------------------------
def main():
    st.set_page_config(
        page_title="MOSDAC Help-Bot",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://mosdac.gov.in',
            'Report a bug': 'https://mosdac.gov.in/contact',
            'About': "## MOSDAC AI Help-Bot v2.1\nGeospatial RAG System for Satellite Data Assistance"
        }
    )
    
    # Custom CSS
    st.markdown("""
        <style>
            .stChatFloatingInputContainer { bottom: 20px; }
            .stChatMessage { padding: 12px; border-radius: 12px; }
            .assistant-message { background-color: #f0f9ff; }
            .user-message { background-color: #f5f5f5; }
            .st-emotion-cache-1y4p8pa { padding-top: 1.5rem; }
            .stButton>button { background-color: #4CAF50; color: white; }
            .stDownloadButton>button { background-color: #008CBA; }
            .stMarkdown h1 { color: #1a73e8; }
        </style>
    """, unsafe_allow_html=True)
    
    # Load custom logo with better error handling
    try:
        # Check if file exists
        if os.path.exists(LOGO_PATH):
            logo = Image.open(LOGO_PATH)
            logging.info(f"Logo found at: {os.path.abspath(LOGO_PATH)}")
        else:
            # Try alternative paths
            desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", LOGO_PATH)
            if os.path.exists(desktop_path):
                logo = Image.open(desktop_path)
                logging.info(f"Logo found on desktop: {desktop_path}")
            else:
                logo = None
                logging.warning(f"Logo not found at: {os.path.abspath(LOGO_PATH)} or {desktop_path}")
    except Exception as e:
        logo = None
        logging.error(f"Error loading logo: {str(e)}")
    
    # Initialize system
    if 'retriever' not in st.session_state:
        try:
            if process_and_cache_data():
                st.session_state.retriever = MOSDACRetriever()
                st.session_state.messages = [{
                    "role": "assistant",
                    "content": "🌐 **MOSDAC Geospatial Help-Bot Ready!** Ask about:\n- Satellite operations\n- Data access\n- Geographic queries\n- Product details"
                }]
            else:
                st.error("Failed to initialize system")
                st.stop()
        except Exception as e:
            st.error(f"Initialization failed: {str(e)}")
            st.stop()
    
    # Sidebar with knowledge graph visualization
    with st.sidebar:
        if logo:
            st.image(logo, use_column_width=True, caption="AI-based Help Bot for Satellite Data")
        else:
            # Fallback to online logo
            st.image("https://mosdac.gov.in/sites/default/files/logo_0.png", width=200)
        
        st.header("MOSDAC Knowledge Explorer")
        
        if st.button("🔄 Refresh Knowledge Base", help="Update the knowledge graph with latest MOSDAC content", use_container_width=True):
            try:
                if process_and_cache_data():
                    st.session_state.retriever = MOSDACRetriever()
                    st.success("Knowledge base updated!")
                    st.rerun()
            except Exception as e:
                st.error(f"Refresh failed: {str(e)}")
        
        if hasattr(st.session_state.retriever, 'knowledge_graph') and st.session_state.retriever.knowledge_graph:
            st.subheader("Knowledge Graph")
            entities = list(st.session_state.retriever.knowledge_graph["entities"].keys())[:10]
            st.info(f"**Key Entities:** {', '.join(entities)}")
            
            if st.checkbox("Show Relationships"):
                rel_count = len(st.session_state.retriever.knowledge_graph["relationships"])
                st.write(f"**{rel_count} relationships** in knowledge graph")
                
                for rel in st.session_state.retriever.knowledge_graph["relationships"][:5]:
                    st.caption(f"🔗 {rel['source']} → {rel['target']}")
        else:
            st.warning("Knowledge graph not available")
        
        st.subheader("Geospatial Tools")
        if st.button("🌍 Show India Coverage Map", help="Display satellite coverage over India", use_container_width=True):
            st.subheader("MOSDAC Satellite Coverage: India")
            with st.spinner("Generating satellite coverage map..."):
                try:
                    # Store map in session state to prevent disappearing
                    if 'india_map' not in st.session_state:
                        st.session_state.india_map = generate_folium_map([20.5937, 78.9629], "India")
                    
                    if st.session_state.india_map:
                        # Use folium_static to prevent map from disappearing
                        folium_static(st.session_state.india_map, width=300, height=300)
                        st.success("Satellite coverage map generated successfully")
                    else:
                        st.error("Could not generate coverage map. Please try again later.")
                except Exception as e:
                    st.error(f"Map generation failed: {str(e)}")
        
        st.divider()
        st.caption("MOSDAC Help-Bot v2.1")
        st.caption("Indian Space Research Organization")
    
    # Main chat interface
    st.title("🛰️ MOSDAC Geospatial Help-Bot")
    st.caption("AI-powered assistant for satellite data and geospatial queries")
    
    # Display messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Handle input
    if prompt := st.chat_input("Ask about MOSDAC data or geospatial queries..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("🛰️ Analyzing your query with satellite data..."):
            try:
                response = generate_response(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.rerun()

if __name__ == "__main__":
    # Install required packages if needed
    required_packages = [
        "streamlit", "sqlite3", "numpy", "requests", "faiss-cpu", 
        "sentence-transformers", "rank-bm25", 
        "geopy", "folium", "streamlit-folium", "Pillow"
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            except:
                logging.warning(f"Failed to install {package}")
    
    main()