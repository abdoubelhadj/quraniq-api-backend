import os
import json
import re
import numpy as np
import faiss
import google.generativeai as genai
import logging
import requests
import cohere
import gc
import time
from typing import List, Dict, Optional

class QuranIQChatbot:
    def __init__(self):
        self.index = None
        self.chunks = []
        self.metadata = []
        self.gemini_model = None
        self.working_model_name = None
        self.is_loaded = False
        self.cohere_client = None
        self.request_count = 0
        self.last_request_time = 0
        self.load_components()

    def find_working_gemini_model(self):
        """Trouve un modèle Gemini fonctionnel."""
        models = [
            "gemini-1.5-flash", 
            "gemini-1.5-pro", 
            "gemini-pro"
        ]
        
        for name in models:
            try:
                model = genai.GenerativeModel(name)
                # Test with a simple prompt
                test_response = model.generate_content("ping")
                if test_response and test_response.text:
                    logging.info(f"✅ Found working Gemini model: {name}")
                    return model, name
            except Exception as e:
                logging.warning(f"⚠️ Model {name} failed: {str(e)[:100]}...")
                continue
        
        logging.error("❌ No working Gemini model found")
        return None, None

    def load_components(self):
        """Charge tous les composants nécessaires au chatbot."""
        try:
            logging.info("🔄 Loading QuranIQ chatbot components...")
            
            # Configure Gemini API
            gemini_api_key = os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
            if not gemini_api_key:
                raise ValueError("GOOGLE_GENERATIVE_AI_API_KEY environment variable not set")
            
            genai.configure(api_key=gemini_api_key)
            logging.info("✅ Gemini API configured")

            # Initialize Cohere client
            cohere_api_key = os.getenv("COHERE_API_KEY")
            if not cohere_api_key:
                raise ValueError("COHERE_API_KEY environment variable not set")
            
            self.cohere_client = cohere.Client(cohere_api_key)
            logging.info("✅ Cohere client initialized")

            # Load FAISS index and metadata from Vercel Blob
            self._load_from_blob()
            
            # Find working Gemini model
            self.gemini_model, self.working_model_name = self.find_working_gemini_model()
            if not self.gemini_model:
                raise Exception("No working Gemini model found")

            self.is_loaded = True
            logging.info("✅ QuranIQ chatbot loaded successfully")
            
            # Force garbage collection to free memory
            gc.collect()
            
        except Exception as e:
            logging.error(f"❌ Error loading chatbot: {e}", exc_info=True)
            self.is_loaded = False
            raise

    def _load_from_blob(self):
        """Load FAISS index and metadata from Vercel Blob"""
        blob_index_url = os.getenv("BLOB_INDEX_URL")
        blob_metadata_url = os.getenv("BLOB_METADATA_URL")
        
        if not blob_index_url or not blob_metadata_url:
            raise ValueError("BLOB_INDEX_URL or BLOB_METADATA_URL not set")

        try:
            # Download FAISS index with retry logic
            logging.info("📥 Downloading FAISS index...")
            for attempt in range(3):
                try:
                    index_response = requests.get(blob_index_url, timeout=120)
                    index_response.raise_for_status()
                    break
                except requests.RequestException as e:
                    if attempt == 2:
                        raise
                    logging.warning(f"Attempt {attempt + 1} failed, retrying: {e}")
                    time.sleep(2)
            
            with open("/tmp/index.faiss", "wb") as f:
                f.write(index_response.content)
            
            self.index = faiss.read_index("/tmp/index.faiss")
            logging.info(f"✅ FAISS index loaded with {self.index.ntotal} vectors")

            # Download metadata with retry logic
            logging.info("📥 Downloading metadata...")
            for attempt in range(3):
                try:
                    metadata_response = requests.get(blob_metadata_url, timeout=120)
                    metadata_response.raise_for_status()
                    break
                except requests.RequestException as e:
                    if attempt == 2:
                        raise
                    logging.warning(f"Attempt {attempt + 1} failed, retrying: {e}")
                    time.sleep(2)
            
            data = metadata_response.json()
            self.chunks = data["chunks"]
            self.metadata = data["metadata"]
            logging.info(f"✅ Metadata loaded: {len(self.chunks)} chunks")
            
            # Verify data consistency
            if len(self.chunks) != len(self.metadata):
                logging.warning(f"⚠️ Mismatch: {len(self.chunks)} chunks vs {len(self.metadata)} metadata entries")
            
            if self.index.ntotal != len(self.chunks):
                logging.warning(f"⚠️ Mismatch: {self.index.ntotal} FAISS vectors vs {len(self.chunks)} chunks")
            
        except requests.RequestException as e:
            logging.error(f"❌ Network error loading from blob: {e}")
            raise
        except Exception as e:
            logging.error(f"❌ Error loading from blob: {e}")
            raise

    def _rate_limit_gemini(self):
        """Implement rate limiting for Gemini API (15 requests per minute for free tier)"""
        current_time = time.time()
        
        # Reset counter every minute
        if current_time - self.last_request_time > 60:
            self.request_count = 0
            self.last_request_time = current_time
        
        # Check if we're approaching the limit
        if self.request_count >= 14:  # Leave 1 request as buffer
            wait_time = 60 - (current_time - self.last_request_time)
            if wait_time > 0:
                logging.warning(f"⚠️ Rate limit approaching, waiting {wait_time:.1f} seconds")
                time.sleep(wait_time)
                self.request_count = 0
                self.last_request_time = time.time()
        
        self.request_count += 1

    def detect_language(self, text: str) -> str:
        """Détecte la langue du texte."""
        try:
            arabic_chars = re.compile(r'[\u0600-\u06FF]')
            if arabic_chars.search(text):
                algerian_words = ['واش', 'كيفاش', 'وين', 'علاش', 'بصح', 'برك', 'حنا', 'نتوما', 'هوما', 'راني', 'راك', 'راها', 'تاع', 'بزاف', 'شوية']
                if any(word in text for word in algerian_words):
                    return "dz"
                return "ar"
            
            english_words = ['what', 'how', 'why', 'when', 'where', 'who', 'is', 'are', 'the', 'and', 'or', 'can', 'should', 'must', 'do', 'did']
            if any(word in text.lower().split() for word in english_words):
                return "en"
            
            return "fr"
        except Exception as e:
            logging.warning(f"Language detection error: {e}")
            return "fr"

    def is_religious_question(self, query: str) -> bool:
        """Vérifie si la question est de nature religieuse avec rate limiting."""
        try:
            # Simple keyword-based check first to avoid API calls when possible
            religious_keywords = {
                'ar': ['الله', 'النبي', 'القرآن', 'الإسلام', 'الصلاة', 'الحج', 'الزكاة', 'الصوم', 'محمد', 'عيسى', 'موسى', 'إبراهيم', 'داوود'],
                'fr': ['allah', 'prophète', 'coran', 'islam', 'prière', 'hajj', 'zakat', 'jeûne', 'mohammed', 'jésus', 'moïse', 'abraham', 'david'],
                'en': ['allah', 'prophet', 'quran', 'islam', 'prayer', 'hajj', 'zakat', 'fasting', 'muhammad', 'jesus', 'moses', 'abraham', 'david'],
                'dz': ['ربي', 'الرسول', 'القرآن', 'الدين', 'الصلاة']
            }
            
            query_lower = query.lower()
            for lang_keywords in religious_keywords.values():
                if any(keyword in query_lower for keyword in lang_keywords):
                    logging.info("Question classified as RELIGIOUS (keyword match)")
                    return True
            
            # If no keywords found, use Gemini API with rate limiting
            self._rate_limit_gemini()
            
            classification_prompt = f"""
            La question suivante est-elle de nature religieuse (Islam) ? 
            Répondez uniquement par "OUI" ou "NON".
            
            Question: "{query}"
            """
            
            response = self.gemini_model.generate_content(classification_prompt)
            classification = response.text.strip().upper()
            
            is_religious = "OUI" in classification
            logging.info(f"Question classified as {'RELIGIOUS' if is_religious else 'NON-RELIGIOUS'} (Gemini)")
            return is_religious
            
        except Exception as e:
            logging.error(f"Error in religious classification: {e}")
            # Default to True for religious keywords, False otherwise
            return any(keyword in query.lower() for keywords in religious_keywords.values() for keyword in keywords)

    def generate_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """Génère l'embedding d'une requête avec Cohere."""
        try:
            response = self.cohere_client.embed(
                texts=[query],
                model="embed-multilingual-v3.0",
                input_type="search_query"
            )
            
            embedding = np.array(response.embeddings[0]).astype("float32").reshape(1, -1)
            logging.info(f"Query embedding generated: {embedding.shape}")
            return embedding
            
        except Exception as e:
            logging.error(f"Error generating embedding: {e}")
            return None

    def search_similar_chunks(self, query: str, top_k: int = 3) -> List[Dict]:
        """Recherche les chunks similaires avec gestion d'erreurs améliorée."""
        try:
            embedding = self.generate_query_embedding(query)
            if embedding is None:
                logging.warning("No embedding generated, returning empty results")
                return []

            # Verify embedding dimensions match FAISS index
            if hasattr(self.index, 'd') and embedding.shape[1] != self.index.d:
                logging.error(f"Embedding dimension mismatch: {embedding.shape[1]} vs {self.index.d}")
                return []

            # Ensure top_k doesn't exceed available vectors
            actual_k = min(top_k, self.index.ntotal, len(self.chunks))
            
            distances, indices = self.index.search(embedding, actual_k)
            
            results = []
            for i, d in zip(indices[0], distances[0]):
                if 0 <= i < len(self.chunks) and 0 <= i < len(self.metadata):
                    results.append({
                        "chunk": self.chunks[i],
                        "source": self.metadata[i],
                        "distance": float(d)
                    })
                else:
                    logging.warning(f"Invalid index {i}, skipping")
            
            logging.info(f"Found {len(results)} similar chunks")
            return results
            
        except Exception as e:
            logging.error(f"Error searching chunks: {e}", exc_info=True)
            return []

    def generate_response(self, query: str, context_chunks: List[Dict], language: str) -> Dict:
        """Génère une réponse avec Gemini et rate limiting."""
        try:
            context = ""
            sources = []
            mode = "general"
            
            # Use context if relevant
            distance_threshold = 0.7
            if context_chunks and context_chunks[0]['distance'] < distance_threshold:
                context = "\n\n".join(
                    f"Source: {c['source']}\nContenu: {c['chunk']}" 
                    for c in context_chunks[:2]
                )
                sources = list(set(c['source'] for c in context_chunks[:2]))
                mode = "hybrid"

            # Apply rate limiting before making Gemini request
            self._rate_limit_gemini()

            # Language-specific prompts (shortened to reduce token usage)
            prompts = {
                "fr": f"""Tu es QuranIQ, assistant islamique. Réponds brièvement et clairement.

Question : {query}
Contexte : {context[:1000] if context else "Aucun contexte spécifique"}""",

                "ar": f"""أنت قرآن آي كيو، مساعد إسلامي. أجب بإيجاز ووضوح.

السؤال: {query}
السياق: {context[:1000] if context else "لا يوجد سياق محدد"}""",

                "en": f"""You are QuranIQ, Islamic assistant. Answer briefly and clearly.

Question: {query}
Context: {context[:1000] if context else "No specific context"}""",

                "dz": f"""راك قرآن آي كيو، مساعد إسلامي. جاوب بإختصار ووضوح.

السؤال: {query}
النص: {context[:1000] if context else "ماكاينش نص محدد"}"""
            }

            prompt = prompts.get(language, prompts["fr"])
            
            result = self.gemini_model.generate_content(prompt)
            
            return {
                "response": result.text.strip(),
                "language": language,
                "sources": sources,
                "mode": mode
            }
            
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            
            # Fallback response based on language
            fallback_responses = {
                "fr": "Désolé, je rencontre des difficultés techniques. Veuillez réessayer dans quelques instants.",
                "ar": "عذراً، أواجه صعوبات تقنية. يرجى المحاولة مرة أخرى بعد قليل.",
                "en": "Sorry, I'm experiencing technical difficulties. Please try again in a few moments.",
                "dz": "سامحني، راني نواجه مشاكل تقنية. عاود جرب بعد شوية."
            }
            
            return {
                "response": fallback_responses.get(language, fallback_responses["fr"]),
                "language": language,
                "sources": [],
                "mode": "error"
            }

    def chat(self, query: str) -> Dict:
        """Fonction principale de chat avec gestion d'erreurs améliorée."""
        try:
            logging.info(f"Processing chat request: {query[:50]}...")
            
            # Detect language
            language = self.detect_language(query)
            logging.info(f"Detected language: {language}")

            # Check if religious question
            if not self.is_religious_question(query):
                non_religious_responses = {
                    "fr": "Je suis QuranIQ, spécialisé uniquement dans les questions islamiques. Posez-moi une question sur l'Islam.",
                    "ar": "أنا قرآن آي كيو، متخصص فقط في الأسئلة الإسلامية. اسألني سؤالاً عن الإسلام.",
                    "en": "I am QuranIQ, specialized only in Islamic questions. Ask me a question about Islam.",
                    "dz": "أنا قرآن آي كيو، متخصص غير في الأسئلة الإسلامية. اسألني على الإسلام."
                }
                
                return {
                    "response": non_religious_responses.get(language, non_religious_responses["fr"]),
                    "language": language,
                    "sources": [],
                    "mode": "non-religious"
                }

            # Search for relevant chunks
            chunks = self.search_similar_chunks(query)
            
            # Generate response
            return self.generate_response(query, chunks, language)
            
        except Exception as e:
            logging.error(f"Error in chat method: {e}", exc_info=True)
            return {
                "response": "Une erreur est survenue. Veuillez réessayer.",
                "language": "fr",
                "sources": [],
                "mode": "error"
            }
