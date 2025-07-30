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
            
            with open("/tmp/index.faiss", "wb") as f:
                f.write(index_response.content)
            
            self.index = faiss.read_index("/tmp/index.faiss")
            logging.info("✅ FAISS index loaded")

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
            
            data = metadata_response.json()
            self.chunks = data["chunks"]
            self.metadata = data["metadata"]
            logging.info(f"✅ Metadata loaded: {len(self.chunks)} chunks")
            
        except requests.RequestException as e:
            logging.error(f"❌ Network error loading from blob: {e}")
            raise
        except Exception as e:
            logging.error(f"❌ Error loading from blob: {e}")
            raise

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
        """Vérifie si la question est de nature religieuse."""
        try:
            classification_prompt = f"""
            La question suivante est-elle de nature religieuse (Islam) ? 
            Répondez uniquement par "OUI" ou "NON".
            
            Question: "{query}"
            """
            
            response = self.gemini_model.generate_content(classification_prompt)
            classification = response.text.strip().upper()
            
            is_religious = "OUI" in classification
            logging.info(f"Question classified as {'RELIGIOUS' if is_religious else 'NON-RELIGIOUS'}")
            return is_religious
            
        except Exception as e:
            logging.error(f"Error in religious classification: {e}")
            return True  # Default to True to be safe

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
        """Recherche les chunks similaires."""
        try:
            embedding = self.generate_query_embedding(query)
            if embedding is None:
                return []

            distances, indices = self.index.search(embedding, top_k)
            
            results = []
            for i, d in zip(indices[0], distances[0]):
                if 0 <= i < len(self.chunks):
                    results.append({
                        "chunk": self.chunks[i],
                        "source": self.metadata[i],
                        "distance": float(d)
                    })
            
            logging.info(f"Found {len(results)} similar chunks")
            return results
            
        except Exception as e:
            logging.error(f"Error searching chunks: {e}")
            return []

    def generate_response(self, query: str, context_chunks: List[Dict], language: str) -> Dict:
        """Génère une réponse avec Gemini."""
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

            # Language-specific prompts
            prompts = {
                "fr": f"""Assalamu alaykum wa rahmatullahi wa barakatuh. Tu es QuranIQ, un assistant islamique expert et respectueux, spécialisé dans le Coran et les enseignements islamiques. 

Réponds toujours avec une perspective islamique, en utilisant les informations fournies dans le contexte ci-dessous si elles sont pertinentes et suffisantes. Si le contexte ne contient pas la réponse directe ou complète, utilise tes connaissances générales approfondies sur l'Islam pour répondre de manière claire et concise. 

Commence toujours tes réponses par une salutation islamique appropriée ou une invocation comme 'Bismillah'.

Question : {query}

Contexte fourni (si pertinent) : {context}""",

                "ar": f"""السلام عليكم ورحمة الله وبركاته. أنت قرآن آي كيو، مساعد إسلامي خبير ومحترم، متخصص في القرآن الكريم والتعاليم الإسلامية. 

أجب دائمًا من منظور إسلامي، مستخدمًا المعلومات المقدمة في السياق أدناه إذا كانت ذات صلة وكافية. إذا لم يحتوي السياق على الإجابة المباشرة أو الكاملة، فاستخدم معرفتك العامة العميقة بالإسلام للإجابة بوضوح وإيجاز. 

ابدأ إجاباتك دائمًا بتحية إسلامية مناسبة أو دعاء مثل 'بسم الله'.

السؤال: {query}

السياق المقدم (إذا كان ذا صلة): {context}""",

                "en": f"""Assalamu alaykum wa rahmatullahi wa barakatuh. You are QuranIQ, an expert and respectful Islamic assistant, specialized in the Quran and Islamic teachings. 

Always respond from an Islamic perspective, using the information provided in the context below if it is relevant and sufficient. If the context does not contain the direct or complete answer, use your deep general knowledge of Islam to answer clearly and concisely. 

Always start your answers with an appropriate Islamic greeting or invocation like 'Bismillah'.

Question: {query}

Provided Context (if relevant): {context}""",

                "dz": f"""السلام عليكم ورحمة الله وبركاته. راك قرآن آي كيو، مساعد إسلامي خبير ومحترم، متخصص في القرآن الكريم والتعاليم الإسلامية. 

جاوب دايما من منظور إسلامي، واستعمل المعلومات لي عطيتك فالنص التحتاني إذا كانت مفيدة وكافية. إذا النص مافيهش الإجابة المباشرة ولا الكاملة، استعمل معرفتك العامة العميقة بالإسلام باش تجاوب بوضوح واختصار. 

دايما ابدا إجاباتك بتحية إسلامية مناسبة ولا دعاء كيما 'بسم الله'.

السؤال: {query}

النص المقدم (إذا كان ذا صلة): {context}"""
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
            return {
                "response": "Désolé, une erreur est survenue lors de la génération de la réponse.",
                "language": language,
                "sources": [],
                "mode": "error"
            }

    def chat(self, query: str) -> Dict:
        """Fonction principale de chat."""
        try:
            logging.info(f"Processing chat request: {query[:50]}...")
            
            # Detect language
            language = self.detect_language(query)
            logging.info(f"Detected language: {language}")

            # Check if religious question
            if not self.is_religious_question(query):
                return {
                    "response": "Je suis QuranIQ, spécialisé uniquement dans les questions islamiques. Posez-moi une question sur l'Islam.",
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
