import os
import json
import re
import numpy as np
import faiss
# import torch # <-- Supprimer cette importation
# from transformers import AutoTokenizer, AutoModel # <-- Supprimer cette importation
import google.generativeai as genai
import logging
import requests
from fal_ai import FalAI # <-- Nouvelle importation pour Fal.ai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class QuranIQChatbot:
    def __init__(self):
        self.index = None
        self.chunks = []
        self.metadata = []
        # self.tokenizer = None # <-- Supprimer
        # self.embedding_model = None # <-- Supprimer
        self.fal_client = None # <-- Nouveau client Fal.ai
        self.gemini_model = None
        self.working_model_name = None
        self.is_loaded = False
        self.load_components()

    def find_working_gemini_model(self):
        """Trouve un modèle Gemini fonctionnel en testant les modèles disponibles."""
        models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro", "models/gemini-1.5-flash", "models/gemini-1.5-pro", "models/gemini-pro"]
        for name in models:
            try:
                model = genai.GenerativeModel(name)
                model.generate_content("ping")
                logging.info(f"Found working Gemini model: {name}")
                return model, name
            except Exception as e:
                logging.warning(f"Model {name} failed to initialize or respond: {e}")
                continue
        return None, None

    def load_components(self):
        """Charge tous les composants nécessaires au chatbot."""
        try:
            logging.info("🔄 Chargement du chatbot...")

            # Get API Key from environment variable
            gemini_api_key = os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
            if not gemini_api_key:
                raise ValueError("GOOGLE_GENERATIVE_AI_API_KEY environment variable not set.")
            genai.configure(api_key=gemini_api_key)

            # Initialize Fal.ai client
            fal_key = os.getenv("FAL_KEY") # <-- Nouvelle variable d'environnement pour Fal.ai
            if not fal_key:
                raise ValueError("FAL_KEY environment variable not set. Required for embedding generation.")
            self.fal_client = FalAI(key=fal_key)
            logging.info("Fal.ai client initialized.")

            # --- LOGIQUE POUR TÉLÉCHARGER DEPUIS VERCEL BLOB (inchangée) ---
            BLOB_INDEX_URL = os.getenv("BLOB_INDEX_URL")
            BLOB_METADATA_URL = os.getenv("BLOB_METADATA_URL")

            if not BLOB_INDEX_URL or not BLOB_METADATA_URL:
                raise ValueError("BLOB_INDEX_URL or BLOB_METADATA_URL environment variables not set. Please configure them on Vercel.")

            # Téléchargement de l'index FAISS
            logging.info(f"Downloading FAISS index from {BLOB_INDEX_URL}")
            index_response = requests.get(BLOB_INDEX_URL)
            index_response.raise_for_status()
            with open("/tmp/index.faiss", "wb") as f:
                f.write(index_response.content)
            self.index = faiss.read_index("/tmp/index.faiss")
            logging.info("FAISS index downloaded and loaded.")

            # Téléchargement des métadonnées
            logging.info(f"Downloading chunks metadata from {BLOB_METADATA_URL}")
            metadata_response = requests.get(BLOB_METADATA_URL)
            metadata_response.raise_for_status()
            data = metadata_response.json()
            self.chunks = data["chunks"]
            self.metadata = data["metadata"]
            logging.info("Chunks metadata downloaded and loaded.")
            # --- FIN DE LA LOGIQUE BLOB ---

            # self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base") # <-- Supprimer
            # self.embedding_model = AutoModel.from_pretrained("xlm-roberta-base") # <-- Supprimer
            # logging.info("Embedding model and tokenizer loaded.") # <-- Supprimer

            self.gemini_model, self.working_model_name = self.find_working_gemini_model()
            if not self.gemini_model:
                raise Exception("Aucun modèle Gemini valide n'a pu être trouvé ou initialisé.")
            
            self.is_loaded = True
            logging.info("✅ Chatbot chargé avec succès.")
        except Exception as e:
            logging.error(f"❌ Erreur lors du chargement du chatbot : {e}", exc_info=True)
            self.is_loaded = False

    def detect_language(self, text):
        """Détecte la langue du texte (fr, ar, en, dz)."""
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

    def is_religious_question(self, query):
        """Vérifie si la question est de nature religieuse."""
        keywords = [
            'islam', 'allah', 'prophète', 'coran', 'hadith', 'prière', 'ramadan', 'hajj', 'zakat', 'shahada', 'mosquée', 'imam', 'sourate', 'ayat', 'dua', 'halal', 'haram', 'sunna', 'fiqh', 'tafsir', 'religion', 'dieu', 'muhammad', 'salat',
            'الله', 'إسلام', 'قرآن', 'حديث', 'صلاة', 'رمضان', 'حج', 'زكاة', 'شهادة', 'مسجد', 'إمام', 'سورة', 'آية', 'دعاء', 'حلال', 'حرام', 'سنة', 'فقه', 'تفسير', 'ربي', 'محمد', 'نبي', 'رسول',
            'واش نصلي', 'كيفاش نصلي', 'وين نصلي', 'علاش نصوم', 'كيفاش نتوضا', 'واش حلال', 'واش حرام', 'كيفاش نقرا القرآن', 'وين القبلة', 'كيفاش نحج', 'صلاتي', 'صومي', 'حجي', 'قرايتي', 'وضوئي', 'دعايا', 'تسبيحي', 'بسم الله', 'الحمد لله', 'إن شاء الله', 'ما شاء الله', 'استغفر الله', 'لا حول ولا قوة إلا بالله', 'طهارة', 'نجاسة', 'وضوء', 'غسل', 'تيمم', 'قبلة', 'مكة', 'مدينة'
        ]
        return any(k in query.lower() for k in keywords)

    def generate_query_embedding(self, query):
        """Génère l'embedding d'une requête en utilisant Fal.ai."""
        try:
            logging.info(f"Generating embedding for query: '{query[:50]}...' using Fal.ai.")
            # Modèle d'embedding sur Fal.ai. Vous pouvez choisir un autre modèle si nécessaire.
            # "sg161222/e5-large-v2-embedding" est un bon modèle d'embedding multilingue.
            # Vérifiez la documentation de Fal.ai pour les modèles disponibles.
            result = self.fal_client.run(
                "fal-ai/e5-large-v2-embedding", # Modèle d'embedding Fal.ai
                arguments={"text": query}
            )
            # Le résultat de Fal.ai est un dictionnaire, l'embedding est dans 'embedding'
            embedding = np.array(result["embedding"], dtype=np.float32).reshape(1, -1)
            logging.info("Query embedding generated successfully via Fal.ai.")
            return embedding
        except Exception as e:
            logging.error(f"Error generating query embedding with Fal.ai: {e}", exc_info=True)
            return None

    def search_similar_chunks(self, query, top_k=3):
        """Recherche les chunks les plus similaires dans l'index FAISS."""
        try:
            logging.info("Starting search for similar chunks.")
            emb = self.generate_query_embedding(query)
            if emb is None:
                logging.warning("Embedding generation failed, returning empty chunks.")
                return []
            
            logging.info("Performing FAISS search.")
            distances, indices = self.index.search(emb, top_k)
            
            results = []
            for i, d in zip(indices[0], distances[0]):
                if 0 <= i < len(self.chunks):
                    results.append({
                        "chunk": self.chunks[i],
                        "source": self.metadata[i],
                        "distance": float(d)
                    })
            logging.info(f"FAISS search completed. Found {len(results)} relevant chunks.")
            return results
        except Exception as e:
            logging.error(f"Error searching similar chunks: {e}", exc_info=True)
            return []

    def generate_response(self, query, context_chunks, language):
        """Génère une réponse en utilisant le modèle Gemini et le contexte RAG."""
        context = ""
        sources = []
        mode = "general"

        if context_chunks and context_chunks[0]['distance'] < 1.0: # Adjust threshold as needed
            context = "\n\n".join(f"Source: {c['source']}\nContenu: {c['chunk']}" for c in context_chunks[:2])
            sources = list(set(c['source'] for c in context_chunks[:2]))
            mode = "hybrid"
            logging.info("Context used for generation.")
        else:
            logging.info("No relevant context found or distance too high, generating general response.")

        prompts = {
            "fr": f"Tu es un expert de l'islam. Réponds clairement et de manière concise. Utilise les informations fournies si pertinentes.\nQuestion : {query}\n{context}",
            "ar": f"أنت خبير في الإسلام. أجب بوضضوح وإيجاز. استخدم المعلومات المقدمة إذا كانت ذات صلة.\nالسؤال: {query}\n{context}",
            "en": f"You are an expert in Islam. Answer clearly and concisely. Use provided information if relevant.\nQuestion: {query}\n{context}",
            "dz": f"راك خبير فالدين. جاوب ببساطة ووضوح. استعمل المعلومات لي عطيتك إذا كانت مفيدة.\nالسؤال: {query}\n{context}",
        }

        prompt = prompts.get(language, prompts["fr"])
        logging.info(f"Sending prompt to Gemini model. Language: {language}, Mode: {mode}")
        try:
            result = self.gemini_model.generate_content(prompt)
            logging.info("Response received from Gemini.")
            return {
                "response": result.text.strip(),
                "language": language,
                "sources": sources,
                "mode": mode
            }
        except Exception as e:
            logging.error(f"Error generating content from Gemini: {e}", exc_info=True)
            return {
                "response": "Désolé, une erreur est survenue lors de la génération de la réponse.",
                "language": language,
                "sources": [],
                "mode": "error"
            }

    def chat(self, query):
        """Fonction principale de chat, gère la détection de langue, la pertinence et la génération de réponse."""
        logging.info(f"Chat request received: {query[:50]}...")
        lang = self.detect_language(query)
        logging.info(f"Detected language: {lang}")

        if not self.is_religious_question(query):
            logging.info("Non-religious question detected.")
            return {
                "response": "Je suis QuranIQ, spécialisé uniquement dans les questions islamiques. Posez-moi une question sur l'Islam.",
                "language": lang,
                "sources": [],
                "mode": "non-religious"
            }
        
        logging.info("Religious question detected. Searching for similar chunks...")
        chunks = self.search_similar_chunks(query)
        logging.info(f"Found {len(chunks)} similar chunks.")
        
        logging.info("Generating response from Gemini model...")
        return self.generate_response(query, chunks, lang)
