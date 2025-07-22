import os
import json
import re
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
import google.generativeai as genai
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class QuranIQChatbot:
    def __init__(self):
        self.index = None
        self.chunks = []
        self.metadata = []
        self.tokenizer = None
        self.embedding_model = None
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
                # Test with a simple prompt to ensure it's callable
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

            # Define paths relative to the current file
            RAG_FOLDER = os.path.join(os.path.dirname(__file__), "rag")
            INDEX_PATH = os.path.join(RAG_FOLDER, "index.faiss")
            METADATA_PATH = os.path.join(RAG_FOLDER, "chunks_metadata.json")

            # Ensure RAG folder exists
            os.makedirs(RAG_FOLDER, exist_ok=True)

            # Load FAISS index
            if os.path.exists(INDEX_PATH):
                self.index = faiss.read_index(INDEX_PATH)
                logging.info(f"FAISS index loaded from {INDEX_PATH}")
            else:
                logging.warning(f"FAISS index not found at {INDEX_PATH}. Creating a dummy index for testing.")
                # Create a dummy index if not found (for initial setup/testing)
                # Dimension should match your embedding model output (xlm-roberta-base is 768)
                dummy_dimension = 768
                self.index = faiss.IndexFlatL2(dummy_dimension)
                # Add some dummy vectors to prevent errors if search is called
                self.index.add(np.random.rand(10, dummy_dimension).astype('float32'))
                faiss.write_index(self.index, INDEX_PATH) # Save dummy for future runs

            # Load chunks and metadata
            if os.path.exists(METADATA_PATH):
                with open(METADATA_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.chunks = data["chunks"]
                    self.metadata = data["metadata"]
                logging.info(f"Chunks metadata loaded from {METADATA_PATH}")
            else:
                logging.warning(f"Chunks metadata not found at {METADATA_PATH}. Creating dummy data.")
                # Create dummy data if not found
                self.chunks = ["Ceci est un chunk de test sur l'Islam.", "Le Prophète Muhammad est le dernier prophète de l'Islam."]
                self.metadata = [{"source": "Dummy Doc 1"}, {"source": "Dummy Doc 2"}]
                with open(METADATA_PATH, 'w', encoding='utf-8') as f:
                    json.dump({"chunks": self.chunks, "metadata": self.metadata}, f, ensure_ascii=False, indent=2)


            self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
            self.embedding_model = AutoModel.from_pretrained("xlm-roberta-base")
            logging.info("Embedding model and tokenizer loaded.")

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
        # Plus de mots pour une meilleure détection
        arabic_chars = re.compile(r'[\u0600-\u06FF]')
        if arabic_chars.search(text):
            # Vérifier les mots spécifiques au dialecte algérien
            algerian_words = ['واش', 'كيفاش', 'وين', 'علاش', 'بصح', 'برك', 'حنا', 'نتوما', 'هوما', 'راني', 'راك', 'راها', 'تاع', 'بزاف', 'شوية']
            if any(word in text for word in algerian_words):
                return "dz"
            return "ar"
        
        english_words = ['what', 'how', 'why', 'when', 'where', 'who', 'is', 'are', 'the', 'and', 'or', 'can', 'should', 'must', 'do', 'did']
        if any(word in text.lower().split() for word in english_words):
            return "en"
        
        # Par défaut, si aucune autre langue n'est détectée
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
        """Génère l'embedding d'une requête."""
        try:
            inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).numpy().astype("float32")
            return embedding
        except Exception as e:
            logging.error(f"Error generating query embedding: {e}", exc_info=True)
            return None

    def search_similar_chunks(self, query, top_k=3):
        """Recherche les chunks les plus similaires dans l'index FAISS."""
        try:
            emb = self.generate_query_embedding(query)
            if emb is None:
                return []
            distances, indices = self.index.search(emb, top_k)
            results = []
            for i, d in zip(indices[0], distances[0]):
                if 0 <= i < len(self.chunks):
                    results.append({
                        "chunk": self.chunks[i],
                        "source": self.metadata[i],
                        "distance": float(d)
                    })
            return results
        except Exception as e:
            logging.error(f"Error searching similar chunks: {e}", exc_info=True)
            return []

    def generate_response(self, query, context_chunks, language):
        """Génère une réponse en utilisant le modèle Gemini et le contexte RAG."""
        context = ""
        sources = []
        mode = "general"

        # Only use context if the top chunk is sufficiently similar (distance threshold)
        if context_chunks and context_chunks[0]['distance'] < 1.0: # Adjust threshold as needed
            context = "\n\n".join(f"Source: {c['source']}\nContenu: {c['chunk']}" for c in context_chunks[:2])
            sources = list(set(c['source'] for c in context_chunks[:2]))
            mode = "hybrid"
        else:
            logging.info("No relevant context found or distance too high, generating general response.")

        prompts = {
            "fr": f"Tu es un expert de l'islam. Réponds clairement et de manière concise. Utilise les informations fournies si pertinentes.\nQuestion : {query}\n{context}",
            "ar": f"أنت خبير في الإسلام. أجب بوضوح وإيجاز. استخدم المعلومات المقدمة إذا كانت ذات صلة.\nالسؤال: {query}\n{context}",
            "en": f"You are an expert in Islam. Answer clearly and concisely. Use provided information if relevant.\nQuestion: {query}\n{context}",
            "dz": f"راك خبير فالدين. جاوب ببساطة ووضوح. استعمل المعلومات لي عطيتك إذا كانت مفيدة.\nالسؤال: {query}\n{context}",
        }

        prompt = prompts.get(language, prompts["fr"])
        try:
            result = self.gemini_model.generate_content(prompt)
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
        lang = self.detect_language(query)
        if not self.is_religious_question(query):
            return {
                "response": "Je suis QuranIQ, spécialisé uniquement dans les questions islamiques. Posez-moi une question sur l'Islam.",
                "language": lang,
                "sources": [],
                "mode": "non-religious"
            }
        
        chunks = self.search_similar_chunks(query)
        return self.generate_response(query, chunks, lang)
