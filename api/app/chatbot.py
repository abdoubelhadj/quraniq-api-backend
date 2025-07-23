import os
import re
import google.generativeai as genai
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class QuranIQChatbot:
    def __init__(self):
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
        """Charge les composants nécessaires au chatbot (uniquement Gemini)."""
        try:
            logging.info("🔄 Chargement du chatbot (version sans RAG)...")

            # Get API Key from environment variable for Gemini
            gemini_api_key = os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
            if not gemini_api_key:
                raise ValueError("GOOGLE_GENERATIVE_AI_API_KEY environment variable not set.")
            genai.configure(api_key=gemini_api_key)

            self.gemini_model, self.working_model_name = self.find_working_gemini_model()
            if not self.gemini_model:
                raise Exception("Aucun modèle Gemini valide n'a pu être trouvé ou initialisé.")
            
            self.is_loaded = True
            logging.info("✅ Chatbot chargé avec succès (version sans RAG).")
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

    def generate_response(self, query, language):
        """Génère une réponse en utilisant le modèle Gemini (sans contexte RAG)."""
        prompts = {
            "fr": f"Tu es un expert de l'islam. Réponds clairement et de manière concise.\nQuestion : {query}",
            "ar": f"أنت خبير في الإسلام. أجب بوضضوح وإيجاز.\nالسؤال: {query}",
            "en": f"You are an expert in Islam. Answer clearly and concisely.\nQuestion: {query}",
            "dz": f"راك خبير فالدين. جاوب ببساطة ووضوح.\nالسؤال: {query}",
        }

        prompt = prompts.get(language, prompts["fr"])
        logging.info(f"Sending prompt to Gemini model. Language: {language}, Mode: general (no RAG)")
        try:
            result = self.gemini_model.generate_content(prompt)
            logging.info("Response received from Gemini.")
            return {
                "response": result.text.strip(),
                "language": language,
                "sources": [], # Plus de sources RAG
                "mode": "general"
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
        """Fonction principale de chat, gère la détection de langue et la génération de réponse."""
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
        
        logging.info("Religious question detected. Generating response from Gemini model (no RAG)...")
        return self.generate_response(query, lang)
