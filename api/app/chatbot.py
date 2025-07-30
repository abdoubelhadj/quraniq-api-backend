import os
import re
import logging
import time
import google.generativeai as genai
from typing import Dict

class QuranIQChatbot:
    def __init__(self):
        self.gemini_model = None
        self.working_model_name = None
        self.is_loaded = False
        self.request_count = 0
        self.last_request_time = 0
        self.load_components()

    def find_working_gemini_model(self):
        """Find a working Gemini model."""
        models = [
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-pro"
        ]
        
        for name in models:
            try:
                model = genai.GenerativeModel(name)
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
        """Load necessary components for the chatbot."""
        try:
            logging.info("🔄 Loading QuranIQ chatbot components...")
            
            gemini_api_key = os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
            if not gemini_api_key:
                raise ValueError("GOOGLE_GENERATIVE_AI_API_KEY environment variable not set")
            
            genai.configure(api_key=gemini_api_key)
            logging.info("✅ Gemini API configured")
            
            self.gemini_model, self.working_model_name = self.find_working_gemini_model()
            if not self.gemini_model:
                raise Exception("No working Gemini model found")
            
            self.is_loaded = True
            logging.info("✅ QuranIQ chatbot loaded successfully")
            
        except Exception as e:
            logging.error(f"❌ Error loading chatbot: {e}", exc_info=True)
            self.is_loaded = False
            raise

    def _rate_limit_gemini(self):
        """Implement rate limiting for Gemini API (15 requests per minute for free tier)."""
        current_time = time.time()
        if current_time - self.last_request_time > 60:
            self.request_count = 0
            self.last_request_time = current_time
        
        if self.request_count >= 14:
            wait_time = 60 - (current_time - self.last_request_time)
            if wait_time > 0:
                logging.warning(f"⚠️ Rate limit approaching, waiting {wait_time:.1f} seconds")
                time.sleep(wait_time)
                self.request_count = 0
                self.last_request_time = time.time()
        
        self.request_count += 1
        logging.info(f"Gemini API request count: {self.request_count}/50 (daily limit)")

    def detect_language(self, text: str) -> str:
        """Detect the language of the text."""
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
        """Check if the question is religious using keywords."""
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
        
        logging.info("Question classified as NON-RELIGIOUS (keyword match)")
        return False

    def generate_response(self, query: str, language: str) -> Dict:
        """Generate a response using Gemini with retry logic."""
        try:
            self._rate_limit_gemini()
            
            prompts = {
                "fr": f"""Tu es QuranIQ, assistant islamique. Réponds brièvement et clairement.
Question : {query}
Contexte : Aucun contexte spécifique""",
                "ar": f"""أنت قرآن آي كيو، مساعد إسلامي. أجب بإيجاز ووضوح.
السؤال: {query}
السياق: لا يوجد سياق محدد""",
                "en": f"""You are QuranIQ, Islamic assistant. Answer briefly and clearly.
Question: {query}
Context: No specific context""",
                "dz": f"""راك قرآن آي كيو، مساعد إسلامي. جاوب بإختصار ووضوح.
السؤال: {query}
النص: ماكاينش نص محدد"""
            }
            
            prompt = prompts.get(language, prompts["fr"])
            
            for attempt in range(3):
                try:
                    result = self.gemini_model.generate_content(prompt)
                    return {
                        "response": result.text.strip(),
                        "language": language,
                        "sources": [],
                        "mode": "general"
                    }
                except Exception as e:
                    if "429" in str(e):
                        wait_time = 8
                        logging.warning(f"Rate limit hit, retrying in {wait_time} seconds (attempt {attempt + 1})")
                        time.sleep(wait_time)
                    else:
                        raise
            
            logging.error("Max retries reached for Gemini API")
            raise Exception("Rate limit exceeded after retries")
            
        except Exception as e:
            logging.error(f"Error generating response: {e}")
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
        """Main chat function with error handling."""
        try:
            start_time = time.time()
            logging.info(f"Processing chat request: {query[:50]}...")
            
            language = self.detect_language(query)
            logging.info(f"Detected language: {language} (took {time.time() - start_time:.2f}s)")
            
            if not self.is_religious_question(query):
                non_religious_responses = {
                    "fr": "Je suis QuranIQ, spécialisé uniquement dans les questions islamiques. Posez-moi une question sur l'Islam.",
                    "ar": "أنا قرآن آي كيو، متخصص فقط في الأسئلة الإسلامية. اسألني سؤالاً عن الإسلام.",
                    "en": "I am QuranIQ, specialized only in Islamic questions. Ask me a question about Islam.",
                    "dz": "أنا قرآن آي كيو، متخصص غير في الأسئلة الإسلامية. اسألني على الإسلام."
                }
                logging.info(f"Total request processing time: {time.time() - start_time:.2f}s")
                return {
                    "response": non_religious_responses.get(language, non_religious_responses["fr"]),
                    "language": language,
                    "sources": [],
                    "mode": "non-religious"
                }
            
            response = self.generate_response(query, language)
            logging.info(f"Total request processing time: {time.time() - start_time:.2f}s")
            return response
            
        except Exception as e:
            logging.error(f"Error in chat method: {e}", exc_info=True)
            return {
                "response": "Une erreur est survenue. Veuillez réessayer.",
                "language": "fr",
                "sources": [],
                "mode": "error"
            }
