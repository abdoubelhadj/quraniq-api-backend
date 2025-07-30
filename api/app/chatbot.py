import os
import re
import logging
import time
import requests
from typing import Dict

class QuranIQChatbot:
    def __init__(self):
        self.api_key = None
        self.working_model_name = "mistralai/mistral-7b-instruct"  # Modèle par défaut
        self.is_loaded = False
        self.request_count = 0
        self.last_request_time = 0
        self.load_components()

    def load_components(self):
        """Load necessary components for the chatbot."""
        try:
            logging.info("🔄 Loading QuranIQ chatbot components...")
            
            self.api_key = os.getenv("OPENROUTER_API_KEY")
            if not self.api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable not set")
            
            # Test API connectivity
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://quraniq-api-backend.onrender.com",
                "X-Title": "QuranIQ API"
            }
            data = {
                "model": self.working_model_name,
                "messages": [{"role": "user", "content": "ping"}]
            }
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=10
            )
            response.raise_for_status()
            if response.json().get("choices"):
                logging.info(f"✅ OpenRouter API configured with model {self.working_model_name}")
                self.is_loaded = True
            else:
                logging.warning("⚠️ OpenRouter API response invalid, proceeding in degraded mode")
            
        except Exception as e:
            logging.error(f"❌ Error loading chatbot: {e}", exc_info=True)
            self.is_loaded = False

    def _rate_limit_openrouter(self):
        """Implement rate limiting for OpenRouter API (adjust based on OpenRouter limits)."""
        current_time = time.time()
        if current_time - self.last_request_time > 60:
            self.request_count = 0
            self.last_request_time = current_time
        
        if self.request_count >= 14:  # Ajuster selon les limites d'OpenRouter
            wait_time = 60 - (current_time - self.last_request_time)
            if wait_time > 0:
                logging.warning(f"⚠️ Rate limit approaching, waiting {wait_time:.1f} seconds")
                time.sleep(wait_time)
                self.request_count = 0
                self.last_request_time = time.time()
        
        self.request_count += 1
        logging.info(f"OpenRouter API request count: {self.request_count}")

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
        """Check if the question is religious using OpenRouter API."""
        if not self.is_loaded:
            logging.warning("OpenRouter API not initialized, defaulting to non-religious classification")
            return False
        
        try:
            self._rate_limit_openrouter()
            
            language = self.detect_language(query)
            prompts = {
                "fr": f"Cette question est-elle liée à l'Islam ? Répondez uniquement par 'oui' ou 'non'. Question : {query}",
                "ar": f"هل هذا السؤال متعلق بالإسلام؟ أجب فقط بـ 'نعم' أو 'لا'. السؤال: {query}",
                "en": f"Is this question related to Islam? Answer only 'yes' or 'no'. Question: {query}",
                "dz": f"هل السؤال هذا يخص الإسلام؟ جاوب غير بـ 'نعم' أو 'لا'. السؤال: {query}"
            }
            
            prompt = prompts.get(language, prompts["en"])
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://quraniq-api-backend.onrender.com",
                "X-Title": "QuranIQ API"
            }
            data = {
                "model": self.working_model_name,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            for attempt in range(3):
                try:
                    response = requests.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers=headers,
                        json=data,
                        timeout=10
                    )
                    response.raise_for_status()
                    response_json = response.json()
                    if "choices" in response_json and response_json["choices"]:
                        answer = response_json["choices"][0]["message"]["content"].strip().lower()
                        logging.info(f"Question classified as {'RELIGIOUS' if answer in ['yes', 'oui', 'نعم'] else 'NON-RELIGIOUS'} (OpenRouter response: {answer})")
                        return answer in ['yes', 'oui', 'نعم']
                    else:
                        raise Exception("Invalid response from OpenRouter API")
                except requests.exceptions.RequestException as e:
                    if response and response.status_code == 429:
                        wait_time = 8
                        logging.warning(f"Rate limit hit in is_religious_question, retrying in {wait_time} seconds (attempt {attempt + 1})")
                        time.sleep(wait_time)
                    else:
                        raise
            
            logging.error("Max retries reached for OpenRouter API in is_religious_question")
            return False  # Fallback to non-religious if API fails
            
        except Exception as e:
            logging.error(f"Error classifying question: {e}")
            return False  # Fallback to non-religious if API fails

    def generate_response(self, query: str, language: str) -> Dict:
        """Generate a response using OpenRouter API with retry logic."""
        if not self.is_loaded:
            logging.error("OpenRouter API not initialized")
            fallback_responses = {
                "fr": "Désolé, le service est temporairement indisponible. Veuillez réessayer plus tard.",
                "ar": "عذراً، الخدمة غير متوفرة مؤقتاً. يرجى المحاولة لاحقاً.",
                "en": "Sorry, the service is temporarily unavailable. Please try again later.",
                "dz": "سامحني، الخدمة مش متوفرة مؤقتاً. عاود جرب بعد."
            }
            return {
                "response": fallback_responses.get(language, fallback_responses["fr"]),
                "language": language,
                "sources": [],
                "mode": "error"
            }
        
        try:
            self._rate_limit_openrouter()
            
            prompts = {
                "fr": f"""Tu es QuranIQ, assistant islamique. Réponds brièvement et clairement en français.
Question : {query}
Contexte : Aucun contexte spécifique""",
                "ar": f"""أنت قرآن آي كيو، مساعد إسلامي. أجب بإيجاز ووضوح بالعربية.
السؤال: {query}
السياق: لا يوجد سياق محدد""",
                "en": f"""You are QuranIQ, Islamic assistant. Answer briefly and clearly in English.
Question: {query}
Context: No specific context""",
                "dz": f"""راك قرآن آي كيو، مساعد إسلامي. جاوب بإختصار ووضوح بالدارجة الجزائرية.
السؤال: {query}
النص: ماكاينش نص محدد"""
            }
            
            prompt = prompts.get(language, prompts["fr"])
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://quraniq-api-backend.onrender.com",
                "X-Title": "QuranIQ API"
            }
            data = {
                "model": self.working_model_name,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            for attempt in range(3):
                try:
                    response = requests.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers=headers,
                        json=data,
                        timeout=10
                    )
                    response.raise_for_status()
                    response_json = response.json()
                    if "choices" in response_json and response_json["choices"]:
                        return {
                            "response": response_json["choices"][0]["message"]["content"].strip(),
                            "language": language,
                            "sources": [],
                            "mode": "general"
                        }
                    else:
                        raise Exception("Invalid response from OpenRouter API")
                except requests.exceptions.RequestException as e:
                    if response and response.status_code == 429:
                        wait_time = 8
                        logging.warning(f"Rate limit hit, retrying in {wait_time} seconds (attempt {attempt + 1})")
                        time.sleep(wait_time)
                    else:
                        raise
            
            logging.error("Max retries reached for OpenRouter API")
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
