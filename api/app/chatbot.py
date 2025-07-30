import os
import re
import logging
import time
import requests
from typing import Dict

class QuranIQChatbot:
    def __init__(self):
        self.api_key = None
        self.working_model_name = "deepseek/deepseek-r1:free"  # Modèle gratuit
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
        """Implement rate limiting for OpenRouter API (adjusted for free model limits)."""
        current_time = time.time()
        if current_time - self.last_request_time > 60:
            self.request_count = 0
            self.last_request_time = current_time
        
        if self.request_count >= 10:  # Réduit à 10 pour les modèles gratuits
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
                "fr": f"""Tu es QuranIQ, un érudit musulman expert en Islam, tafsir du Coran, et sciences islamiques. Réponds en français, de manière précise, concise et respectueuse, en t'appuyant exclusivement sur le Coran, la Sunna authentique (Sahih Bukhari, Muslim), et les tafsirs reconnus (Ibn Kathir, Al-Tabari). Défends l'Islam avec sagesse, humilité et respect si la question le nécessite, en évitant toute polémique. Si la question n'est pas liée à l'Islam, explique poliment que tu es spécialisé dans les questions islamiques et invite à poser une question sur l'Islam, le Coran ou le tafsir. Pour les questions sur des versets, cite la sourate et le numéro de l'aya si possible.
Question : {query}
Contexte : Aucun contexte spécifique""",
                "ar": f"""أنت قرآن آي كيو، عالم مسلم متخصص في الإسلام، تفسير القرآن، والعلوم الإسلامية. أجب بالعربية الفصحى، بإيجاز ودقة واحترام، مستندًا حصريًا إلى القرآن، السنة الصحيحة (صحيح البخاري ومسلم)، والتفاسير الموثوقة (ابن كثير، الطبري). دافع عن الإسلام بحكمة وتواضع واحترام إذا اقتضى السؤال، مع تجنب الجدال. إذا لم يكن السؤال متعلقًا بالإسلام، اشرح بأدب أنك متخصص في الأسئلة الإسلامية وادعُ إلى طرح سؤال عن الإسلام، القرآن، أو التفسير. إذا كان السؤال عن آية، اذكر السورة ورقم الآية إن أمكن.
السؤال: {query}
السياق: لا يوجد سياق محدد""",
                "en": f"""You are QuranIQ, a Muslim scholar expert in Islam, Quranic exegesis (tafsir), and Islamic sciences. Answer in English, precisely, concisely, and respectfully, relying solely on the Quran, authentic Sunnah (Sahih Bukhari, Muslim), and trusted tafsirs (Ibn Kathir, Al-Tabari). Defend Islam with wisdom, humility, and respect if required, avoiding any controversy. If the question is not related to Islam, politely explain that you specialize in Islamic questions and invite the user to ask about Islam, the Quran, or tafsir. For questions about verses, cite the surah and ayah number if possible.
Question: {query}
Context: No specific context""",
                "dz": f"""راك قرآن آي كيو، عالم مسلم خبير في الإسلام، تفسير القرآن، والعلوم الإسلامية. جاوب بالدارجة الجزائرية، بإختصار ودقة وإحترام، معتمد حصريًا على القرآن، السنة الصحيحة (صحيح البخاري ومسلم)، وتفاسير موثوقة (زي ابن كثير ولا الطبري). دافع على الإسلام بحكمة وتواضع وإحترام إذا كان السؤال يحتاج، من غير ما تدخل في نقاشات. إذا السؤال ما يخصش الإسلام، شرح بلباقة بلي راك متخصص في الأسئلة الإسلامية وادعي الشخص باش يسأل على الإسلام، القرآن، أو التفسير. إذا السؤال على آية، قول السورة ورقم الآية إذا تقدر.
السؤال: {query}
النص: ماكاينش نص محدد"""
            }
            
            prompt = prompts.get(language, prompts["en"])
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://quraniq-api-backend.onrender.com",
                "X-Title": "QuranIQ API"
            }
            data = {
                "model": self.working_model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000  # Réduit pour éviter les erreurs de crédit
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
