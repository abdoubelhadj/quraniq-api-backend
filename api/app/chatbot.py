import os
import json
import re
import numpy as np
import faiss
import google.generativeai as genai
import logging
import requests
from openai import OpenAI # Importez le client OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class QuranIQChatbot:
    def __init__(self):
        self.index = None
        self.chunks = []
        self.metadata = []
        self.gemini_model = None
        self.working_model_name = None
        self.is_loaded = False
        self.openai_client = None # Client OpenAI
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
            logging.info("🔄 Chargement du chatbot (avec RAG via OpenAI Embeddings)...")

            # Get API Key from environment variable for Gemini
            gemini_api_key = os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
            if not gemini_api_key:
                raise ValueError("GOOGLE_GENERATIVE_AI_API_KEY environment variable not set.")
            genai.configure(api_key=gemini_api_key)

            # Initialiser le client OpenAI pour les embeddings
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set. Please configure it for OpenAI embeddings.")
            self.openai_client = OpenAI(api_key=openai_api_key)
            logging.info("OpenAI client initialized for embeddings.")

            # --- LOGIQUE POUR TÉLÉCHARGER DEPUIS VERCEL BLOB ---
            BLOB_INDEX_URL = os.getenv("BLOB_INDEX_URL")
            BLOB_METADATA_URL = os.getenv("BLOB_METADATA_URL")

            if not BLOB_INDEX_URL or not BLOB_METADATA_URL:
                raise ValueError("BLOB_INDEX_URL or BLOB_METADATA_URL environment variables not set. Please configure them.")

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

            self.gemini_model, self.working_model_name = self.find_working_gemini_model()
            if not self.gemini_model:
                raise Exception("Aucun modèle Gemini valide n'a pu être trouvé ou initialisé.")
            
            self.is_loaded = True
            logging.info("✅ Chatbot chargé avec succès (avec RAG via OpenAI Embeddings).")
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
        """Vérifie si la question est de nature religieuse en utilisant le modèle Gemini."""
        try:
            logging.info(f"Classifying question '{query[:50]}...' as religious or not using Gemini.")
            classification_prompt = f"""
            La question suivante est-elle de nature religieuse (Islam) ? Répondez uniquement par "OUI" ou "NON".
            Question: "{query}"
            """
            response = self.gemini_model.generate_content(classification_prompt)
            classification = response.text.strip().upper()
            
            if "OUI" in classification:
                logging.info(f"Question '{query[:50]}...' classified as RELIGIOUS.")
                return True
            else:
                logging.info(f"Question '{query[:50]}...' classified as NON-RELIGIOUS.")
                return False
        except Exception as e:
            logging.error(f"Erreur lors de la classification de la question par Gemini : {e}", exc_info=True)
            # En cas d'erreur, par défaut, nous la traitons comme non religieuse pour éviter des boucles ou des réponses non pertinentes.
            return False

    def generate_query_embedding(self, query):
        """Génère l'embedding d'une requête en utilisant OpenAI."""
        try:
            logging.info("Starting query embedding generation with OpenAI.")
            model_name = "text-embedding-ada-002" 
            response = self.openai_client.embeddings.create(input=[query], model=model_name)
            embedding = np.array(response.data[0].embedding).astype("float32").reshape(1, -1)
            logging.info("Query embedding generated successfully with OpenAI.")
            return embedding
        except Exception as e:
            logging.error(f"Error generating query embedding with OpenAI: {e}", exc_info=True)
            return None

    def search_similar_chunks(self, query, top_k=3):
        """Recherche les chunks les plus similaires dans l'index FAISS."""
        try:
            logging.info("Starting search for similar chunks.")
            emb = self.generate_query_embedding(query) # Appel à OpenAI ici
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

        # Note sur le seuil de distance :
        # Le seuil de 1.0 est très strict. Si vos embeddings ne sont pas parfaitement alignés,
        # même des chunks pertinents pourraient être ignorés.
        # Vous pourriez vouloir l'ajuster (par exemple, 0.5, 0.7, ou même 1.5-2.0 selon la qualité de vos embeddings)
        # pour permettre plus de flexibilité.
        # Pour l'instant, je le laisse à 1.0 mais gardez cela à l'esprit pour le débogage.
        if context_chunks and context_chunks[0]['distance'] < 1.0:
            context = "\n\n".join(f"Source: {c['source']}\nContenu: {c['chunk']}" for c in context_chunks[:2])
            sources = list(set(c['source'] for c in context_chunks[:2]))
            mode = "hybrid"
            logging.info("Context used for generation.")
        else:
            logging.info("No relevant context found or distance too high, generating general response.")

        # Prompts améliorés pour une persona islamique et une meilleure utilisation du RAG
        prompts = {
            "fr": f"""Assalamu alaykum wa rahmatullahi wa barakatuh.
Tu es QuranIQ, un assistant islamique expert et respectueux, spécialisé dans le Coran et les enseignements islamiques.
Réponds toujours avec une perspective islamique, en utilisant les informations fournies dans le contexte ci-dessous si elles sont pertinentes et suffisantes.
Si le contexte ne contient pas la réponse directe ou complète, utilise tes connaissances générales approfondies sur l'Islam pour répondre de manière claire et concise.
Commence toujours tes réponses par une salutation islamique appropriée ou une invocation comme 'Bismillah'.

Question : {query}
Contexte fourni (si pertinent) :
{context}
""",
            "ar": f"""السلام عليكم ورحمة الله وبركاته.
أنت قرآن آي كيو، مساعد إسلامي خبير ومحترم، متخصص في القرآن الكريم والتعاليم الإسلامية.
أجب دائمًا من منظور إسلامي، مستخدمًا المعلومات المقدمة في السياق أدناه إذا كانت ذات صلة وكافية.
إذا لم يحتوي السياق على الإجابة المباشرة أو الكاملة، فاستخدم معرفتك العامة العميقة بالإسلام للإجابة بوضوح وإيجاز.
ابدأ إجاباتك دائمًا بتحية إسلامية مناسبة أو دعاء مثل 'بسم الله'.

السؤال: {query}
السياق المقدم (إذا كان ذا صلة):
{context}
""",
            "en": f"""Assalamu alaykum wa rahmatullahi wa barakatuh.
You are QuranIQ, an expert and respectful Islamic assistant, specialized in the Quran and Islamic teachings.
Always respond from an Islamic perspective, using the information provided in the context below if it is relevant and sufficient.
If the context does not contain the direct or complete answer, use your deep general knowledge of Islam to answer clearly and concisely.
Always start your answers with an appropriate Islamic greeting or invocation like 'Bismillah'.

Question: {query}
Provided Context (if relevant):
{context}
""",
            "dz": f"""السلام عليكم ورحمة الله وبركاته.
راك قرآن آي كيو، مساعد إسلامي خبير ومحترم، متخصص في القرآن الكريم والتعاليم الإسلامية.
جاوب دايما من منظور إسلامي، واستعمل المعلومات لي عطيتك فالنص التحتاني إذا كانت مفيدة وكافية.
إذا النص مافيهش الإجابة المباشرة ولا الكاملة، استعمل معرفتك العامة العميقة بالإسلام باش تجاوب بوضوح واختصار.
دايما ابدا إجاباتك بتحية إسلامية مناسبة ولا دعاء كيما 'بسم الله'.

السؤال: {query}
النص المقدم (إذا كان ذا صلة):
{context}
""",
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

        # Utiliser la nouvelle méthode de détection basée sur Gemini
        if not self.is_religious_question(query):
            logging.info("Non-religious question detected by Gemini.")
            return {
                "response": "Je suis QuranIQ, spécialisé uniquement dans les questions islamiques. Posez-moi une question sur l'Islam.",
                "language": lang,
                "sources": [],
                "mode": "non-religious"
            }
        
        logging.info("Religious question detected by Gemini. Searching for similar chunks...")
        chunks = self.search_similar_chunks(query)
        logging.info(f"Found {len(chunks)} similar chunks.")
        
        logging.info("Generating response from Gemini model...")
        return self.generate_response(query, chunks, lang)
