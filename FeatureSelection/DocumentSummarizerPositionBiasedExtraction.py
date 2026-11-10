import networkx as nx
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import warnings

# Suppress sklearn warnings to keep the output clean during multiprocessing
warnings.filterwarnings("ignore", category=UserWarning)

class DocumentSummarizerPositionBiasedExtraction:
    def __init__(self, max_sentences=1, position_penalty=1.5, max_input_sentences=75):
        self.max_sentences = max_sentences
        self.position_penalty = position_penalty
        self.max_input_sentences = max_input_sentences
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

    def generate_summary(self, text):
        # GUARDRAIL 1: Intercept NaNs, Nones, and pure whitespace
        if text is None or str(text).strip() == "" or str(text).lower() == 'nan':
            # 6-word placeholder to ensure it passes your min_doc_length=6 preprocessor rule
            return "unparsable document content removed during summarization"

        # Coerce to string to prevent any unexpected float types
        text = str(text).strip()

        # Pre-slice the raw string to save memory and CPU
        char_limit = self.max_input_sentences * 200
        text_chunk = text[:char_limit]

        sentences = sent_tokenize(text_chunk)
        
        # GUARDRAIL 2: NLTK failed to find sentences (e.g., no punctuation in text)
        if len(sentences) == 0:
            words = text_chunk.split()
            # Extract roughly 20 words as a proxy for 1 sentence
            fallback_summary = " ".join(words[:self.max_sentences * 20])
            return fallback_summary if fallback_summary.strip() else "unparsable document content removed during summarization"

        sentences = sentences[:self.max_input_sentences]
        
        if len(sentences) <= self.max_sentences:
            return " ".join(sentences)

        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            sentence_vectors = vectorizer.fit_transform(sentences)
        except ValueError:
            # GUARDRAIL 3: TF-IDF failed (document is pure math, symbols, or unparsable gibberish)
            return " ".join(sentences[:self.max_sentences])

        similarity_matrix = cosine_similarity(sentence_vectors)
        nx_graph = nx.from_numpy_array(similarity_matrix)

        try:
            base_scores = nx.pagerank(nx_graph)
        except nx.PowerIterationFailedConvergence:
            base_scores = {i: 1.0 / len(sentences) for i in range(len(sentences))}

        biased_scores = {}
        for i in range(len(sentences)):
            decay_multiplier = 1.0 / (1.0 + (self.position_penalty * i))
            biased_scores[i] = base_scores.get(i, 0) * decay_multiplier

        ranked_sentences = sorted(((biased_scores[i], s) for i, s in enumerate(sentences)), reverse=True)
        extracted_sentences = [ranked_sentences[i][1] for i in range(self.max_sentences)]
        
        original_order_summary = [s for s in sentences if s in extracted_sentences]
        final_summary = " ".join(original_order_summary).strip()

        # GUARDRAIL 4: If the final output is completely empty, force a slice
        if not final_summary:
            return " ".join(sentences[:self.max_sentences])

        return final_summary