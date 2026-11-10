import os
# Prevents tokenizers from deadlocking when using pandarallel multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

from nltk.tokenize import sent_tokenize
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sentence_transformers import SentenceTransformer
import nltk
import warnings

# Suppress sklearn and memory leak warnings to keep the output clean during multiprocessing
warnings.filterwarnings("ignore", category=UserWarning)

class DocumentSummarizerClustering:
    def __init__(self, k=1, max_char_buffer=10000):
        """
        Initializes the Semantic Centroid summarization module.
        
        Args:
            k (int): The number of semantic clusters to form (equates to the number of extracted sentences).
            max_char_buffer (int): Pre-slices long raw texts before running sentence tokenization to prevent memory leaks.
        """
        self.k = k
        self.max_char_buffer = max_char_buffer
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

        # Initialize the lightweight embedding model (Requires ~80MB of RAM)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

    def generate_summary(self, text):
        # GUARDRAIL 1: Intercept NaNs, Nones, and pure whitespace
        if text is None or str(text).strip() == "" or str(text).lower() == 'nan':
            return "unparsable document content removed during summarization"

        # Coerce to string and pre-slice to save memory and CPU
        text = str(text).strip()
        text_chunk = text[:self.max_char_buffer]

        sentences = sent_tokenize(text_chunk)
        
        # GUARDRAIL 2: NLTK failed to find sentences (e.g., no punctuation in text)
        if len(sentences) == 0:
            words = text_chunk.split()
            fallback_summary = " ".join(words[:self.k * 20])
            return fallback_summary if fallback_summary.strip() else "unparsable document content removed during summarization"
        
        # GUARDRAIL 3: Document is too short to cluster
        if len(sentences) <= self.k:
            return " ".join(sentences)

        # 1. Generate dense semantic vectors for all sentences
        embeddings = self.encoder.encode(sentences, show_progress_bar=False)

        # 2. Perform K-Means clustering to identify topical centroids
        # n_init='auto' suppresses sklearn convergence warnings
        kmeans = KMeans(n_clusters=self.k, n_init='auto', random_state=42)
        try:
            kmeans.fit(embeddings)
        except ValueError:
            # Fallback if clustering fails due to numerical instability in the embeddings
            return " ".join(sentences[:self.k])

        # 3. Find the single closest sentence embedding to each cluster centroid
        closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings)

        # 4. Extract and sort chronologically to maintain document flow
        extracted_sentences = [sentences[i] for i in sorted(closest_indices)]
        final_summary = " ".join(extracted_sentences).strip()

        # GUARDRAIL 4: If the final output is completely empty, force a slice
        if not final_summary:
            return " ".join(sentences[:self.k])

        return final_summary