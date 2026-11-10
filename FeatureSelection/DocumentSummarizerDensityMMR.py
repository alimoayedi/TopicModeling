import os
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import warnings

# Suppress sklearn and memory leak warnings to keep the output clean during multiprocessing
warnings.filterwarnings("ignore", category=UserWarning)

# Maximal Marginal Relevance (MMR)
# MMR iteratively selects sentences that are highly relevant to the overall document vector, 
# but strictly penalizes sentences that are too similar to those already selected.
class DocumentSummarizerDensityMMR:
    def __init__(self, max_sentences=5, top_m=15, decay_factor=0.1, max_char_buffer=10000):
        """
        Initializes the Keyphrase Density & MMR-Lite summarization module.
        
        Args:
            max_sentences (int): Number of sentences to extract.
            top_m (int): The number of global keyphrases to identify.
            decay_factor (float): The penalty multiplier for redundant keyphrases (0.0 to 1.0).
            max_char_buffer (int): Pre-slices long raw texts to prevent memory leaks.
        """
        self.max_sentences = max_sentences
        self.top_m = top_m
        self.decay_factor = decay_factor
        self.max_char_buffer = max_char_buffer
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

    def generate_summary(self, text):
        # GUARDRAIL 1: Intercept NaNs, Nones, and pure whitespace
        if text is None or str(text).strip() == "" or str(text).lower() == 'nan':
            return "unparsable document content removed during summarization"

        text = str(text).strip()
        text_chunk = text[:self.max_char_buffer]

        sentences = sent_tokenize(text_chunk)
        
        # GUARDRAIL 2: NLTK failed to find sentences
        if len(sentences) == 0:
            words = text_chunk.split()
            fallback_summary = " ".join(words[:self.max_sentences * 20])
            return fallback_summary if fallback_summary.strip() else "unparsable document content removed during summarization"
        
        # GUARDRAIL 3: Document is too short
        if len(sentences) <= self.max_sentences:
            return " ".join(sentences)

        # --- PHASE 1: Global Keyphrase Identification ---
        # Treating sentences as the corpus calculates proper term frequencies
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=self.top_m, stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
        except ValueError:
            # Fallback if text is purely mathematical symbols or stop words
            return " ".join(sentences[:self.max_sentences])
        
        # Calculate the global weight of each keyphrase (sum across all sentences)
        global_weights = np.array(tfidf_matrix.sum(axis=0)).flatten()
        
        # Create a boolean mask indicating which keyphrases exist in which sentences (Shape: N_sentences x M_keyphrases)
        presence_matrix = (tfidf_matrix.toarray() > 0).astype(float)

        # Calculate the length penalty denominator for all sentences at once
        # $$Score(s) = \frac{\sum_{k \in s} W_k}{\log_{10}(|s| + \epsilon)}$$
        word_counts = np.array([len(s.split()) for s in sentences])
        penalties = np.log10(word_counts + 1.0)
        penalties[penalties == 0] = 1.0 # Prevent division by zero

        # --- PHASE 2 & 3: Density Scoring & Dynamic Redundancy Penalty (MMR-Lite) ---
        selected_indices = []
        unselected_indices = set(range(len(sentences)))
        current_weights = global_weights.copy()

        for _ in range(self.max_sentences):
            best_idx = -1
            best_score = -1.0
            
            # Evaluate remaining sentences
            for i in unselected_indices:
                # Dot product seamlessly calculates the sum of the weights of the present keyphrases
                score = np.dot(presence_matrix[i], current_weights) / penalties[i]
                if score > best_score:
                    best_score = score
                    best_idx = i
            
            if best_idx == -1 or best_score <= 0.0:
                break # Stop if no meaningful sentences remain
                
            selected_indices.append(best_idx)
            unselected_indices.remove(best_idx)
            
            # Apply MMR-Lite Decay Penalty to the specific keyphrases found in the selected sentence
            present_mask = presence_matrix[best_idx] > 0
            current_weights[present_mask] *= self.decay_factor

        # --- PHASE 4: Chronological Reconstruction ---
        extracted_sentences = [sentences[i] for i in sorted(selected_indices)]
        final_summary = " ".join(extracted_sentences).strip()

        if not final_summary:
            return " ".join(sentences[:self.max_sentences])

        return final_summary