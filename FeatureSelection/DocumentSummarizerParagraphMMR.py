import os
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import warnings
import re

# Suppress sklearn and memory leak warnings to keep the output clean during multiprocessing
warnings.filterwarnings("ignore", category=UserWarning)

class DocumentSummarizerParagraphMMR:
    def __init__(self, sentences_per_para=1, top_m=20, decay_factor=0.5, max_char_buffer=50000):
        """
        Initializes the Paragraph-Level MMR summarizer.
        
        Args:
            sentences_per_para (int): Number of sentences to extract per paragraph.
            top_m (int): The number of global keyphrases to identify across the whole document.
            decay_factor (float): The penalty multiplier for used keyphrases (0.0 to 1.0).
            max_char_buffer (int): Expanded buffer for LexGLUE SCOTUS to capture deep paragraphs.
        """
        self.sentences_per_para = sentences_per_para
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

        text = str(text).strip()[:self.max_char_buffer]
        
        # 1. Isolate Document Paragraphs (SCOTUS uses newline breaks)
        # Drop paragraphs shorter than 10 characters (often page numbers or stray formatting)
        raw_paragraphs = [p.strip() for p in re.split(r'\n+', text) if len(p.strip()) > 10]
        
        if not raw_paragraphs:
            return "unparsable document content removed during summarization"

        # 2. Tokenize into sentences while tracking paragraph boundaries
        doc_sentences = []
        para_boundaries = [] # Stores tuples of (start_idx, end_idx) for each paragraph
        
        current_idx = 0
        for para in raw_paragraphs:
            para_sents = sent_tokenize(para)
            if not para_sents:
                continue
            doc_sentences.extend(para_sents)
            para_boundaries.append((current_idx, current_idx + len(para_sents)))
            current_idx += len(para_sents)
            
        if len(doc_sentences) == 0:
            return "unparsable document content removed during summarization"
            
        # GUARDRAIL 2: If the document is already hyper-dense (1 sentence per para average)
        if len(doc_sentences) <= len(para_boundaries): 
            return " ".join(doc_sentences)

        # 3. Global Keyphrase Extraction (Thematic Fingerprint)
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=self.top_m, stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(doc_sentences)
        except ValueError:
            # Fallback: take the first sentence of every paragraph if TF-IDF fails
            return " ".join([doc_sentences[b[0]] for b in para_boundaries])
            
        # Calculate global weights and presence masks
        global_weights = np.array(tfidf_matrix.sum(axis=0)).flatten()
        presence_matrix = (tfidf_matrix.toarray() > 0).astype(float)
        
        # Calculate the logarithmic length penalty denominator for all sentences at once
        word_counts = np.array([len(s.split()) for s in doc_sentences])
        penalties = np.log10(word_counts + 1.0)
        penalties[penalties == 0] = 1.0 
        
        # 4. Paragraph-Level Density Scoring & MMR Extraction
        selected_indices = []
        current_weights = global_weights.copy()
        
        for start_idx, end_idx in para_boundaries:
            # If the paragraph only has 1 sentence, keep it and move on
            if end_idx - start_idx <= self.sentences_per_para:
                selected_indices.extend(range(start_idx, end_idx))
                continue
                
            para_selected = []
            unselected = set(range(start_idx, end_idx))
            
            for _ in range(self.sentences_per_para):
                best_idx = -1
                best_score = -1.0
                
                # Evaluate sentences strictly within the current paragraph boundary
                for i in unselected:
                    score = np.dot(presence_matrix[i], current_weights) / penalties[i]
                    if score > best_score:
                        best_score = score
                        best_idx = i
                
                # Fallback: If no keyphrases exist in this paragraph, pick the opening sentence
                if best_idx == -1 or best_score <= 0.0:
                    if not para_selected: 
                        best_idx = start_idx
                    else:
                        break
                        
                para_selected.append(best_idx)
                if best_idx in unselected:
                    unselected.remove(best_idx)
                
                # MMR Decay: Penalize the keyphrases found so the next paragraph 
                # (or next sentence) is mathematically pushed to explore a new theme
                present_mask = presence_matrix[best_idx] > 0
                current_weights[present_mask] *= self.decay_factor
                
            selected_indices.extend(para_selected)
            
        # 5. Chronological Reconstruction
        extracted_sentences = [doc_sentences[i] for i in sorted(selected_indices)]
        final_summary = " ".join(extracted_sentences).strip()
        
        if not final_summary:
            return " ".join([doc_sentences[b[0]] for b in para_boundaries])
            
        return final_summary