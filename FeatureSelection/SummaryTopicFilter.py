import os
# Prevents tokenizers from deadlocking when using pandarallel multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

import numpy as np
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import warnings

# Suppress warnings to keep the output clean during multiprocessing
warnings.filterwarnings("ignore", category=UserWarning)

class SummaryTopicFilter:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initializes the Bi-Encoder Topic Filter.
        Uses a lightweight sentence transformer to map text into a semantic space.
        """
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
            
        # Initialize on CPU to safely bypass the CUDA multiprocessing fork crash
        self.encoder = SentenceTransformer(model_name, device='cpu')
        
    def filter_summary(self, row, summary_col='pMMRsummary', topic_col='target_topic', std_margin=0.0):
        summary_text = row[summary_col]
        topic_text = row[topic_col]
        
        # GUARDRAIL 1: Intercept invalid rows
        if not isinstance(summary_text, str) or not str(summary_text).strip():
            return ""
        if not isinstance(topic_text, str) or not str(topic_text).strip():
            return summary_text # Cannot filter without a topic anchor
            
        sentences = sent_tokenize(summary_text)
        
        # GUARDRAIL 2: If the summary is only 1 sentence, thresholding is mathematically moot
        if len(sentences) <= 1:
            return summary_text
            
        # 1. Embed the Anchor (Topic) and the Candidates (Sentences)
        topic_embedding = self.encoder.encode([topic_text], show_progress_bar=False, device='cpu')
        sentence_embeddings = self.encoder.encode(sentences, show_progress_bar=False, device='cpu')
        
        # 2. Calculate Cosine Similarity
        # This returns an array of scores representing how closely each sentence matches the topic
        similarities = cosine_similarity(topic_embedding, sentence_embeddings)[0]
        
        # 3. Dynamic Mean Thresholding
        # Calculate the dynamic mean ($\mu$) for this specific document's summary
        mean_sim = np.mean(similarities)
        sd_sim = np.std(similarities)
        
        # 4. Filter and Reconstruct
        # Keep sentences where score > $\mu$, preserving original chronological order
        filtered_sentences = [sentences[i] for i in range(len(sentences)) if similarities[i] > mean_sim + std_margin * sd_sim]
        
        # GUARDRAIL 3: Failsafe in case of zero variance 
        if not filtered_sentences:
            return summary_text
            
        return " ".join(filtered_sentences)