import spacy
import numpy as np

class NameEntityRecognitionFeature:
    """
    This class is responsible for generating Named Entity Recognition (NER) features from text data.
    It scans the text to find the density of specific Named Entities and returns a entity-count Dimensional 
    vector for each document.
    """
    def __init__(self, language_model='en_core_web_sm'):
        self.language_model = language_model

    def generate_ner_density(self, df, text_column, track_labels=None, top_k=6,):
        """
        Scans raw text to find the density of Named Entities.
        If track_labels is None, it dynamically profiles the corpus to find the top_k most frequent labels.
        Returns:
            ner_vectors: The list of numerical vectors.
            track_labels: The list of labels used (so they can be reused on test data).
        """
        # Load the selected language model dynamically
        nlp = spacy.load(self.language_model, disable=["parser", "lemmatizer"])
        
        # --- DYNAMIC PROFILING STAGE ---
        if track_labels is None:
            print(f"Dynamically profiling corpus to find the top {top_k} NER labels...")
            global_counts = {}
            for doc in nlp.pipe(df[text_column].astype(str), batch_size=100):
                for ent in doc.ents:
                    global_counts[ent.label_] = global_counts.get(ent.label_, 0) + 1
            
            # Sort labels by highest frequency and keep the top_k
            track_labels = [label for label, count in sorted(global_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]]
            print(f"Auto-Selected Labels: {track_labels}")
        else:
            print(f"Using provided labels: {track_labels} across {len(df)} documents...")

        # --- EXTRACTION STAGE ---
        ner_vectors = []
        for doc in nlp.pipe(df[text_column].astype(str), batch_size=100):
            counts = {label: 0 for label in track_labels}
            
            for ent in doc.ents:
                if ent.label_ in counts:
                    counts[ent.label_] += 1
                    
            doc_len = len(doc) if len(doc) > 0 else 1
            density_vector = [counts[label] / doc_len for label in track_labels]
            ner_vectors.append(np.array(density_vector, dtype=np.float32))
            
        return ner_vectors, track_labels