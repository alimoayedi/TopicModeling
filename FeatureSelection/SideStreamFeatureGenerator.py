from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class SideStreamFeatureGenerator:
    def __init__(self, max_features=5000, use_tfidf=True):
        """
        Initializes the side stream generator.
        max_features: Caps the vocabulary size to control memory.
        use_tfidf: If True, uses TF-IDF. If False, uses raw Term Frequency (TF).
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            use_idf=use_tfidf, 
            norm='l2' if use_tfidf else None # L2 norm is standard for TF-IDF
        )
        self.is_fitted = False

    def fit_transform(self, raw_documents):
        """
        Fits the vocabulary on the training data and returns the feature matrix.
        Pass ONLY your training set's untrimmed documents here.
        """
        features = self.vectorizer.fit_transform(raw_documents)
        self.is_fitted = True
        return features.toarray()

    def transform(self, raw_documents):
        """
        Transforms validation/test data using the already fitted vocabulary.
        """
        if not self.is_fitted:
            raise ValueError("The vectorizer must be fitted on training data first.")
        
        features = self.vectorizer.transform(raw_documents)
        return features.toarray()