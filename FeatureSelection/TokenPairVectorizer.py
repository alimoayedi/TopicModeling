import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import CustomFuncLib as cus

class TokenPairVectorizer:
    def __init__(self, tuple_size=2):
        self.tuple_size = tuple_size
        self.unique_tuples_lst = None
        # We override the tokenizer and preprocessor because we pass pre-formed strings
        self.vectorizer = CountVectorizer(tokenizer=lambda x: x, preprocessor=None, lowercase=False)
        
    def _generate_tuple_strings(self, lst):
        """Converts the list of integer tokens into a list of joined string tuples."""
        tuples = cus.make_tuple(lst, self.tuple_size)
        # Format as "int_int" so the CountVectorizer can process them natively
        return [f"{t[0]}_{t[1]}" for t in tuples]

    def fit(self, vectorized_df):
        """
        Fits the CountVectorizer to construct the sparse matrix of unique token pairs.
        """
        print("Generating Tuples using Sparse Vectorization...")
        # 1. Convert lists of tokens into lists of tuple strings
        doc_tuple_strings = vectorized_df.apply(self._generate_tuple_strings)
        
        # 2. Fast C-optimized frequency counting
        sparse_matrix = self.vectorizer.fit_transform(doc_tuple_strings)
        
        # 3. Store the vocabulary
        self.unique_tuples_lst = list(self.vectorizer.get_feature_names_out())
        print(f"Tuple_2 unique tokens found: {len(self.unique_tuples_lst)}")
        
        # 4. Return as the DataFrame expected by the downstream pipeline
        return pd.DataFrame(sparse_matrix.toarray(), index=vectorized_df.index, columns=self.unique_tuples_lst)
    
    def transform(self, vectorized_df):
        """
        Transforms validation/test data using the fitted tuple vocabulary.
        """
        doc_tuple_strings = vectorized_df.apply(self._generate_tuple_strings)
        sparse_matrix = self.vectorizer.transform(doc_tuple_strings)
        
        return pd.DataFrame(sparse_matrix.toarray(), index=vectorized_df.index, columns=self.unique_tuples_lst)