import CustomFuncLib as cus

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TextVectorizationModel:
    def __init__(self, apply_cosine_similarity_reduction=False):
        self.apply_cosine_similarity_reduction = apply_cosine_similarity_reduction
        self.tfidf_vectorizer = TfidfVectorizer(tokenizer=cus.white_space_splitter, 
                                                preprocessor=None, 
                                                stop_words=None, 
                                                max_df=1.0, 
                                                min_df=1, 
                                                max_features=None)
        self.count_vectorizer = CountVectorizer(tokenizer=cus.white_space_splitter, 
                                                preprocessor=None, 
                                                stop_words=None, 
                                                max_df=1.0, 
                                                min_df=1, 
                                                max_features=None)
        self.vocab_size = 0
        self.cosine_sim_score = None
        self.tf_matrix = None
        self.tfidf_matrix = None
    
    def fit(self, df_to_train):
        df_joined_tokens = df_to_train.apply(lambda tokenized_txt: " ".join(tokenized_txt))

        # Fit TF-IDF and Count vectorizer on training data
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(df_joined_tokens)
        self.tf_matrix = self.count_vectorizer.fit_transform(df_joined_tokens)
        self.vocab_size = len(self.count_vectorizer.vocabulary_)
        
        df_vectorized = self.transform(df_to_train)
        self.cosine_sim_score = cosine_similarity(self.tf_matrix.T)

        print("Number of tokens: " + str(self.vocab_size))

        if self.apply_cosine_similarity_reduction:
            df_vectorized = df_vectorized.apply(
                lambda lst: cus.remove_high_correlated_tokens(self.cosine_sim_score, lst)
                )
            print("Number of tokens after cosine similarity: " + str(cus.get_number_of_tokens(df_vectorized)))
        
        return df_vectorized
    
    def transform(self, df_to_vectorize):
        # Transform validation or test datasets using the trained vocabulary
        return df_to_vectorize.apply(
            lambda lst: cus.vectorize(self.tfidf_vectorizer.vocabulary_, lst)
        )
    
    def get_vocab_size(self) -> int:
        return len(self.tfidf_vectorizer.vocabulary_)