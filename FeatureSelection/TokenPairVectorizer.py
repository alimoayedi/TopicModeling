import pandas as pd
import CustomFuncLib as cus

class TokenPairVectorizer:
    def __init__(self, tuple_size=2):
        self.unique_tuples_lst = None  # Will store the unique token pairs found during training
        self.tuple_size = tuple_size  # Size of the tuple, default is 2
        
    def fit(self, vectorized_df):
        """
        Fit method to compute unique token pairs from training data.
        """
        # Step 1: Compute tuple_2 (token pairs) for each document in the training data
        doc_tuples_df = vectorized_df.apply(lambda lst: cus.make_tuple(lst, self.tuple_size))

        # Step 2: Extract unique token pairs from the training data
        self.unique_tuples_lst = cus.get_unique_token_pairs(doc_tuples_df)
        print('Tuple_2 unique tokens: ' + str(len(self.unique_tuples_lst)))
        
        tuple_doc_freq_df = pd.DataFrame(0, index=vectorized_df.index, columns=self.unique_tuples_lst)
       
        for index, doc_tuple_lst in doc_tuples_df.items():
            tuple_doc_freq_df.loc[index] = cus.get_token_pairs_count(self.unique_tuples_lst, doc_tuple_lst)
          
        return tuple_doc_freq_df  # Return the DataFrame containing the counts
    
    def transform(self, vectorized_df):
        """
        Transform method to compute the counts of the token pairs in validation/test data.
        """
        # Step 1: Compute tuple_2 (token pairs) for each document in the validation/test data
        doc_tuples_df = vectorized_df.apply(lambda lst: cus.make_tuple(lst, self.tuple_size))
        
        tuple_doc_freq_df = pd.DataFrame(0, index=vectorized_df.index, columns=self.unique_tuples_lst)

        for index, doc_tuple_lst in doc_tuples_df.items():
            tuple_doc_freq_df.loc[index] = cus.get_token_pairs_count(self.unique_tuples_lst, doc_tuple_lst)
        
        return tuple_doc_freq_df  # Return the DataFrame containing the counts
