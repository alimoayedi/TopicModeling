import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
import scipy.sparse as sp

class TupleFeatureReductionMutual:
    def __init__(self, keep_fraction=0.1):
        """
        keep_fraction: The percentage of tuples to retain (default 0.1 == 10%).
        """
        self.keep_fraction = keep_fraction
        self.selector = None
        self.selected_tuple_indices = None
        self.selected_tuple_vocab = None

    def fit_transform(self, X_train, y_train, tuple_vocabulary):
        """
        Selects the most informative tuples using the Chi-Square (chi2) statistic.
        """
        print(f"--- Starting Chi-Square Tuple Reduction ---")
        num_original_features = X_train.shape[1]
        
        # Calculate exactly how many features is the requested percentage
        proportional_k = max(1, int(num_original_features * self.keep_fraction))
        
        print(f"Original Tuples: {num_original_features} | Target ({(self.keep_fraction*100):.1f}%): {proportional_k}")

        # 1. Initialize the Chi-Square Selector (fast and accurate for sparse data)
        self.selector = SelectKBest(score_func=chi2, k=proportional_k)

        # 2. Fit the selector to the training data and transform it
        X_train_reduced = self.selector.fit_transform(X_train, y_train)

        # 3. Retrieve the indices of the tuples that survived the cut
        self.selected_tuple_indices = self.selector.get_support(indices=True)

        # 4. Map the surviving indices back to the actual text tuples 
        self.selected_tuple_vocab = [tuple_vocabulary[i] for i in self.selected_tuple_indices]

        print(f"Reduction Complete. New Training Matrix Shape: {X_train_reduced.shape}")
        
        return X_train_reduced, self.selected_tuple_vocab

    def transform(self, X_target):
        """
        Applies the fitted selector to the Validation or Test sets.
        """
        if self.selector is None:
            raise ValueError("The selector has not been fitted. Call fit_transform first on training data.")
        
        return self.selector.transform(X_target)
    
    def get_selected_vocabulary(self):
        if self.selected_tuple_vocab is None:
            raise ValueError("Vocabulary not yet extracted. Run fit_transform first.")
        return self.selected_tuple_vocab