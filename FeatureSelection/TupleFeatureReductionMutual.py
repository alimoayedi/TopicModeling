import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import scipy.sparse as sp

class TupleFeatureReductionMutual:
    def __init__(self, keep_fraction=0.1):
        """
        keep_fraction: The percentage of tuples to retain (deafult 0.1 == 10%)
        """
        self.keep_fraction = keep_fraction
        self.selector = None
        self.selected_tuple_indices = None
        self.selected_tuple_vocab = None

    def fit_transform(self, X_train, y_train, tuple_vocabulary):
        """
        Selects the most informative tuples using Mutual Information.
        
        X_train: The vectorized tuple matrix (sparse or dense)
        y_train: The labels for the training set (Classes 0, 1, 2, 3)
        tuple_vocabulary: The list of string tuples corresponding to X_train columns
        """
        print(f"--- Starting Mutual Information Tuple Reduction ---")
        num_original_features = X_train.shape[1]
        
        # Calculate exactly how many features is 10%
        k_to_keep = max(1, int(num_original_features * self.keep_fraction))
        print(f"Original Tuples: {num_original_features} | Tuples to Keep: {k_to_keep}")

        # 1. Initialize the Mutual Information Selector
        # mutual_info_classif handles class imbalances naturally by calculating 
        # information gain rather than global margin error.
        self.selector = SelectKBest(score_func=mutual_info_classif, k=k_to_keep)

        # 2. Fit the selector to the training data and transform it
        X_train_reduced = self.selector.fit_transform(X_train, y_train)

        # 3. Retrieve the indices of the tuples that survived the cut
        self.selected_tuple_indices = self.selector.get_support(indices=True)

        # 4. Map the surviving indices back to the actual text tuples 
        # (This is CRITICAL for the FeatureGenerator to build the Adjacency Matrix later)
        self.selected_tuple_vocab = [tuple_vocabulary[i] for i in self.selected_tuple_indices]

        print(f"Reduction Complete. New Training Matrix Shape: {X_train_reduced.shape}")
        
        return X_train_reduced, self.selected_tuple_vocab

    def transform(self, X_target):
        """
        Applies the fitted MI selector to the Validation or Test sets.
        Ensures the exact same 10% of features are kept.
        """
        if self.selector is None:
            raise ValueError("The selector has not been fitted. Call fit_transform first on training data.")
        
        X_target_reduced = self.selector.transform(X_target)
        return X_target_reduced
    
    def get_selected_vocabulary(self):
        """Returns the list of surviving tuple strings."""
        if self.selected_tuple_vocab is None:
            raise ValueError("Vocabulary not yet extracted. Run fit_transform first.")
        return self.selected_tuple_vocab