import pandas as pd
import CustomFuncLib as cus
from sklearn.feature_selection import RFE

class TupleFeatureReduction:
    def __init__(self, estimator, num_features_to_select, num_steps):
        self.estimator = estimator
        self.num_features_to_select = num_features_to_select
        self.num_steps = num_steps
        self.selected_features = None
        self.selected_features_rank = None

    def __feature_selection_rfe(self, pair_tokens_df, labels_df):

        # Create the RFE object with desired estimator and number of features to select
        selector = RFE(estimator=self.estimator, n_features_to_select=self.num_features_to_select, step=self.num_steps)

        # Fit the RFE on the training data
        selector.fit(pair_tokens_df, labels_df)

        # Extract the selected features based on RFE rankings
        self.selected_features_rank = selector.ranking_

        # Get the selected features based on model performance
        self.selected_features = pair_tokens_df.columns[selector.support_]

    def fit(self, pair_tokens_df, labels_df):
        """
        Fit the model to select pairs based on training data.
        """
        # Select features (pairs) using RFE
        self.__feature_selection_rfe(pair_tokens_df, labels_df)
        print("No. of selected features:", len(self.selected_features))
        
        # Return the transformed training dataframe
        return self.selected_features_rank, self.selected_features


