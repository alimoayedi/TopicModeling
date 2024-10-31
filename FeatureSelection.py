

class CustomFeatureSelection:
    def __init__(self, max_feature_dim, features_df, labels_df):
        self.max_feature_dim = max_feature_dim
        self.features_df = features_df
        self.labels_df = labels_df

    
    def featureDimensionCheck(self):
        # features dimension
        features_dim = self.features_df.apply(lambda col: len(col.iloc[0]))
        if features_dim.eq(self.max_feature_dim).all() : # all similar dimension
            return None
        else:
            return features_dim[features_dim > self.max_feature_dim].index.tolist()
        
    def sizeAdaption(self):



