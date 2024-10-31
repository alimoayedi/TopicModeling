import numpy as np
import pandas as pd

from tensorflow.keras.layers import Input, Conv1D, AveragePooling1D, GlobalAveragePooling1D
from tensorflow.keras.models import Model

class CustomDimensionReduction:
    def __init__(self, train_features_df, labels_df, max_feature_dim):
        self.features_df = train_features_df
        self.labels_df = labels_df
        self.max_feature_dim = max_feature_dim
        self.mismatched_features = None
        self.model = None
    
    def featureDimensionCheck(self):
        # features dimension
        features_dim = self.features_df.apply(lambda col: len(col.iloc[0]))
        if not features_dim.eq(self.max_feature_dim).all() : # all similar dimension
            self.mismatched_features = features_dim[features_dim > self.max_feature_dim].index.tolist()
        
    def generateModel(self, interested_feature, filter_sizes, kernel_sizes, pool_sizes, padding):

        if len(filter_sizes) == len(kernel_sizes) == len(pool_sizes):
            model_hyper_parameters = list(zip(filter_sizes, kernel_sizes, pool_sizes))
        else:
            raise ValueError("The lists 'filter_sizes', 'kernel_sizes', and 'pool_sizes' do not match in size.")

        feature_shape = np.array(self.features_df[interested_feature].iloc[0]).shape

        input_layer = Input(shape=(feature_shape[0],1))
        passing_layer = input_layer

        for setting in model_hyper_parameters:
            passing_layer = Conv1D(filters=setting[0], kernel_size=setting[1], activation='relu', padding=padding)(passing_layer)
            passing_layer = AveragePooling1D(pool_size=setting[2])(passing_layer)

        output_layer = GlobalAveragePooling1D()(passing_layer)
        
        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.model.summary()
        
        return self.model

    def generate_fit(self, interested_feature, filter_sizes, kernel_sizes, pool_sizes, padding='valid'):
        
        self.generateModel(interested_feature, filter_sizes, kernel_sizes, pool_sizes, padding=padding)

        feature_data = np.array([np.array(doc).reshape(-1, 1) for doc in self.features_df[interested_feature]])

        model_output = self.model.predict(feature_data)

        return model_output
    
    def fit(self, data_df):
        if self.model is None:
            raise TypeError("No model found. First generate a model.")
        
        feature_data = np.array([np.array(doc).reshape(-1, 1) for doc in data_df])

        model_output = self.model.predict(feature_data)

        return model_output




        


