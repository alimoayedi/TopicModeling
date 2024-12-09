import numpy as np
import pandas as pd
import CustomFeatureSelection
from CustomDimensionReduction import CustomDimensionReduction
from FeatureGenerator import FeatureGenerator
import CustomFuncLib as cus

from sklearn.feature_selection import SequentialFeatureSelector



# variables 
max_doc_length = 256 # embedding and truncation
apply_cosine_similarity_reduction = False
favorite_topics = ['acq', 'corn', 'crude', 'earn']
num_topics = len(favorite_topics)

# load data
trainDocs, trainTopics, valDocs, valTopcis, testDocs, testTopics = cus.load_data()

feature_generator = FeatureGenerator(max_doc_length=max_doc_length, 
                                     num_topics=num_topics,
                                     apply_cosine_similarity_reduction=True)

feature_generator.setDataset(train=trainDocs, trainTopics=trainTopics['topics_lst'], validation=valDocs, test=testDocs)

trainDocs['trimmed'], valDocs['trimmed'], testDocs['trimmed'] = feature_generator.truncate_documents(trainDocs['preprocess'], 
                                                                                                  valDocs['preprocess'], 
                                                                                                  testDocs['preprocess'], 
                                                                                                  max_doc_length)
trainDocs, valDocs, testDocs = feature_generator.generateFeatures()

# vectorized padded data will be used for different embedding models
trainDocs['embedd'] = trainDocs['vectorized_padded']
valDocs['embedd'] = valDocs['vectorized_padded']
testDocs['embedd'] = testDocs['vectorized_padded']

features_lst=['vectorized_padded', 'embedd', 'pos_padded', 'tf', 'tfidf', 'term_topic_weight', 'tuple_2']
train_selected_features, val_selected_features, test_selected_features = trainDocs[features_lst], valDocs[features_lst], testDocs[features_lst]

train_labels_array = np.array(trainTopics['one_hot'].tolist())
validation_labels_array = np.array(valTopcis['one_hot'].tolist())

# number of unique terms in the whole database
vocab_size = cus.get_number_of_tokens(trainDocs['vectorized_padded'])

# settings that are used for training the NN model.
# more layers for each feature can be added, however, the number of
# layers for different parameters must match
feature_settings={
    'filter_sizes': [[512, 256, 128, 128, 64], # vectorized_padded
                     [512, 256, 128, 128, 64], # embedd
                     [512, 256, 128, 128, 64], # pos_padded
                     [512, 256, 128, 128, 64], # tf
                     [512, 256, 128, 128, 64], # tf_idf
                     [512, 256, 128, 128, 64], # term_topic_weight
                     [512, 256, 128, 128, 64]], # tuple_2
    'kernel_sizes': [[3, 3, 3, 3, 3],
                     [3, 3, 3, 3, 3],
                     [3, 3, 3, 3, 3],
                     [3, 3, 3, 3, 3],
                     [3, 3, 3, 3, 3],
                     [3, 3, 3, 3, 3],
                     [3, 3, 3, 3, 3]],
    'pool_sizes': [[3, 3, 3, 2, 2],
                   [3, 3, 3, 2, 2],
                   [3, 3, 3, 2, 2],
                   [3, 3, 3, 2, 2],
                   [3, 3, 3, 2, 2],
                   [3, 3, 3, 2, 2],
                   [3, 3, 3, 2, 2]],
    'embedding':[False, True, False, False, False, False, False],
    'feature_dim':[max_doc_length, max_doc_length, max_doc_length, vocab_size, vocab_size, 1024, 4578]
}

# convert features setting into form of a dataframe
feature_settings = pd.DataFrame(feature_settings, index=features_lst)

# dense layer settings
dense_settings = {
    'dense':[64, 32, 16],
    'dropout':[0.1, 0.1, 0.1]
}

# convert dense dictionary into a dataframe
dense_settings = pd.DataFrame(dense_settings)


fs = CustomFeatureSelection.CustomFeatureSelection(train_df=train_selected_features,
                                                   train_labels=train_labels_array,
                                                   test_df=val_selected_features,
                                                   test_labels=validation_labels_array,
                                                   num_classes = num_topics)


fs.embedding_setup(vocab_size=vocab_size, embedded_output_dim=max_doc_length)

fs.UnEqualSizedFeatureSelection.forward_selection(feature_settings, dense_settings, evaluation = 'report')


# fs.equalSizedFeatureSelection()





##############################################################
# Dimension Reduction

dimensionReduction = CustomDimensionReduction(features_df=train_selected_features, labels_df=trainTopics['topics_lst'], max_feature_dim=max_doc_length)
dimensionReduction.featureDimensionCheck()
mismatched_features = dimensionReduction.mismatched_features

filter_sizes = [512, 512, 256, 256, 128, 128, 256]
kernel_sizes = [3, 3, 3, 3, 3, 3, 3]
pool_sizes = [3, 3, 3, 3, 2, 2, 2]

for feature in mismatched_features:
    trainDocs['re_'+feature] = dimensionReduction.generate_fit(feature, filter_sizes=filter_sizes, kernel_sizes=kernel_sizes, pool_sizes=pool_sizes, padding='same').tolist()
    valDocs['re_'+feature] = dimensionReduction.fit(valDocs[feature]).tolist()
    testDocs['re_'+feature] = dimensionReduction.fit(testDocs[feature]).tolist()

# Forward or backward selection
train_features_lst = ['pos_padded', 're_tf', 're_tfidf', 're_term_topic_weight', 're_tuple_2']

train_labels_array = np.array(trainTopics['one_hot'].tolist())
validation_labels_array = np.array(valTopcis['one_hot'].tolist())

fs = CustomFeatureSelection.CustomFeatureSelection(trainDocs[train_features_lst], train_labels_array, valDocs[train_features_lst], validation_labels_array, feature_dim=max_doc_length, num_classes=num_topics)

vocab_size = cus.get_number_of_tokens(trainDocs['vectorized_padded'])
fs.embedding_setup(vocab_size=vocab_size, embedded_output_dim=max_doc_length)

fs.forward_feature_selection(evaluation = 'micro')

fs.backward_feature_selection(evaluation = 'micro')


