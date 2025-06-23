import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split

from LoadDataset import LoadReutersDataset
from ReutersPreprocessor import ReutersPreprocessor
from DatasetVisualization import DatasetVisualization
import CustomFeatureSelection
from CustomDimensionReduction import CustomDimensionReduction
from FeatureGenerator import FeatureGenerator
import CustomFuncLib as cus
from Summarization import Summarization

from sklearn.feature_selection import SequentialFeatureSelector


def load_original_dataset(path):
    # load data
    loader = LoadReutersDataset(path)
    documents_dic, topics_dic, _, _, _, _, _ = loader.load()
    return documents_dic, topics_dic

def load_saved_dataset():
    trainDocs = pd.read_parquet('/content/trainDocsDictionary.parquet')
    trainTopics = pd.read_parquet('/content/trainTopics.parquet')
    valDocs = pd.read_parquet('/content/valDocsDictionary.parquet')
    valTopcis = pd.read_parquet('/content/valTopics.parquet')
    testDocs = pd.read_parquet('/content/testDocsDictionary.parquet')
    testTopics = pd.read_parquet('/content/testTopics.parquet')
    return trainDocs, trainTopics, valDocs, valTopcis, testDocs, testTopics

def printDatasetDescription(trainDocs, trainTopics, valDocs, valTopcis, testDocs, testTopics):
    print("Train Shape:", trainDocs.shape, trainTopics.shape)
    for topic in favorite_topics:
        length = len(trainTopics.index[trainTopics.map(lambda x: x == topic).any(axis=1)])
        print("Category ", topic, " counts in the train set:", length)
    print("\n")
    print("Validation Shape:", valDocs.shape, valTopcis.shape)
    for topic in favorite_topics:
        length = len(valTopcis.index[valTopcis.map(lambda x: x == topic).any(axis=1)])
        print("Category ", topic, " counts in the validation set:", length)
    print("\n")
    print("Test Shape:", testDocs.shape, testTopics.shape)
    for topic in favorite_topics:
        length = len(testTopics.index[testTopics.map(lambda x: x == topic).any(axis=1)])
        print("Category ", topic, " counts in the test set:", length)



# variables
file_directory = '/content/drive/MyDrive/Colab\ Notebooks/PhD\ Thesis'
num_samples = 2000 # train+test+validation
test_percentage = 0.2 # percentage of data used for test and validation
min_doc_length = 6 # smaller docs are removed
max_doc_length = 100 # maximum doc length for embedding and truncation
apply_cosine_similarity_reduction = False
favorite_topics = ['acq', 'corn', 'crude', 'earn']
num_topics = len(favorite_topics)



# load original data
raw_documents, raw_topics = load_original_dataset(path=file_directory + '/reuters21578')

# convert to dataframe
raw_documents = pd.DataFrame.from_dict(raw_documents, orient='index', columns=['doc'])
raw_topics = pd.DataFrame.from_dict(raw_topics, orient='index')

# set the index column name
raw_documents.index.name = 'index'
raw_topics.index.name = 'index'

# remove all the documents without any specific topic
topicless_raw_documents = raw_topics.notna().any(axis=1)
raw_documents = raw_documents[topicless_raw_documents]
raw_topics = raw_topics[topicless_raw_documents]

# filter documents and keep only the ones with favorite topics
raw_documents = raw_documents[raw_topics.isin(favorite_topics).any(axis=1)]
raw_topics = raw_topics[raw_topics.isin(favorite_topics).any(axis=1)]

# take sample of documents
raw_documents = pd.DataFrame(raw_documents.sample(n=num_samples, random_state=42, replace=False))
raw_topics = pd.DataFrame(raw_topics.loc[raw_documents.index])


# preprocess data
documents, topics = ReutersPreprocessor().preprocess(documents=raw_documents,
                                                     topics=raw_topics,
                                                     min_doc_length=min_doc_length,
                                                     favorite_topics=favorite_topics,
                                                     remove_stopwords=True,
                                                     lemmatize=True,
                                                     lang='english')

# plots the distribution of documents length after preprocessing
DatasetVisualization().dataset_distribution(documents['preprocess'])


# summarize documents
# summarization = Summarization('facebook/bart-large-cnn')#model_name
# documents['summarized'] = summarization.summarize(documents['preprocess'],
#                                                  oml= 100,
#                                                  smaxl=100,
#                                                  sminl=50,
#                                                  truncation_length=512,
#                                                  truncation=True)


# documents.to_pickle("documents_summarized.pkl")
# topics.to_pickle("topics.pkl")

# Load the DataFrame back
# documents = pd.read_pickle("documents_summarized.pkl")
# topics = pd.read_pickle("topics.pkl")

# plots the distribution of documents length after summarization
DatasetVisualization().dataset_distribution(dataset = documents['summarized'])

# tokenize summarized texts as we use tokenized document for classification and feature generation
documents['tokenize_summary']=documents['summarize'].apply(lambda txt: txt.split(' '))

# only one label is saved. Single labeled classification for now. :)
topics['topic'] = topics['topics_lst'].apply(lambda lst: lst[0])


# if smaller set of summarized documents required.
# documents = pd.DataFrame(documents.sample(n=500, random_state=42, replace=False))
# topics = pd.DataFrame(topics.loc[documents.index])





# # split data into train test and 
# rand = random.randint(10,99)
# trainValDocs, testDocs, trainValTopics, testTopics = train_test_split(documents, topics, test_size=test_percentage, random_state=rand)
# trainDocs, valDocs, trainTopics, valTopcis = train_test_split(trainValDocs, trainValTopics, test_size=test_percentage, random_state=rand)



# #join preprocced tokens to make a string. used in tf-idf and cosine scoring.
# documents['joined_tokens'] = documents['preprocess'].apply(lambda tokens: " ".join(tokens))


summarization = Summarization.Summarization()#model_name
trainDocs['summarize'] = summarization.summarize(trainDocs['preprocess'], 
                                                 original_max_length = 512,
                                                 sum_max_length = 80,
                                                 sum_min_length = 50,
                                                 truncation = True)



feature_generator = FeatureGenerator(max_doc_length=max_doc_length, 
                                     num_topics=num_topics,
                                     apply_cosine_similarity_reduction=True)

feature_generator.setDataset(train=trainDocs, trainTopics=trainTopics['topics_lst'], validation=valDocs, test=testDocs)

trainDocs.loc[:, 'trimmed'] = trainDocs['preprocess'].apply(lambda doc: cus.trim_documents(doc, max_doc_length))
valDocs.loc[:, 'trimmed'] = valDocs['preprocess'].apply(lambda doc: cus.trim_documents(doc, max_doc_length))
testDocs.loc[:, 'trimmed'] = testDocs['preprocess'].apply(lambda doc: cus.trim_documents(doc, max_doc_length))

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


