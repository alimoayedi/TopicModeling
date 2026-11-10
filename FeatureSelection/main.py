import nltk
import numpy as np
import pandas as pd
import random

import nltk
from nltk.corpus import reuters

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SequentialFeatureSelector

import tensorflow as tf

# from LoadDataset import LoadReutersDataset
from ReutersPreprocessor import ReutersPreprocessor
from DatasetVisualization import DatasetVisualization
import CustomFeatureSelection
from CustomDimensionReduction import CustomDimensionReduction
from FeatureGenerator import FeatureGenerator
import FeatureSelectionWithAttentionLayerGenetic
import CustomFuncLib as cus
from DocumentSummarization import DocumentSummarization
from DocumentSummarizerExtractive import DocumentSummarizerExtractive
from SideStreamFeatureGenerator import SideStreamFeatureGenerator

nltk.download('averaged_perceptron_tagger_eng')
nltk.download('reuters')

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

def process_topics(df, top_topics):
    def filter_and_sample(topics):
        valid = [t for t in topics if t in top_topics]
        return random.choice(valid) if valid else None

    # Create the new column and drop rows that don't match any top 10 topic
    df['selected_topic'] = df['topics_lst'].apply(filter_and_sample)
    return df.dropna(subset=['selected_topic'])



fileids = reuters.fileids()
data = []
for file_id in fileids:
    data.append({
        'file_id': file_id,
        'split': 'train' if file_id.startswith('training/') else 'test',
        'topics_lst': reuters.categories(file_id),
        'doc': reuters.raw(file_id)
    })

df = pd.DataFrame(data)

# 2. Split into Train and Test
train_df = df[df['split'] == 'train'].copy()
test_df = df[df['split'] == 'test'].copy()

# 3. Identify Top 10 Topics (based on the full dataset or just training)
# It is best practice to define top topics based on the training set
top_10_topics = train_df['topics_lst'].explode().value_counts().nlargest(10).index.tolist()

# 5. Apply to both splits
train_df = process_topics(train_df, top_10_topics)
test_df = process_topics(test_df, top_10_topics)

print(f"Training set size: {len(train_df)}")
print(f"Testing set size: {len(test_df)}")
print(f"Top 10 Topics: {top_10_topics}")

trainTopics = train_df[['topics_lst', 'selected_topic']]
testTopics = test_df[['topics_lst', 'selected_topic']]

# 1. Initialize the LabelEncoder
encoder = LabelEncoder()

# 2. Fit the encoder on the training data (to learn the mapping)
# We use the training data only to ensure the encoder doesn't see test data labels
encoder.fit(trainTopics['selected_topic'])

# 3. Transform both train and test splits
# This replaces the strings with integers (e.g., 'acq' -> 0, 'crude' -> 1, etc.)
trainTopics['topic_id'] = encoder.transform(trainTopics['selected_topic'])
testTopics['topic_id'] = encoder.transform(testTopics['selected_topic'])

# 4. Verification: Print the mapping to know which ID belongs to which topic
label_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
print("Mapping created:", label_mapping)

# Convert to one-hot format
num_classes = len(encoder.classes_)
train_labels = tf.keras.utils.to_categorical(trainTopics['topic_id'], num_classes)
test_labels = tf.keras.utils.to_categorical(testTopics['topic_id'], num_classes)

trainTopics['one_hot'] = train_labels.tolist()
testTopics['one_hot'] = test_labels.tolist()

print(f"Shape of one-hot encoded labels: {train_labels.shape}")

# @title Dropoff
dropoff_percent = 0.3

# split data into train and test
rand = random.randint(10,99)
train_df, _, trainTopics, _ = train_test_split(train_df, trainTopics, test_size=dropoff_percent, random_state=rand, stratify=trainTopics['selected_topic'])

pp = ReutersPreprocessor()

train_df = pp.preprocess(train_df, min_doc_length = 6)
test_df = pp.preprocess(test_df, min_doc_length = 6)

trainTopics = trainTopics.loc[train_df.index]
testTopics = testTopics.loc[test_df.index]

trainTopics['topic_id'] = trainTopics['topic_id'].apply(lambda x: [x])
testTopics['topic_id'] = testTopics['topic_id'].apply(lambda x: [x])

train_df.shape, trainTopics.shape, test_df.shape, testTopics.shape

#Generate Extractive Summaries

summarizer = DocumentSummarizerExtractive(max_sentences=8)

print("Generating summaries for the training set...")
train_df['summary'] = train_df['doc'].apply(summarizer.generate_summary)

print("Generating summaries for the testing set...")
test_df['summary'] = test_df['doc'].apply(summarizer.generate_summary)

# Split validation set from train set
# variables
val_percentage = 0.1 # percentage of data used for test and validation

# split data into train and test
rand = random.randint(10,99)
trainDocs, valDocs, trainTopics, valTopics = train_test_split(trainDocs, trainTopics, test_size=val_percentage, random_state=rand, stratify=trainTopics['selected_topic'])
testDocs = test_df

trainDocs.shape, trainTopics.shape, valDocs.shape, valTopics.shape, testDocs.shape, testTopics.shape

side_feature_generator = SideStreamFeatureGenerator(max_features=5000, use_tfidf=True)
# Pass the entire column as a list to build the matrix, then map back to the dataframe
trainDocs['original_tfidf'] = list(side_feature_generator.fit_transform(trainDocs['doc'].tolist()))
valDocs['original_tfidf'] = list(side_feature_generator.transform(valDocs['doc'].tolist()))
testDocs['original_tfidf'] = list(side_feature_generator.transform(testDocs['doc'].tolist()))

max_doc_length = 256
num_topics = len(top_10_topics)

# Route the generated extractive summaries directly into the main sequence stream
trainDocs.loc[:, 'trimmed'] = trainDocs['summary']
valDocs.loc[:, 'trimmed'] = valDocs['summary']
testDocs.loc[:, 'trimmed'] = testDocs['summary']

feature_generator = FeatureGenerator(max_doc_length=max_doc_length,
                                     num_topics=num_topics,
                                     apply_cosine_similarity_reduction=True)

feature_generator.setDataset(train=trainDocs, trainTopics=trainTopics['topic_id'], validation=valDocs, test=testDocs)

vectorizedModel, lda_model, trainDocs, valDocs, testDocs = feature_generator.generateFeatures()

dynamic_ner_dim = len(trainDocs.iloc[0]['ner_density'])
print(f"Detected Dynamic NER Dimension: {dynamic_ner_dim}")

dynamic_embed_dim = len(trainDocs.iloc[0]['contextual_embedd'])
print(f"Detected Dynamic Contextual Embed Dimension: {dynamic_embed_dim}")

trainDocs['embedd'] = trainDocs['vectorized_padded']
valDocs['embedd'] = valDocs['vectorized_padded']
testDocs['embedd'] = testDocs['vectorized_padded']

features_lst = ['vectorized_padded', 'embedd', 'pos_padded', 'tf', 'tfidf', 'term_topic_weight', 'tuple_2',  'ner_density', 'contextual_embedd']
train_selected_features, val_selected_features, test_selected_features = trainDocs[features_lst], valDocs[features_lst], testDocs[features_lst]

train_labels_array = np.array(trainTopics['one_hot'].tolist())
validation_labels_array = np.array(valTopics['one_hot'].tolist())
test_labels_array = np.array(testTopics['one_hot'].tolist())

# number of unique terms in the whole database
vocab_size = vectorizedModel.vocab_size

dim_term_topic = feature_generator.num_topics * feature_generator.max_doc_length
dim_tuple_2 = feature_generator.selected_tuple_count
A_matrix = feature_generator.global_adjacency_matrix # Grab the graph structure
dynamic_ner_dim = len(trainDocs.iloc[0]['ner_density'])
dynamic_embed_dim = len(trainDocs.iloc[0]['contextual_embedd'])

# settings that are used for training the NN model.
# more layers for each feature can be added, however, the number of
# layers for different parameters must match
feature_settings={
    # Change the 7th model_type from 'dense' to 'graph'
    'model_type': ['transformer', 'transformer', 'transformer', 'dense', 'dense', 'dense', 'graph', 'dense', 'dense'],
    'num_transformer_blocks': [1, 1, 1, 0, 0, 0, 0, None, None],
    'num_heads': [2, 2, 2, 0, 0, 0, 0, None, None],
    'ff_dim': [128, 128, 128, 128, 128, 128, 128, 128, 128],
    'embedding': [False, True, False, False, False, False, False, None, None],
    'normalize': [False, False, False, True, True, True, True, None, None],
    'feature_dim': [max_doc_length, max_doc_length, max_doc_length, vocab_size, vocab_size, dim_term_topic, dim_tuple_2, dynamic_ner_dim, dynamic_embed_dim],

    # Pass the Adjacency matrix ONLY to the 7th feature branch
    'adjacency_matrix': [None, None, None, None, None, None, A_matrix, None, None]
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

lda_settings = {
    'lda_gating': True,
    'lda_dense': 64,
    'lda_dropout': 0.1
    }

fs = FeatureSelectionWithAttentionLayerGenetic.CustomFeatureSelection(train_df=train_selected_features,
                                                                      train_labels=train_labels_array,
                                                                      test_df=val_selected_features,
                                                                      test_labels=validation_labels_array,
                                                                      num_classes = num_topics)


fs.embedding_setup(vocab_size=vocab_size, embedded_output_dim=max_doc_length)
print("Starting Evolutionary Feature Selection...")
best_features, best_score = fs.UnEqualSizedFeatureSelection.genetic_algorithm_selection(
    features_lst=features_lst,
    settings=feature_settings,             # Your existing network settings
    dense_settings=dense_settings, # Your existing dense layer settings
    lda_settings=lda_settings,     # Your existing LDA settings
    epochs=10,                     # Number of epochs for training each subset
    batch_size=32,                 # Batch size for training
    evaluation='macro',            # The exact metric you want the GA to maximize
    population_size=10,            # Number of subsets per generation
    generations=15,                # How many evolutionary cycles to run
    mutation_rate=0.3              # 30% chance a child subset alters its features
)

print(f"Evolution Complete! Best Subset: {best_features} with F1: {best_score}")

