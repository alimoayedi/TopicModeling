from LoadDataset import LoadReutersDataset
import Tokenizer
import CustomFuncLib as cus
import warnings

import pandas as pd
import numpy as np

from keras.utils import to_categorical


class ReutersPreprocessor:

    def __init__(self):
        self.favorite_topics = None
        pass

    # replace topics with their indexes, unfound topics are replaced by 99!
    def __replace_with_index(self, lst):
        return [self.favorite_topics.index(topic) for topic in lst if topic in self.favorite_topics ]

    def preprocess(self, documents_dic, topics_dic, num_sample, min_doc_lenght, favorite_topics):
        
        self.favorite_topics = favorite_topics
        # file_directory = '/content/drive/MyDrive/ColabNotebooks'
        # rand_sample = 2000
        # minimum_doc_length = 6

        # loader = LoadReutersDataset(data_path=file_directory + '/reuters21578')
        # documents_dic, topics_dic, _, _, _, _, _ = loader.load()

        documents = pd.DataFrame.from_dict(documents_dic, orient='index', columns=['doc'])
        topics = pd.DataFrame.from_dict(topics_dic, orient='index')

        # # If you want to name the index, you can set the index name
        documents.index.name = 'index'
        topics.index.name = 'index'

        # remove all the documents without any specific topic
        topicless_documents = topics.notna().any(axis=1)
        documents = documents[topicless_documents]
        topics = topics[topicless_documents]

        # filter documents and keep only the ones with favorite topics
        # favorite_topics = ['acq', 'money-fx', 'grain', 'crude', 'trade', 'interest', 'ship', 'wheat', 'corn', 'oilseed']
        # favorite_topics = ['acq', 'corn', 'crude', 'earn']
        documents = documents[topics.isin(favorite_topics).any(axis=1)]
        topics = topics[topics.isin(favorite_topics).any(axis=1)]

        documents = pd.DataFrame(documents.sample(n=num_sample, random_state=42, replace=False))
        topics = pd.DataFrame(topics.loc[documents.index])

        # preprocess data by tokenization and lemmatization
        documents['preprocess'] = documents['doc'].apply(lambda text: Tokenizer.tokenize(text, lemmatize=True))

        # drop preprocessed documents with length less than 6
        topics = topics[documents['preprocess'].str.len() > min_doc_lenght]
        documents = documents[documents['preprocess'].str.len() > min_doc_lenght]

        # remove duplicate terms from each document
        # documents['preprocess'] = documents['preprocess'].apply(remove_duplicates_terms)

        #join preprocced tokens to make a string. used in tf-idf and cosine scoring.
        documents['joined_tokens'] = documents['preprocess'].apply(lambda tokens: " ".join(tokens))

        # combine all the topics into a list
        topics['topics_lst'] = topics.values.tolist()

        # replace topics with their indexes, unfound topics are replaced by zero
        topics['topics_lst'] = topics['topics_lst'].apply(self.__replace_with_index)

        # convert labels into a one-hot coding
        topics['one_hot'] = topics['topics_lst'].apply(lambda topic_lst: list(np.sum(to_categorical(topic_lst, num_classes=len(favorite_topics)), axis=0)))

        rand = random.randint(10,99)

        trainValDocs, testDocs, trainValTopics, testTopics = train_test_split(documents, topics, test_size=0.2, random_state=rand)
        trainDocs, valDocs, trainTopics, valTopcis = train_test_split(trainValDocs, trainValTopics, test_size=0.2, random_state=rand)
        print(trainDocs.shape)
        print(trainTopics.shape)
        print(valDocs.shape)
        print(valTopcis.shape)
        print(testDocs.shape)
        print(testTopics.shape)
        print("\n_______________________\n")
        # Print the count of documents in each category for train, validation, and test sets
        for topic in favorite_topics:
            length = len(trainTopics.index[trainTopics.applymap(lambda x: x == topic).any(axis=1)])
            print("Category ", topic, " counts in the train set:", length)
            length = len(valTopcis.index[valTopcis.applymap(lambda x: x == topic).any(axis=1)])
            print("Category ", topic, " counts in the validation set:", length)
            length = len(testTopics.index[testTopics.applymap(lambda x: x == topic).any(axis=1)])
            print("Category ", topic, " counts in the test set:", length)
            print("_______________________")