from Tokenizer import Tokenizer

import pandas as pd
import numpy as np

from keras.utils import to_categorical


class ReutersPreprocessor:

    def __init__(self):
        self.favorite_topics = None

    # replace topics with their indexes, unfound topics are replaced by 99!
    def __replace_with_index(self, lst):
        return [self.favorite_topics.index(topic) for topic in lst if topic in self.favorite_topics ]

    def preprocess(self, documents, topics, min_doc_length, favorite_topics, remove_stopwords=True, lemmatize=True, lang='english'):
        
        # set favorite variables as a global variable
        self.favorite_topics = favorite_topics

        # preprocess data by tokenization and lemmatization
        documents['preprocess'] = documents['doc'].apply(lambda text: Tokenizer(lang).tokenize(text, remove_stopwords, lemmatize))

        # drop preprocessed documents with length less than 6
        topics = topics[documents['preprocess'].str.len() > min_doc_length]
        documents = documents[documents['preprocess'].str.len() > min_doc_length]

        # remove duplicate terms from each document
        # documents['preprocess'] = documents['preprocess'].apply(remove_duplicates_terms)

        # combine all the topics into a list
        topics['topics_lst'] = topics.values.tolist()

        # replace topics with their indexes, unfound topics are replaced by zero
        topics['topics_lst'] = topics['topics_lst'].apply(self.__replace_with_index)

        # convert labels into a one-hot coding
        topics['one_hot'] = topics['topics_lst'].apply(lambda topic_lst: list(np.sum(to_categorical(topic_lst, num_classes=len(favorite_topics)), axis=0)))

        return documents, topics