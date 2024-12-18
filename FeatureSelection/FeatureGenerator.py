import sys
sys.path.insert(1, '/Users/Ali/Documents')

from TextVectorizationModel import TextVectorizationModel
from TokenPairVectorizer import TokenPairVectorizer
from TupleFeatureReduction import TupleFeatureReduction
import CustomFuncLib as cus
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVR
from xgboost import XGBClassifier


class FeatureGenerator():
    def __init__(self, max_doc_length, num_topics, apply_cosine_similarity_reduction=False):
        self.max_doc_length = max_doc_length
        self.num_topics = num_topics
        self.apply_cosine_similarity_reduction = apply_cosine_similarity_reduction
        self.trainDocs = None
        self.trainTopics = None
        self.valDocs = None
        self.testDocs = None
    
    def setDataset(self, train, trainTopics, validation, test):
        self.trainDocs = train
        self.trainTopics = trainTopics
        self.valDocs = validation
        self.testDocs = test

    def truncate_documents(self, train_df, validate_df, test_df):
        # truncate document
        train_truncated = train_df.apply(lambda doc: cus.trim_documents(doc, self.max_doc_length))
        val_truncated = validate_df.apply(lambda doc: cus.trim_documents(doc, self.max_doc_length))
        test_truncated = test_df.apply(lambda doc: cus.trim_documents(doc, self.max_doc_length))
        return train_truncated, val_truncated, test_truncated

    def __get_term_topic_df(self, vectorized_df, topics_df, terms_lst, multi_label=False):
        # generate the Term-Topic Dictionary
        vocab_size = cus.get_number_of_tokens(vectorized_df)
        token_topic_df = pd.DataFrame(np.zeros(shape=(vocab_size, self.num_topics)), columns=range(self.num_topics), index=terms_lst)

        for index in self.trainTopics.index:
            topics_lst = topics_df.loc[index]
            term_vector = vectorized_df.loc[index]
            for topic in topics_lst:
                for term in term_vector:
                    token_topic_df.loc[term][topic] += 1
        
        if not multi_label:
            for topic in token_topic_df.columns:
                col_of_topic = token_topic_df[topic] != 0
                other_cols = token_topic_df.drop(columns=topic).eq(0).all(axis=1)

                # Combine the conditions
                condition = col_of_topic & other_cols

                # Mark the rows that meet the condition with 1
                token_topic_df['mark'] = np.where(condition, 1, 0)
                topic_terms_index = token_topic_df.sort_values(by=['mark',topic], ascending=False).iloc[100:].index
                token_topic_df.loc[topic_terms_index, topic] = 0
            token_topic_df.drop(columns='mark', inplace=True)
        
        return token_topic_df

    def __get_term_topic_weights(self, vectorized_df, token_topic_df):
        
        # train term-topic weights
        token_topic_weight = pd.DataFrame(None, columns=range(self.num_topics), index=vectorized_df.index)

        for topic in list(token_topic_df.columns):
            token_topic_weight[topic] = vectorized_df.apply(lambda lst: [token_topic_df.loc[term][topic] for term in lst])
        
        return token_topic_weight

    def generateFeatures(self):
        # TODO trimmed documents are in form of list. We need to join the tokens and form the texts again.

        # Vectorization of the train documents 
        vectorization_model = TextVectorizationModel(apply_cosine_similarity_reduction=False)
        self.trainDocs.loc[:, 'vectorized'] = vectorization_model.fit(self.trainDocs['trimmed'])
        self.valDocs.loc[:, 'vectorized'] = vectorization_model.transform(self.valDocs['trimmed'])
        self.testDocs.loc[:, 'vectorized'] = vectorization_model.transform(self.testDocs['trimmed'])

        # pad vectorized documents
        self.trainDocs.loc[:,'vectorized_padded'] = self.trainDocs['vectorized'].apply(lambda lst: cus.padding(lst, self.max_doc_length))
        self.valDocs.loc[: 'vectorized_padded'] = self.valDocs['vectorized'].apply(lambda lst: cus.padding(lst, self.max_doc_length))
        self.testDocs.loc[: 'vectorized_padded'] = self.testDocs['vectorized'].apply(lambda lst: cus.padding(lst, self.max_doc_length))

        # calculation of the tf score for train, validation and test data
        train_unique_terms = list(set([token for tokenized_doc in self.trainDocs['vectorized'] for token in tokenized_doc]))
        train_tf = pd.DataFrame(
            vectorization_model.tf_matrix[:, train_unique_terms].toarray(), columns=train_unique_terms, index=self.trainDocs.index
            )
        self.trainDocs.loc[:,'tf'] = pd.DataFrame(train_tf.apply(cus.create_list, axis=1)).iloc[:, 0].values

        val_joined_tokens = self.valDocs['trimmed'].apply(cus.join_tokens)
        val_tf_matrix = vectorization_model.count_vectorizer.transform(val_joined_tokens)
        val_tf = pd.DataFrame(
            val_tf_matrix[:, train_unique_terms].toarray(), columns=train_unique_terms, index=self.valDocs.index
            )
        self.valDocs.loc[:, 'tf'] = pd.DataFrame(val_tf.apply(cus.create_list, axis=1)).iloc[:, 0].values

        test_joined_tokens = self.testDocs['trimmed'].apply(cus.join_tokens)
        test_tf_matrix = vectorization_model.count_vectorizer.transform(test_joined_tokens)
        test_tf = pd.DataFrame(test_tf_matrix[:, train_unique_terms].toarray(), columns=train_unique_terms, index=self.testDocs.index)
        self.testDocs.loc[:, 'tf'] = pd.DataFrame(test_tf.apply(cus.create_list, axis=1)).iloc[:, 0].values

        # calculation of tf-idf score for train, validation, and test sets
        train_tfidf = pd.DataFrame(vectorization_model.tfidf_matrix[:, train_unique_terms].toarray(), columns=train_unique_terms, index=self.trainDocs.index)
        self.trainDocs.loc[:, 'tfidf'] = pd.DataFrame(train_tfidf.apply(cus.create_list, axis=1)).iloc[:, 0].values

        val_tfidf_matrix = vectorization_model.tfidf_vectorizer.transform(val_joined_tokens)
        val_tfidf = pd.DataFrame(val_tfidf_matrix[:, train_unique_terms].toarray(), columns=train_unique_terms, index=self.valDocs.index)
        self.valDocs.loc[:, 'tfidf'] = pd.DataFrame(val_tfidf.apply(cus.create_list, axis=1)).iloc[:, 0].values

        test_tfidf_matrix = vectorization_model.tfidf_vectorizer.transform(test_joined_tokens)
        test_tfidf = pd.DataFrame(test_tfidf_matrix[:, train_unique_terms].toarray(), columns=train_unique_terms, index=self.testDocs.index)
        self.testDocs.loc[:, 'tfidf'] = pd.DataFrame(test_tfidf.apply(cus.create_list, axis=1)).iloc[:, 0].values

        # calculate POS tags of vectorized documents in train, validation and test dataset
        self.trainDocs.loc[:, 'pos_tag'] = self.trainDocs['preprocess'].apply(lambda lst: cus.pos_tagger(lst))
        self.valDocs.loc[:, 'pos_tag'] = self.valDocs['preprocess'].apply(lambda lst: cus.pos_tagger(lst))
        self.testDocs.loc[:, 'pos_tag'] = self.testDocs['preprocess'].apply(lambda lst: cus.pos_tagger(lst))

        self.trainDocs.loc[:, 'pos_padded'] = self.trainDocs['pos_tag'].apply(lambda lst: cus.padding(lst, self.max_doc_length))
        self.valDocs.loc[:, 'pos_padded'] = self.valDocs['pos_tag'].apply(lambda lst: cus.padding(lst, self.max_doc_length))
        self.testDocs.loc[:, 'pos_padded'] = self.testDocs['pos_tag'].apply(lambda lst: cus.padding(lst, self.max_doc_length))

        token_topic_df = self.__get_term_topic_df(self.trainDocs['vectorized'], self.trainTopics, train_unique_terms, multi_label=False)
        token_topic_weight_df = self.__get_term_topic_weights(self.trainDocs['vectorized'], token_topic_df)

        # term-topic weight of train documents
        for topic in list(token_topic_weight_df.columns):
            token_topic_weight_df[topic] = token_topic_weight_df[topic].apply(lambda lst: cus.padding(lst, self.max_doc_length))

        # concatinate term-topic weights
        self.trainDocs.loc[:, 'term_topic_weight'] = token_topic_weight_df.apply(lambda lst: cus.concatenate_arrays(lst),  axis=1)


        # term-topic weight of validation documents
        validation_term_topic_df = pd.DataFrame(None, columns=range(self.num_topics), index=self.valDocs.index)

        for topic in list(validation_term_topic_df.columns):
            validation_term_topic_df[topic] = self.valDocs['vectorized'].apply(
                lambda lst: [token_topic_df.loc[term][topic] if term in token_topic_df.index else 0 for term in lst]
                )
            validation_term_topic_df[topic] = validation_term_topic_df[topic].apply(lambda lst: cus.padding(lst, self.max_doc_length))

        self.valDocs.loc[:, 'term_topic_weight'] = validation_term_topic_df.apply(lambda lst: cus.concatenate_arrays(lst),  axis=1)


        # term-topic weight of test documents
        test_term_topic_df = pd.DataFrame(None, columns=range(self.num_topics), index=self.testDocs.index)

        for topic in list(test_term_topic_df.columns):
            test_term_topic_df[topic] = self.testDocs['vectorized'].apply(
                lambda lst: [token_topic_df.loc[term][topic] if term in token_topic_df.index else 0 for term in lst]
                )
            test_term_topic_df[topic] = test_term_topic_df[topic].apply(lambda lst: cus.padding(lst, self.max_doc_length))

        self.testDocs.loc[:, 'term_topic_weight'] = test_term_topic_df.apply(lambda lst: cus.concatenate_arrays(lst),  axis=1)


        pair_vectorizer = TokenPairVectorizer(tuple_size=2)
        # Fit the model on the training data
        train_doc_tuple_df = pair_vectorizer.fit(self.trainDocs['vectorized'])

        # Apply the model to the validation or test data
        val_doc_tuple_df = pair_vectorizer.transform(self.valDocs['vectorized'])
        test_doc_tuple_df = pair_vectorizer.transform(self.testDocs['vectorized'])


        svc = LinearSVR(max_iter=5000)
        # xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

        train_single_label_df = self.trainTopics.apply(lambda lst: lst[0])

        tuple_selector = TupleFeatureReduction(estimator=svc, num_features_to_select=0.1, num_steps=100)
        # Train (fit) on the training data
        _ , selected_tuples_lst = tuple_selector.fit(train_doc_tuple_df, train_single_label_df)

        self.trainDocs.loc[:, 'tuple_2'] = train_doc_tuple_df[selected_tuples_lst].apply(cus.create_list, axis=1)
        self.valDocs.loc[:, 'tuple_2'] = val_doc_tuple_df[selected_tuples_lst].apply(cus.create_list, axis=1)
        self.testDocs.loc[:, 'tuple_2'] = test_doc_tuple_df[selected_tuples_lst].apply(cus.create_list, axis=1)

        return self.trainDocs, self.valDocs, self.testDocs

