import sys
sys.path.insert(1, '/Users/Ali/Documents')

from TextVectorizationModel import TextVectorizationModel
from TokenPairVectorizer import TokenPairVectorizer
from TupleFeatureReduction import TupleFeatureReduction
import CustomFuncLib as cus
import pandas as pd
import numpy as np

from sklearn.svm import LinearSVR
from sklearn.decomposition import LatentDirichletAllocation

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
        self.vocab_size = None
        self.selected_tuple_count = None
    
    def setDataset(self, train, trainTopics, validation, test):
        self.trainDocs = train
        self.trainTopics = trainTopics
        self.valDocs = validation
        self.testDocs = test

    def __get_term_topic_df(self, vectorized_df, topics_df, terms_lst, multi_label=False):
        # generate the Term-Topic Dictionary
        token_topic_df = pd.DataFrame(0, index=range(terms_lst), columns=range(self.num_topics), dtype=int)

        for index, topics_lst in topics_df.items():
            term_vector = vectorized_df.loc[index]
            for term in term_vector:
                topics_lst = list(set(topics_lst))
                token_topic_df.loc[term, topics_lst] += 1  # Update all relevant topics for the term
        

        if not multi_label:
            # Filter terms for each topic
            for topic in range(self.num_topics):
                topic_counts = token_topic_df[topic]
                is_only_in_topic = (topic_counts > 0) & (token_topic_df.drop(columns=topic).eq(0).all(axis=1))

                # Mark relevant terms
                token_topic_df['mark'] = np.where(is_only_in_topic, 1, 0)
                # Retain the top 100 terms for this topic
                topic_terms_to_zero = token_topic_df.sort_values(by=['mark', topic], ascending=[False, False]).iloc[100:].index
                token_topic_df.loc[topic_terms_to_zero, topic] = 0
            token_topic_df.drop(columns='mark', inplace=True)
        
        return token_topic_df

    def __get_term_topic_weights(self, vectorized_df, token_topic_df):
        
        # train term-topic weights
        token_topic_weight = pd.DataFrame(None, columns=range(self.num_topics), index=vectorized_df.index)

        for topic in list(token_topic_df.columns):
            token_topic_weight[topic] = vectorized_df.apply(lambda lst: [token_topic_df.loc[term][topic] for term in lst])
        
        return token_topic_weight
    
    def __fit_lda(train_bow, val_bow, test_bow, num_topics):
        """
        Fit LDA on training bow data and infer topic distributions for train & val.
        texts_*_bow: document-term matrix (counts) for LDA fitting/inference.
        Returns: theta_train (n_train, num_topics), theta_val (n_val, num_topics), lda_model.
        """
        lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        theta_train = lda_model.fit_transform(train_bow)  # shape (n_train, num_topics)
        theta_val = lda_model.transform(val_bow)      # shape (n_val, num_topics)
        theta_test = lda_model.transform(test_bow)      # shape (n_val, num_topics)
        # Note: transform returns distributions (rows sum to 1).
        return lda_model, theta_train, theta_val, theta_test

    def generateFeatures(self):
        # TODO trimmed documents are in form of list. We need to join the tokens and form the texts again.

        # Vectorization of the train documents 
        vectorization_model = TextVectorizationModel(apply_cosine_similarity_reduction=False)
        self.trainDocs['vectorized'] = vectorization_model.fit(self.trainDocs['trimmed'])
        self.valDocs['vectorized'] = vectorization_model.transform(self.valDocs['trimmed'])
        self.testDocs['vectorized'] = vectorization_model.transform(self.testDocs['trimmed'])
        
        self.vocab_size = vectorization_model.get_vocab_size()

        # pad vectorized documents
        self.trainDocs.loc[:,'vectorized_padded'] = self.trainDocs['vectorized'].apply(lambda lst: cus.padding(lst, self.max_doc_length))
        self.valDocs.loc[:, 'vectorized_padded'] = self.valDocs['vectorized'].apply(lambda lst: cus.padding(lst, self.max_doc_length))
        self.testDocs.loc[:, 'vectorized_padded'] = self.testDocs['vectorized'].apply(lambda lst: cus.padding(lst, self.max_doc_length))

        # calculation of the tf score for train, validation and test data
        train_tf = pd.DataFrame(vectorization_model.tf_matrix.toarray(), columns=range(vectorization_model.get_vocab_size()), index=self.trainDocs.index)
        self.trainDocs['tf'] = pd.DataFrame(train_tf.apply(cus.create_list, axis=1))

        val_joined_tokens = self.valDocs['trimmed'].apply(lambda txt: " ".join(txt))
        val_tf_matrix = vectorization_model.count_vectorizer.transform(val_joined_tokens)
        val_tf = pd.DataFrame(val_tf_matrix.toarray(), columns=range(vectorization_model.get_vocab_size()), index=self.valDocs.index)
        self.valDocs['tf'] = pd.DataFrame(val_tf.apply(cus.create_list, axis=1))

        test_joined_tokens = self.testDocs['trimmed'].apply(lambda txt: " ".join(txt))
        test_tf_matrix = vectorization_model.count_vectorizer.transform(test_joined_tokens)
        test_tf = pd.DataFrame(test_tf_matrix.toarray(), columns=range(vectorization_model.get_vocab_size()), index=self.testDocs.index)
        self.testDocs['tf'] = pd.DataFrame(test_tf.apply(cus.create_list, axis=1))

        # calculation of tf-idf score for train, validation, and test sets
        train_tfidf = pd.DataFrame(vectorization_model.tfidf_matrix.toarray(), columns=range(vectorization_model.get_vocab_size()), index=self.trainDocs.index)
        self.trainDocs['tfidf'] = pd.DataFrame(train_tfidf.apply(cus.create_list, axis=1))

        val_tfidf_matrix = vectorization_model.tfidf_vectorizer.transform(val_joined_tokens)
        val_tfidf = pd.DataFrame(val_tfidf_matrix.toarray(), columns=range(vectorization_model.get_vocab_size()), index=self.valDocs.index)
        self.valDocs['tfidf'] = pd.DataFrame(val_tfidf.apply(cus.create_list, axis=1))

        test_tfidf_matrix = vectorization_model.tfidf_vectorizer.transform(test_joined_tokens)
        test_tfidf = pd.DataFrame(test_tfidf_matrix.toarray(), columns=range(vectorization_model.get_vocab_size()), index=self.testDocs.index)
        self.testDocs['tfidf'] = pd.DataFrame(test_tfidf.apply(cus.create_list, axis=1))

        lda_model, lda_theta_train, lda_theta_val, lda_theta_tost = self.__fit_lda(self.trainDocs['tf'].to_list(),
                                                                                   self.valDocs['tf'].to_list(),
                                                                                   self.testDocs['tf'].to_list(),
                                                                                   self.num_topics)
        
        self.trainDocs['lda'] = (pd.DataFrame(lda_theta_train, index=self.trainDocs.index)).apply(cus.create_list, axis=1)
        self.valDocs['lda'] = (pd.DataFrame(lda_theta_val, index=self.valDocs.index)).apply(cus.create_list, axis=1)
        self.testDocs['lda'] = (pd.DataFrame(lda_theta_tost, index=self.testDocs.index)).apply(cus.create_list, axis=1)

        # calculate POS tags of vectorized documents in train, validation and test dataset
        self.trainDocs.loc[:, 'pos_tag'] = self.trainDocs['preprocess'].apply(lambda lst: cus.pos_tagger(lst))
        self.valDocs.loc[:, 'pos_tag'] = self.valDocs['preprocess'].apply(lambda lst: cus.pos_tagger(lst))
        self.testDocs.loc[:, 'pos_tag'] = self.testDocs['preprocess'].apply(lambda lst: cus.pos_tagger(lst))

        self.trainDocs.loc[:, 'pos_padded'] = self.trainDocs['pos_tag'].apply(lambda lst: cus.padding(lst, self.max_doc_length))
        self.valDocs.loc[:, 'pos_padded'] = self.valDocs['pos_tag'].apply(lambda lst: cus.padding(lst, self.max_doc_length))
        self.testDocs.loc[:, 'pos_padded'] = self.testDocs['pos_tag'].apply(lambda lst: cus.padding(lst, self.max_doc_length))

        token_topic_df = self.__get_term_topic_df(self.trainDocs['vectorized'], self.trainTopics, vectorization_model.get_vocab_size(), multi_label=False)
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

        self.selected_tuple_count = len(selected_tuples_lst)

        self.trainDocs.loc[:, 'tuple_2'] = train_doc_tuple_df[selected_tuples_lst].apply(cus.create_list, axis=1)
        self.valDocs.loc[:, 'tuple_2'] = val_doc_tuple_df[selected_tuples_lst].apply(cus.create_list, axis=1)
        self.testDocs.loc[:, 'tuple_2'] = test_doc_tuple_df[selected_tuples_lst].apply(cus.create_list, axis=1)

        return vectorization_model, lda_model, self.trainDocs, self.valDocs, self.testDocs