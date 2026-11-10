import sys
sys.path.insert(1, '/Users/Ali/Documents')

from TextVectorizationModel import TextVectorizationModel
from TokenPairVectorizer import TokenPairVectorizer
from TupleFeatureReduction import TupleFeatureReduction
from TupleFeatureReductionMutual import TupleFeatureReductionMutual
from NameEntityRecognitionFeature import NameEntityRecognitionFeature
from SequenceEmbeddingsFeature import SequenceEmbeddingsFeature
import CustomFuncLib as cus
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from sklearn.svm import LinearSVR
from sklearn.decomposition import LatentDirichletAllocation

from xgboost import XGBClassifier

# Initialize tqdm for pandas
tqdm.pandas(desc="Processing Rows")

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
        self.global_adjacency_matrix = None
        self.raw_text_col = None
    
    def setDataset(self, train, trainTopics, validation, test):
        self.trainDocs = train.copy()
        self.trainTopics = trainTopics.copy()
        self.valDocs = validation.copy()
        self.testDocs = test.copy()

    def setNERColumn(self, ner_column):
        self.raw_text_col = ner_column

    def __get_term_topic_df(self, vectorized_df, topics_df, terms_lst, multi_label=False):
        # generate the Term-Topic Dictionary
        token_topic_matrix = np.zeros((terms_lst, self.num_topics), dtype=int)

        for index, topics_lst in topics_df.items():
            term_vector = vectorized_df.loc[index]
            if isinstance(topics_lst, (int, float, np.integer, np.floating)):
                unique_topics = [int(topics_lst)]
            else:
                unique_topics = list(set(topics_lst))

            for t in unique_topics:
                np.add.at(token_topic_matrix[:, t], term_vector, 1)

        token_topic_df = pd.DataFrame(token_topic_matrix, index=range(terms_lst), columns=range(self.num_topics))

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
        token_topic_matrix = token_topic_df.values
        result = {}
        for topic in range(self.num_topics):
            topic_weights = token_topic_matrix[:, topic]
            result[topic] = [list(topic_weights[lst]) for lst in vectorized_df]
        
        return pd.DataFrame(result, index=vectorized_df.index)
    
    def __get_weights_for_docs(self, token_topic_matrix, vectorized_docs, max_len):
        term_topic_weights = []
        num_topics = token_topic_matrix.shape[1]
        
        for lst in vectorized_docs:
            valid_terms = [term for term in lst if term < token_topic_matrix.shape[0]]
            
            # 1. Truncate if the document exceeds the maximum length
            valid_terms = valid_terms[:max_len]
            
            if not valid_terms:
                # If empty, return a flattened list of zeros of the exact required size
                term_topic_weights.append(np.zeros(num_topics * max_len).tolist())
                continue
                
            # 2. Fetch the weights using fast NumPy indexing. 
            # Shape will be (len(valid_terms), num_topics)
            weights = token_topic_matrix[valid_terms, :] 
            
            # 3. Pad the sequence dimension to ensure uniform shape (max_len, num_topics)
            pad_amount = max_len - len(valid_terms)
            if pad_amount > 0:
                weights = np.pad(weights, ((0, pad_amount), (0, 0)), mode='constant', constant_values=0)
                
            # 4. Transpose and flatten: shape becomes a 1D array of length (num_topics * max_len)
            # This perfectly mimics the structural output of your older `cus.concatenate_arrays` logic
            flat_weights = weights.T.flatten().tolist()
            term_topic_weights.append(flat_weights)
            
        return term_topic_weights
    
    def __fit_lda(self, train_bow, val_bow, test_bow, num_topics):
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
    
    def generate_summary_features(self, df):
        """
        Processes the 'summary' column to generate new feature arrays.
        """
        # Example: Generate contextual embeddings specifically for the summary
        summary_embeddings = self._generate_contextual_embeddings(df['summary'].tolist())
        df['summary_embedd'] = list(summary_embeddings)
        return df

    def generateFeatures(self):
        # TODO trimmed documents are in form of list. We need to join the tokens and form the texts again.

        # Vectorization of the train documents
        print("Vectorizing documents...") 
        vectorization_model = TextVectorizationModel(apply_cosine_similarity_reduction=False)
        self.trainDocs['vectorized'] = vectorization_model.fit(self.trainDocs['trimmed'])
        self.valDocs['vectorized'] = vectorization_model.transform(self.valDocs['trimmed'])
        self.testDocs['vectorized'] = vectorization_model.transform(self.testDocs['trimmed'])
        
        self.vocab_size = vectorization_model.get_vocab_size()

        # pad vectorized documents
        # Leave sequences unpadded for the DynamicBatchGenerator
        print("Passing unpadded vectors for dynamic batching...")
        # pad vectorized documents
        print("Padding vectors...")
        self.trainDocs.loc[:,'vectorized_padded'] = self.trainDocs['vectorized'].apply(lambda lst: cus.padding(lst, self.max_doc_length))
        self.valDocs.loc[:, 'vectorized_padded'] = self.valDocs['vectorized'].apply(lambda lst: cus.padding(lst, self.max_doc_length))
        self.testDocs.loc[:, 'vectorized_padded'] = self.testDocs['vectorized'].apply(lambda lst: cus.padding(lst, self.max_doc_length))

        # calculation of the tf score for train, validation and test data
        self.trainDocs['tf'] = vectorization_model.tf_matrix.toarray().tolist()

        val_joined_tokens = self.valDocs['trimmed'].apply(lambda txt: " ".join(txt))
        val_tf_matrix = vectorization_model.count_vectorizer.transform(val_joined_tokens)
        self.valDocs['tf'] = val_tf_matrix.toarray().tolist()

        test_joined_tokens = self.testDocs['trimmed'].apply(lambda txt: " ".join(txt))
        test_tf_matrix = vectorization_model.count_vectorizer.transform(test_joined_tokens)
        self.testDocs['tf'] = test_tf_matrix.toarray().tolist()

        # calculation of tf-idf score for train, validation, and test sets
        self.trainDocs['tfidf'] = vectorization_model.tfidf_matrix.toarray().tolist()

        test_tfidf_matrix = vectorization_model.tfidf_vectorizer.transform(test_joined_tokens)
        self.testDocs['tfidf'] = test_tfidf_matrix.toarray().tolist()

        val_tfidf_matrix = vectorization_model.tfidf_vectorizer.transform(val_joined_tokens)
        self.valDocs['tfidf'] = val_tfidf_matrix.toarray().tolist()

        test_tfidf_matrix = vectorization_model.tfidf_vectorizer.transform(test_joined_tokens)
        self.testDocs['tfidf'] = test_tfidf_matrix.toarray().tolist()

        print("Fitting LDA Model...")
        lda_model, lda_theta_train, lda_theta_val, lda_theta_tost = self.__fit_lda(self.trainDocs['tf'].to_list(),
                                                                                   self.valDocs['tf'].to_list(),
                                                                                   self.testDocs['tf'].to_list(),
                                                                                   self.num_topics)
        
        self.trainDocs['lda'] = lda_theta_train.tolist()
        self.valDocs['lda'] = lda_theta_val.tolist()
        self.testDocs['lda'] = lda_theta_tost.tolist()

        # calculate POS tags of vectorized documents in train, validation and test dataset
        self.trainDocs.loc[:, 'pos_tag'] = self.trainDocs['preprocess'].apply(lambda lst: cus.pos_tagger(lst))
        self.valDocs.loc[:, 'pos_tag'] = self.valDocs['preprocess'].apply(lambda lst: cus.pos_tagger(lst))
        self.testDocs.loc[:, 'pos_tag'] = self.testDocs['preprocess'].apply(lambda lst: cus.pos_tagger(lst))

        self.trainDocs.loc[:, 'pos_padded'] = self.trainDocs['pos_tag'].apply(lambda lst: cus.padding(lst, self.max_doc_length))
        self.valDocs.loc[:, 'pos_padded'] = self.valDocs['pos_tag'].apply(lambda lst: cus.padding(lst, self.max_doc_length))
        self.testDocs.loc[:, 'pos_padded'] = self.testDocs['pos_tag'].apply(lambda lst: cus.padding(lst, self.max_doc_length))

        token_topic_df = self.__get_term_topic_df(self.trainDocs['vectorized'], self.trainTopics, vectorization_model.get_vocab_size(), multi_label=False)
        
        # Extract the raw matrix for fast NumPy indexing
        token_topic_matrix = token_topic_df.values
        
        # Fast generation of heavily padded, uniformly sized arrays
        self.trainDocs.loc[:, 'term_topic_weight'] = self.__get_weights_for_docs(token_topic_matrix, self.trainDocs['vectorized'], self.max_doc_length)
        self.valDocs.loc[:, 'term_topic_weight'] = self.__get_weights_for_docs(token_topic_matrix, self.valDocs['vectorized'], self.max_doc_length)
        self.testDocs.loc[:, 'term_topic_weight'] = self.__get_weights_for_docs(token_topic_matrix, self.testDocs['vectorized'], self.max_doc_length)

        pair_vectorizer = TokenPairVectorizer(tuple_size=2)
        # Fit the model on the training data
        train_doc_tuple_df = pair_vectorizer.fit(self.trainDocs['vectorized'])

        # Apply the model to the validation or test data
        val_doc_tuple_df = pair_vectorizer.transform(self.valDocs['vectorized'])
        test_doc_tuple_df = pair_vectorizer.transform(self.testDocs['vectorized'])


        # svc = LinearSVR(max_iter=5000)
        # # xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

        # train_single_label_df = self.trainTopics.apply(lambda lst: lst[0])

        # tuple_selector = TupleFeatureReduction(estimator=svc, num_features_to_select=0.1, num_steps=100)
        # # Train (fit) on the training data
        # _ , selected_tuples_lst = tuple_selector.fit(train_doc_tuple_df, train_single_label_df)

        # Prepare 1D labels for scikit-learn's mutual_info_classif (just like you did for SVR)
        train_single_label_df = self.trainTopics.apply(lambda lst: lst[0])

        # The vocabulary is simply the column names of your tuple dataframe
        tuple_vocabulary_list = list(train_doc_tuple_df.columns)

        tuple_selector = TupleFeatureReductionMutual(keep_fraction=0.1)
        tuple_selector.fit_transform(train_doc_tuple_df, train_single_label_df, tuple_vocabulary_list)
        selected_tuples_lst = tuple_selector.get_selected_vocabulary()

        self.selected_tuple_count = len(selected_tuples_lst)

        # Named Entity Density
        print("Generating NER Density...")
        ner = NameEntityRecognitionFeature(language_model = 'en_core_web_sm')

        # It will automatically find the top 6 labels based on the training corpus context
        train_ner_vectors, dynamic_labels = ner.generate_ner_density(self.trainDocs, text_column=self.raw_text_col, track_labels=None, top_k=6)
        self.trainDocs.loc[:, 'ner_density'] = train_ner_vectors

        val_ner_vectors, _ = ner.generate_ner_density(self.valDocs, text_column=self.raw_text_col, track_labels=dynamic_labels)
        self.valDocs.loc[:, 'ner_density'] = val_ner_vectors

        test_ner_vectors, _ = ner.generate_ner_density(self.testDocs, text_column=self.raw_text_col, track_labels=dynamic_labels)
        self.testDocs.loc[:, 'ner_density'] = test_ner_vectors

        # Contextual Sequence Embeddings (BERT)
        print("Generating Contextual Embeddings...")
        seq_embedd = SequenceEmbeddingsFeature(model_name='all-MiniLM-L6-v2')
        self.trainDocs.loc[:, 'contextual_embedd'] = seq_embedd.generate_contextual_embeddings(self.trainDocs, text_column=self.raw_text_col)
        self.valDocs.loc[:, 'contextual_embedd'] = seq_embedd.generate_contextual_embeddings(self.valDocs, text_column=self.raw_text_col)
        self.testDocs.loc[:, 'contextual_embedd'] = seq_embedd.generate_contextual_embeddings(self.testDocs, text_column=self.raw_text_col)

        ### NEW CODE STARTS HERE ###
        print("Constructing PMI-Weighted Graph Adjacency Matrix for tuples...")
        
        # 1. Mathematically calculate Co-occurrences across the corpus
        # Convert the counts to 1s and 0s (presence/absence in a document)
        doc_presence = (train_doc_tuple_df[selected_tuples_lst] > 0).astype(int)
        total_docs = len(doc_presence)
        
        # Sum down the columns to find out how many documents contain each tuple
        tuple_doc_counts = doc_presence.sum(axis=0).values 
        
        # Fast matrix multiplication to find how many times Pair A and Pair B appear in the same document
        # Transpose (Tuples x Docs) multiplied by (Docs x Tuples) = (Tuples x Tuples)
        co_occurrence_matrix = doc_presence.T.dot(doc_presence).values 

        # 2. Build the Weighted Adjacency Matrix
        A = np.zeros((self.selected_tuple_count, self.selected_tuple_count), dtype=np.float32)
        tuple_word_sets = [set(t) for t in selected_tuples_lst]
        
        for i in range(self.selected_tuple_count):
            for j in range(i + 1, self.selected_tuple_count):
                
                # Structural Condition: Do the two tuples share a word?
                if not tuple_word_sets[i].isdisjoint(tuple_word_sets[j]):
                    
                    # Fetch how many times they appeared together
                    co_occur_count = co_occurrence_matrix[i, j]
                    
                    if co_occur_count > 0:
                        # Pointwise Mutual Information (PMI) Calculation
                        p_t1 = tuple_doc_counts[i] / total_docs
                        p_t2 = tuple_doc_counts[j] / total_docs
                        p_pair = co_occur_count / total_docs
                        
                        pmi = np.log(p_pair / (p_t1 * p_t2))
                        
                        # Positive PMI (PPMI): Drop negative relations to remove noise
                        weight = max(pmi, 0.0)
                    else:
                        # Fallback small weight if they share a word but never appear in the same document
                        weight = 0.01 
                        
                    A[i, j] = weight
                    A[j, i] = weight
                    
        # Add Self-Loops (Identity Matrix) so a node passes information to itself
        np.fill_diagonal(A, 1.0)
        
        # 3. Normalize the Weighted Laplacian
        print("Normalizing the Graph Matrix for GCN...")
        
        # Calculate the Degree Matrix (sum of weights connected to each node)
        rowsum = A.sum(axis=1)
        
        # Calculate D^(-0.5)
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0 # Handle division by zero
        D_inv_sqrt = np.diag(d_inv_sqrt)
        
        # Symmetric Normalization: D^(-0.5) * A * D^(-0.5)
        A_normalized = D_inv_sqrt.dot(A).dot(D_inv_sqrt)

        self.global_adjacency_matrix = A_normalized
        print("Graph Construction Complete.")

        ### NEW CODE ENDS HERE ###
        self.trainDocs.loc[:, 'tuple_2'] = train_doc_tuple_df[selected_tuples_lst].values.tolist()
        self.valDocs.loc[:, 'tuple_2'] = val_doc_tuple_df[selected_tuples_lst].values.tolist()
        self.testDocs.loc[:, 'tuple_2'] = test_doc_tuple_df[selected_tuples_lst].values.tolist()

        return vectorization_model, lda_model, self.trainDocs, self.valDocs, self.testDocs