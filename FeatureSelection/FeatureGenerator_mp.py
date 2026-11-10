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

# [UPDATE - SPEED] Imported libraries for CPU parallelization and sparse matrix operations
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import scipy.sparse as sp

# Initialize tqdm for pandas
tqdm.pandas(desc="Processing Rows")

# [UPDATE - SPEED] Moved helper functions outside the class to ensure they can be pickled 
# by Python's multiprocessing backend. Nested functions or lambdas will throw PicklingErrors.
def _apply_padding(args):
    """Unpacks arguments for parallel padding application."""
    lst, max_len = args
    return cus.padding(lst, max_len)

def _apply_pos_tagger(lst):
    """Wrapper for parallel POS tagging."""
    return cus.pos_tagger(lst)


class FeatureGenerator():
    def __init__(self, max_doc_length, num_topics, stop_words_lang=None, max_df=1, min_df=1, tuple_fraction_to_keep=0.1, apply_cosine_similarity_reduction=False):
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
        self.stop_words_lang = stop_words_lang
        self.max_df=max_df
        self.min_df=min_df
        self.tuple_fraction_to_keep = tuple_fraction_to_keep
    
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
        
        # [UPDATE - SPEED] Determine optimal CPU cores for multiprocessing tasks
        num_cores = max(1, mp.cpu_count() - 1)

        # Vectorization of the train documents
        print("Vectorizing documents...") 
        vectorization_model = TextVectorizationModel(apply_cosine_similarity_reduction=False, 
                                                     stop_words_lang = self.stop_words_lang ,
                                                     max_df=self.max_df, 
                                                     min_df=self.min_df
                                                     )

        self.trainDocs['vectorized'] = vectorization_model.fit(self.trainDocs['trimmed'])
        self.valDocs['vectorized'] = vectorization_model.transform(self.valDocs['trimmed'])
        self.testDocs['vectorized'] = vectorization_model.transform(self.testDocs['trimmed'])
        
        self.vocab_size = vectorization_model.get_vocab_size()

        print("Passing unpadded vectors for dynamic batching...")
        print("Padding vectors...")
        
        # [UPDATE - SPEED] Replaced Pandas `.apply()` with ProcessPoolExecutor. 
        # Pandas apply inherently acts as a single-threaded for-loop. Dispatching this to multiple CPU threads 
        # handles large data arrays orders of magnitude faster. Chunking prevents memory overload.
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            train_pad_args = [(lst, self.max_doc_length) for lst in self.trainDocs['vectorized']]
            val_pad_args = [(lst, self.max_doc_length) for lst in self.valDocs['vectorized']]
            test_pad_args = [(lst, self.max_doc_length) for lst in self.testDocs['vectorized']]
            
            self.trainDocs.loc[:,'vectorized_padded'] = list(executor.map(_apply_padding, train_pad_args, chunksize=self.max_doc_length))
            self.valDocs.loc[:, 'vectorized_padded'] = list(executor.map(_apply_padding, val_pad_args, chunksize=self.max_doc_length))
            self.testDocs.loc[:, 'vectorized_padded'] = list(executor.map(_apply_padding, test_pad_args, chunksize=self.max_doc_length))

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

        print("Fitting LDA Model...")
        lda_model, lda_theta_train, lda_theta_val, lda_theta_tost = self.__fit_lda(self.trainDocs['tf'].to_list(),
                                                                                   self.valDocs['tf'].to_list(),
                                                                                   self.testDocs['tf'].to_list(),
                                                                                   self.num_topics)
        
        self.trainDocs['lda'] = lda_theta_train.tolist()
        self.valDocs['lda'] = lda_theta_val.tolist()
        self.testDocs['lda'] = lda_theta_tost.tolist()

        # calculate POS tags of vectorized documents in train, validation and test dataset
        print("Applying POS Tags...")
        
        # [UPDATE - SPEED] Deployed ProcessPoolExecutor for POS Tagging.
        # NLP models (like POS taggers) are extremely CPU intensive. Sending the text corpus 
        # across available cores cuts processing time linearly by the core count.
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            self.trainDocs.loc[:, 'pos_tag'] = list(executor.map(_apply_pos_tagger, self.trainDocs['preprocess'], chunksize=self.max_doc_length))
            self.valDocs.loc[:, 'pos_tag'] = list(executor.map(_apply_pos_tagger, self.valDocs['preprocess'], chunksize=self.max_doc_length))
            self.testDocs.loc[:, 'pos_tag'] = list(executor.map(_apply_pos_tagger, self.testDocs['preprocess'], chunksize=self.max_doc_length))

            # Pad the resulting POS tags using the same parallel executor
            train_pos_pad_args = [(lst, self.max_doc_length) for lst in self.trainDocs['pos_tag']]
            val_pos_pad_args = [(lst, self.max_doc_length) for lst in self.valDocs['pos_tag']]
            test_pos_pad_args = [(lst, self.max_doc_length) for lst in self.testDocs['pos_tag']]
            
            self.trainDocs.loc[:, 'pos_padded'] = list(executor.map(_apply_padding, train_pos_pad_args, chunksize=self.max_doc_length))
            self.valDocs.loc[:, 'pos_padded'] = list(executor.map(_apply_padding, val_pos_pad_args, chunksize=self.max_doc_length))
            self.testDocs.loc[:, 'pos_padded'] = list(executor.map(_apply_padding, test_pos_pad_args, chunksize=self.max_doc_length))


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

        # Prepare 1D labels for scikit-learn's mutual_info_classif
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
        
        # [UPDATE - SPEED] Converted binary presence matrix to a scipy sparse CSR matrix. 
        # For large NLP datasets, this condenses memory requirements significantly and allows 
        # sparse linear algebra optimizations during the dot product multiplication.
        doc_presence_sparse = sp.csr_matrix((train_doc_tuple_df[selected_tuples_lst].values > 0).astype(np.int32))
        total_docs = doc_presence_sparse.shape[0]
        
        # Sum down the columns to find out how many documents contain each tuple
        tuple_doc_counts = np.array(doc_presence_sparse.sum(axis=0)).flatten()
        
        # [UPDATE - SPEED] Fast sparse matrix multiplication for co-occurrences.
        # Transpose (Tuples x Docs) multiplied by (Docs x Tuples) = (Tuples x Tuples)
        co_occurrence_matrix = doc_presence_sparse.T.dot(doc_presence_sparse).toarray()

        A = np.zeros((self.selected_tuple_count, self.selected_tuple_count), dtype=np.float32)
        
        # [UPDATE - SPEED] Replaced the deeply nested O(N^2) loops with an Inverted Index lookup.
        # Iterating through every possible tuple combination to check if they share a word scales terribly. 
        # By building a dictionary mapping words to their parent tuple indices, we dynamically extract 
        # only the exact index combinations that satisfy the structural condition, bypassing millions of dead loops.
        word_to_tuples = {}
        for idx, t in enumerate(selected_tuples_lst):
            for w in t:
                if w not in word_to_tuples:
                    word_to_tuples[w] = []
                word_to_tuples[w].append(idx)
        
        # Create a unique set of edges representing tuple pairs that share at least one word
        shared_word_pairs = set()
        for w, indices in word_to_tuples.items():
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    # Order indices to avoid duplicate directed edges
                    idx1, idx2 = indices[i], indices[j]
                    if idx1 > idx2:
                        idx1, idx2 = idx2, idx1
                    shared_word_pairs.add((idx1, idx2))
        
        # [UPDATE - SPEED] Pre-calculate the independent base probabilities outside the loop.
        # Vectorized division replaces repetitive scalar math inside the iterator.
        p_t_array = tuple_doc_counts / total_docs

        # Iterate ONLY over the valid structural connections
        for i, j in shared_word_pairs:
            co_occur_count = co_occurrence_matrix[i, j]
            
            if co_occur_count > 0:
                p_t1 = p_t_array[i]
                p_t2 = p_t_array[j]
                p_pair = co_occur_count / total_docs
                
                # Pointwise Mutual Information (PMI) Calculation
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