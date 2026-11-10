from keras.models import Model
from keras.layers import (Input, concatenate, Dense, Flatten, 
                          Dropout, Conv1D, AveragePooling1D, 
                          Embedding, BatchNormalization, Lambda, Multiply)

from tensorflow.keras.layers import (Input, concatenate, Dense, Flatten, 
                                     Dropout, Embedding, BatchNormalization, 
                                     Lambda, Multiply, MultiHeadAttention, 
                                     LayerNormalization, GlobalAveragePooling1D, GlobalMaxPooling1D, Layer, Add) # Added MultiHeadAttention, LayerNormalization, and GlobalAveragePooling1D

from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow as tf
from keras.optimizers import Adam
from sklearn.metrics import f1_score, accuracy_score, classification_report
import numpy as np
import pandas as pd
import random
import gc

import numpy as np

class SemanticEmbeddingInjection(tf.keras.layers.Layer):
    def __init__(self, feature_dim, embed_dim, **kwargs):
        super(SemanticEmbeddingInjection, self).__init__(**kwargs)
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim

    def build(self, input_shape):
        # Create a trainable semantic vector for every single tuple node
        self.node_embeddings = self.add_weight(
            shape=(self.feature_dim, self.embed_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='tuple_semantic_embeddings'
        )
        super(SemanticEmbeddingInjection, self).build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch_size, feature_dim, 1)
        # Multiply node frequencies by their semantic vectors natively
        return inputs * self.node_embeddings

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.feature_dim, self.embed_dim)

class PositionalEmbedding(Layer):
    def __init__(self, max_len, embed_dim, **kwargs):
        super().__init__(**kwargs)
        # CRITICAL FIX: Tell Keras this layer accepts and propagates masks
        self.supports_masking = True
        
        # We only need the position embeddings here; tokens are already embedded
        self.position_embeddings = tf.keras.layers.Embedding(
            input_dim=max_len, output_dim=embed_dim
        )

    def call(self, inputs):
        # inputs shape is (batch_size, seq_len, embed_dim)
        # Extract the dynamic sequence length of the current batch
        seq_len = tf.shape(inputs)[1] 
        
        # Generate position indices [0, 1, 2, ..., seq_len-1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        
        # Embed the positions
        embedded_positions = self.position_embeddings(positions)
        
        # Add the positional encodings directly to the incoming token embeddings
        return inputs + embedded_positions

    # CRITICAL FIX: Explicitly pass the input mask downstream
    def compute_mask(self, inputs, mask=None):
        return mask

class GraphConvolution(tf.keras.layers.Layer):
    def __init__(self, units, adjacency_matrix, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.units = units
        
        # Calculate the Normalized Graph Laplacian (Standard GCN math)
        A_hat = adjacency_matrix + np.eye(adjacency_matrix.shape[0]) # Add self-loops
        D_hat = np.array(np.sum(A_hat, axis=1)).flatten() # Degree matrix
        D_half_inv = np.diag(np.power(D_hat, -0.5))
        A_norm = np.dot(D_half_inv, np.dot(A_hat, D_half_inv))
        
        # Store as a non-trainable constant tensor
        self.A_norm = tf.constant(A_norm, dtype=tf.float32)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros', trainable=True)
        super(GraphConvolution, self).build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch_size, num_nodes, feature_dim)
        xw = tf.matmul(inputs, self.W)

        # Multiply graph structure by node features using Einstein summation
        out = tf.einsum('ij,bjk->bik', self.A_norm, xw)
        return tf.nn.relu(out + self.b)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.units)
    
class DynamicBatchGenerator(tf.keras.utils.Sequence):
    # Added 'settings' parameter
    def __init__(self, df, features, labels, batch_size, settings, lda_gating=False, full_doc_col='full_doc_tfidf', **kwargs):
        super().__init__(**kwargs) 
        
        self.df = df
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        self.settings = settings # Store settings to check architecture routes
        self.lda_gating = lda_gating
        self.full_doc_col = full_doc_col

    def __len__(self):
        return int(np.ceil(len(self.df) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_df = self.df.iloc[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size : (idx + 1) * self.batch_size]

        X_batch = []
        
        for feature in self.features:
            feature_data = batch_df[feature].values

            if isinstance(feature_data[0], (list, np.ndarray)) and len(np.shape(feature_data[0])) == 1:
                # Check if the feature goes to a Dense layer requiring fixed padding
                model_type = self.settings.loc[feature].get('model_type', 'transformer')
                
                if model_type == 'dense':
                    # Force global padding for Dense architectures
                    required_len = int(self.settings.loc[feature]['feature_dim'])
                    padded_feature = pad_sequences(feature_data, padding='post', maxlen=required_len, value=0)
                else:
                    # Allow dynamic padding for Transformers
                    padded_feature = pad_sequences(feature_data, padding='post', value=0)
                    
                X_batch.append(padded_feature)
            else:
                X_batch.append(np.stack(feature_data))

        if self.lda_gating:
            full_tfidf_data = batch_df[self.full_doc_col].values
            X_batch.append(np.stack(full_tfidf_data))

        return tuple(X_batch), np.array(batch_labels)
    
class CustomFeatureSelection:
    def __init__(self, train_df, train_labels, test_df, test_labels, num_classes):
        self.train_df = train_df
        self.train_labels = train_labels
        self.test_df = test_df
        self.test_labels = test_labels
        
        self.num_classes = num_classes

        self.model = None
        
        self.vocab_size = None
        self.embedded_output_dim = None
        self.trainable = True
        self.embedding_weight = None

        self.lda_input = None
        self.lda_branch = None

        self.results = []  # To store results across iterations

        # self.EqualSizedFeatureSelection = self.EqualSizedFeatureSelection(self)
        self.UnEqualSizedFeatureSelection = self.UnEqualSizedFeatureSelection(self)

    def setVocabSize(self, value):
        self.vocab_size = value

    def embedding_setup(self, vocab_size, embedded_output_dim, pretrained=None):
        self.vocab_size = vocab_size
        self.embedded_output_dim = embedded_output_dim
        if pretrained == 'GloVe':
            
            self.trainable = False
        elif pretrained == 'Word2Vec':

            self.trainable = False

            # Function to evaluate model performance on validation data
    
    def evaluate_model(self, validation_generator):
        # Bypass TF Dataset strict shape checking by predicting batch-by-batch natively
        predictions = []
        
        for i in range(len(validation_generator)):
            X_batch, _ = validation_generator[i]
            batch_preds = self.model.predict_on_batch(X_batch)
            predictions.append(np.array(batch_preds))
            
        # Stack all batch predictions into a single continuous array
        predictions = np.vstack(predictions)
        predicted_labels = np.argmax(predictions, axis=1)
        
        # Check if test_labels are one-hot encoded, convert predicted_labels if necessary
        if self.test_labels.ndim > 1:  # if test_labels is one-hot
            true_labels = np.argmax(self.test_labels, axis=1)
        else:
            true_labels = self.test_labels
        
        # Generate the classification report with per-class and overall metrics
        report = classification_report(true_labels, predicted_labels, output_dict=True, zero_division=0)
        
        # Extract the required metrics
        results = {
            "accuracy": accuracy_score(true_labels, predicted_labels),
            "micro": f1_score(true_labels, predicted_labels, average="micro"),
            "macro": f1_score(true_labels, predicted_labels, average="macro"),
            "per_class_metrics": {f"class_{cls}": metrics for cls, metrics in report.items() if cls.isdigit()},
        }
        return results
    
    def print_performance(self, features, score):
        # Prepare the metrics for tabular format
        row = {'Features': str(features),  # Convert features to string for display
            'Accuracy': score['accuracy'],
            'Micro F1': score['micro'],
            'Macro F1': score['macro']}
        
        # Add per-class metrics as additional columns
        for cls, metrics in score['per_class_metrics'].items():
            row[f"Class {cls} Precision"] = metrics['precision']
            row[f"Class {cls} Recall"] = metrics['recall']
            row[f"Class {cls} F1-score"] = metrics['f1-score']
        
        # Append the row to results
        self.results.append(row)
        
        # Convert to DataFrame and display
        results_df = pd.DataFrame(self.results)
        print(results_df.to_string(index=False))  # Print the DataFrame as a table

    def get_neural_topic_stream(self, tfidf_input, num_topics, dense_units, dropout_rate):

            # First layer: Compress the massive TF-IDF vector into a dense representation
            x_topic = Dense(dense_units, activation='relu', name="ntm_encoder_1")(tfidf_input)
            x_topic = BatchNormalization(name="ntm_batch_1")(x_topic)
            x_topic = Dropout(dropout_rate, name="ntm_dropout_1")(x_topic)
            
            # Second layer: Extract higher-level semantic features
            x_topic = Dense(dense_units // 2, activation='relu', name="ntm_encoder_2")(x_topic)
            x_topic = BatchNormalization(name="ntm_batch_2")(x_topic)
            
            # Output layer: Generate the Topic Probability Distribution
            # Softmax ensures the outputs sum to 1.0, exactly like traditional LDA probabilities
            topic_distribution = Dense(num_topics, activation='softmax', name="ntm_topic_distribution")(x_topic)

            return topic_distribution

    class UnEqualSizedFeatureSelection:
        
        def __init__(self, global_instances):
            self.global_instances = global_instances
            
        def __customAdjustableModel(self, features, settings, dense_settings, lda_settings):
            # saves inputs and layers
            inputs = []
            layers_to_concatinate = []

            dense_settings = list(zip(dense_settings['dense'], dense_settings['dropout']))
            lda_gating = False
            if lda_settings is not None:
                lda_gating = lda_settings.get("lda_gating")

            # ==========================================================
            # Variable to hold the LDA tensor if the GA selects it
            # ==========================================================
            lda_tensor = None

            # loops over features
            for feature in features:
                # ==========================================================
                # NEW ROUTE: Bypass sequence architecture for Dense features
                # ==========================================================
                if feature in ['ner_density', 'contextual_embedd']:

                    setting = settings.loc[feature]

                    feature_dim = int(setting['feature_dim'])
                    ff_dim = int(setting.get('ff_dim', 128))
                    
                    input_layer = Input(shape=(feature_dim,), name=f"input_{feature}")
                    
                    # Pass through a standard feed-forward block to match the depth of other features
                    x = Dense(ff_dim, activation='relu')(input_layer)
                    x = BatchNormalization()(x)
                    x = Dropout(0.3)(x)
                    
                    layers_to_concatinate.append(x)
                    inputs.append(input_layer)
                    
                    continue # Skip the rest of the loop for this feature
                # ==========================================================

                # gets the settings of each feature
                setting = settings.loc[feature]

# beginning of the update                
                # gets the feature dimension
                feature_dim = int(setting['feature_dim'])
                model_type = setting.get('model_type', 'transformer') # Identify the architecture route

                if setting['embedding']:
                    input_layer = Input(shape=(None,), name=f"input_{feature}")

                    # 1. Token Embedding and ACTIVATE MASKING: mask_zero=True forces the network to ignore padded 0s
                    token_embeddings = Embedding(input_dim = self.global_instances.vocab_size,
                                                 output_dim = self.global_instances.embedded_output_dim,
                                                 weights = self.global_instances.embedding_weight,
                                                 trainable = self.global_instances.trainable,
                                                 mask_zero = True)(input_layer)
                    
                    if model_type == 'transformer':
                        # Only apply positional encoding if it is going to a transformer
                        # 2. Positional Embedding (Wrapped in layer to safely handle dynamic KerasTensors)
                        passing_layer = PositionalEmbedding(
                            max_len=5000, 
                            embed_dim=self.global_instances.embedded_output_dim
                        )(token_embeddings)
                    else:
                        passing_layer = token_embeddings

                else:
                    if model_type == 'transformer':
                        # 1. Accept dynamic sequence length (None,) instead of rigid feature_dim
                        input_layer = Input(shape=(None,), name=f"input_{feature}")
                        
                        # 2. Expand to 3D (batch, dynamic_steps, 1) so Attention matrix math works
                        passing_layer = Lambda(lambda x: tf.expand_dims(x, axis=-1), name=f"expand_{feature}")(input_layer)
                    else:
                        # Bag of Words strictly needs a rigid 2D shape (batch, vocab_size)
                        input_layer = Input(shape=(feature_dim,), name=f"input_{feature}")
                        passing_layer = input_layer

                inputs.append(input_layer)

                if setting['normalize']:
                    passing_layer = BatchNormalization()(passing_layer)

                # --- 2. ARCHITECTURE ROUTING ---
                ff_dim = int(setting.get('ff_dim', 128))

                if model_type == 'transformer':
                    # A. Transformer Branch (For Sequential Data)
                    num_blocks = int(setting.get('num_transformer_blocks', 1))
                    num_heads = int(setting.get('num_heads', 2))
                    
                    # Safe fallback for key_dim if embedded_output_dim is not set
                    embed_dim = self.global_instances.embedded_output_dim if hasattr(self.global_instances, 'embedded_output_dim') else 64
                    key_dim = int(setting.get('key_dim', embed_dim // num_heads))
                
                    for _ in range(num_blocks):
                        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(passing_layer, passing_layer)
                        attention_output = Dropout(0.3)(attention_output) # Increased dropout for regularization
                        out1 = LayerNormalization(epsilon=1e-6)(passing_layer + attention_output)
                        
                        ffn_output = Dense(ff_dim, activation='relu')(out1)
                        ffn_output = Dense(passing_layer.shape[-1])(ffn_output)
                        ffn_output = Dropout(0.3)(ffn_output) # Increased dropout for regularization
                        passing_layer = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

                    # Compress the sequence into a single vector
                    branch_output = GlobalAveragePooling1D()(passing_layer)
                    layers_to_concatinate.append(branch_output)
                
                elif model_type == 'dense':
                    # B. Dense Branch (For Bag-of-Words / Frequency Data)

                    # Solves OOM by avoiding NxN attention matrices on huge vocabularies
                    dense_out = Dense(ff_dim, activation='relu')(passing_layer)
                    dense_out = Dropout(0.4)(dense_out) # Heavy dropout to prevent memorization of specific word frequencies
                    dense_out = Dense(ff_dim // 2, activation='relu')(dense_out)
                    dense_out = Dropout(0.4)(dense_out)
                    
                    layers_to_concatinate.append(dense_out)

                elif model_type == 'graph':
                    # C. Graph Convolution Branch (For Co-occurrence Tuples)
                    A = setting['adjacency_matrix']
                    
                    # Reshape the flat tuple array (batch, N) into node features (batch, N, 1)
                    freq_x = tf.keras.layers.Reshape((feature_dim, 1))(passing_layer)
                    
                    # 2. Inject Semantic Meaning: Create Trainable Embeddings for the Tuple Nodes
                    # This gives each of the tuples a unique 128-dimensional vector
                    graph_x = SemanticEmbeddingInjection(feature_dim=feature_dim, embed_dim=128)(freq_x)

                    # 3. Pass through Message Passing
                    num_gcn_layers = int(setting.get('num_gcn_layers', 2))
                    
                    for _ in range(num_gcn_layers):
                        graph_x = GraphConvolution(ff_dim, A)(graph_x)
                        graph_x = Dropout(0.3)(graph_x)
                    
                    # 4. Extract the strongest topical signals
                    branch_output = GlobalMaxPooling1D()(graph_x)
                    layers_to_concatinate.append(branch_output)

            if lda_gating:
                full_tfidf_dim = lda_settings.get("full_tfidf_dim", self.global_instances.vocab_size)
                input_full_tfidf = Input(shape=(full_tfidf_dim,), name="input_full_doc_tfidf")
                
                inputs.append(input_full_tfidf)
                
                # num_topics is now extracted dynamically as a hyperparameter
                num_latent_topics = lda_settings.get("num_topics", 50)
                
                lda_tensor = self.global_instances.get_neural_topic_stream(
                    tfidf_input = input_full_tfidf, 
                    num_topics = num_latent_topics, 
                    dense_units = lda_settings.get("lda_dense", 128), 
                    dropout_rate = lda_settings.get("lda_dropout", 0.3)
                )
# end of update
            if lda_gating and lda_tensor is not None:
                print("\n--- LDA Gating Activated ---")
                # apply lda gating directly using the ALREADY REGISTERED lda_tensor
                gate_hidden = Dense(max(64, self.global_instances.num_classes), activation='relu', name="gate_hidden")(lda_tensor)
                
                # UPDATE 1: Change 'sigmoid' to 'tanh' and initialize bias to 0 
                # so the gate starts completely neutral (acting as a pure pass-through initially)
                gate_vector = Dense(len(layers_to_concatinate), activation='tanh', name="branch_gate", bias_initializer='zeros')(gate_hidden)

                gated_layers_to_concatinate = []

                for index, branch in enumerate(layers_to_concatinate):
                    # Extract gate scalar for branch i
                    gate_i = Lambda(lambda vector, idx=index: vector[:, idx:idx+1], name=f"gate_slice_{index}")(gate_vector)
                    
                    # UPDATE 2: Calculate the modulation (boost or penalty)
                    modulation = Multiply(name=f"gated_modulation_{index}")([branch, gate_i])
                    
                    # UPDATE 3: The Residual Connection - Add modulation back to the original unmodified branch
                    final_gated_branch = Add(name=f"residual_gate_{index}")([branch, modulation])
                    
                    gated_layers_to_concatinate.append(final_gated_branch)
                
                layers_to_concatinate = gated_layers_to_concatinate


            if len(layers_to_concatinate) > 1:
                merged = concatenate(layers_to_concatinate)
            else:
                merged = layers_to_concatinate[0]

            passing_layer = merged
            # Additional dense layers for further processing
            for dense_setting in dense_settings:
                passing_layer = Dense(dense_setting[0], activation='relu')(passing_layer)
                passing_layer = Dropout(0.1)(passing_layer)

            output_layer = Dense(self.global_instances.num_classes, activation='softmax')(passing_layer)

            # Compile the model
            self.global_instances.model = Model(inputs=inputs, outputs=output_layer)
            self.global_instances.model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
            
            # self.global_instances.model.summary()
            
        def forward_selection(self, features_settings, dense_settings, lda_settings = None, evaluation = 'accuracy', epochs=5, batch_size=32):
            if not isinstance(self.global_instances.train_df,  pd.DataFrame):
                raise ValueError("train type must be a pandas dataframe")
            
            if not isinstance(self.global_instances.test_df,  pd.DataFrame):
                raise ValueError("test type must be a pandas dataframe")
            
            if not isinstance(features_settings,  pd.DataFrame):
                raise ValueError("settings type must be a pandas dataframe, including filter_size, kernel_size, pool_size, embedding (boolean), feature_dim (dimension -> int)")
            
            if not isinstance(dense_settings,  pd.DataFrame):
                raise ValueError("settings type must be a pandas dataframe, including dense list and dropout list")
            
            if not (isinstance(lda_settings,  dict) or lda_settings is None):
                raise ValueError("settings type must be a pandas dataframe, including gating(boolean), dense_size, and dropout")

            # gets the list of features
            remained_features = self.global_instances.train_df.columns.to_list()
            best_score = 0.0
            selected_features = []

            while remained_features:
                best_feature = None
                for feature in remained_features:
                    current_features = selected_features + [feature]
                    
                    self.__trainModel(current_features,
                                      features_settings.loc[current_features],
                                      dense_settings,
                                      lda_settings,
                                      epochs,
                                      batch_size)
                    
                    validate_array = [np.stack(self.global_instances.test_df[feature]) for feature in current_features]

                    # Evaluate model on validation data
                    score = self.global_instances.evaluate_model(validate_array)

                    if score[evaluation] > best_score: 
                        best_score = score[evaluation]
                        best_feature = feature
                    
                    self.global_instances.print_performance(current_features, score)
                
                # If no improvement, stop the process
                if best_feature:
                    selected_features.append(best_feature)
                    remained_features.remove(best_feature)
                else:
                    break

            return selected_features, best_score
        
        def backward_selection(self, features_settings, dense_settings, evaluation = 'accuracy', epochs=5, batch_size=32):
            if not isinstance(self.global_instances.train_df,  pd.DataFrame):
                raise ValueError("train type must be a pandas dataframe")
            
            if not isinstance(self.global_instances.test_df,  pd.DataFrame):
                raise ValueError("test type must be a pandas dataframe")
            
            if not isinstance(features_settings,  pd.DataFrame):
                raise ValueError("settings type must be a pandas dataframe, including kernel_size, pool_size, embedding (boolean), feature_dim (dimension -> int)")
            
            if not isinstance(dense_settings,  pd.DataFrame):
                raise ValueError("settings type must be a pandas dataframe, including kernel_size, pool_size, embedding (boolean), feature_dim (dimension -> int)")

            selected_features = set(self.global_instances.train_df.columns)
            
            self.__trainModel(selected_features, 
                    features_settings.loc[list(selected_features)], 
                    dense_settings, 
                    epochs, 
                    batch_size)

            validate_array = [np.stack(self.global_instances.test_df[feature]) for feature in selected_features]

            # Evaluate model on validation data
            score = self.global_instances.evaluate_model(validate_array)

            self.global_instances.print_performance(selected_features, score)

            best_score = score[evaluation]
            best_combination = selected_features

            while len(selected_features) > 1:
                for feature in selected_features:
                    current_features = selected_features - {feature}
                    
                    self.__trainModel(current_features,
                                      features_settings.loc[list(current_features)],
                                      dense_settings,
                                      epochs,
                                      batch_size)
                    
                    validate_array = [np.stack(self.global_instances.test_df[feature]) for feature in current_features]
                                        
                    # Evaluate model on validation data
                    score = self.global_instances.evaluate_model(validate_array)

                    if score[evaluation] > best_score:
                        best_score = score[evaluation]
                        best_combination = current_features

                    self.global_instances.print_performance(current_features, score)
                
                # Update selected features with the best combination if it improved F-score
                if best_combination != selected_features:
                    selected_features = best_combination
                else:
                    break

            return selected_features, best_score
        
        def randomized_selection(self, features_settings, dense_settings, lda_settings = None, num_iterations=20, evaluation = 'accuracy', epochs=5, batch_size=32):
            if not isinstance(self.global_instances.train_df,  pd.DataFrame):
                raise ValueError("train type must be a pandas dataframe")
            
            if not isinstance(self.global_instances.test_df,  pd.DataFrame):
                raise ValueError("test type must be a pandas dataframe")
            
            if not isinstance(features_settings,  pd.DataFrame):
                raise ValueError("settings type must be a pandas dataframe, including kernel_size, pool_size, embedding (boolean), feature_dim (dimension -> int)")
            
            if not isinstance(dense_settings,  pd.DataFrame):
                raise ValueError("settings type must be a pandas dataframe, including kernel_size, pool_size, embedding (boolean), feature_dim (dimension -> int)")
        
            if not (isinstance(lda_settings,  dict) or lda_settings is None):
                raise ValueError("settings type must be a pandas dataframe, including gating(boolean), dense_size, and dropout")

            features_lst = self.global_instances.train_df.columns.to_list()
            waiting_features = set()
            iteration_count = 0

            # Select the best feature from the features_lst
            current_features, current_score = self.forward_selection(features_settings, dense_settings, evaluation=evaluation, epochs=epochs, batch_size=batch_size)
            
            while iteration_count < num_iterations and len(current_features) < len(features_lst):
                # Select a random feature from the remaining features_lst
                remaining_features = list(set(features_lst) - set(current_features) - waiting_features)

                if not remaining_features:
                    break
                
                random_feature = random.choice(remaining_features)
                current_features.append(random_feature)
                
                waiting_features.clear()  # clear waiting features for future chance of being added to the feature combination

                self.__trainModel(current_features,
                                    features_settings.loc[current_features],
                                    dense_settings,
                                    lda_settings,
                                    epochs,
                                    batch_size)
                
                validate_array = [np.stack(self.global_instances.test_df[feature]) for feature in current_features]

                # Evaluate model on validation data
                new_score = self.global_instances.evaluate_model(validate_array)

                self.global_instances.print_performance(current_features, new_score)

                if new_score[evaluation] > current_score:
                    current_score = new_score[evaluation]
                elif len(current_features) > 2:
                    # If performance did not improve, remove a random feature from current_features
                    feature_to_remove = random.choice(current_features)
                    current_features.remove(feature_to_remove)
                    waiting_features.add(feature_to_remove)

                iteration_count += 1

            return current_features, current_score 

        def genetic_algorithm_selection(self, features_lst, settings, dense_settings, lda_settings, epochs, batch_size, evaluation='Macro F1', population_size=6, generations=5, mutation_rate=0.3):
            """
            A Genetic Algorithm to find the optimal feature subset.
            Escapes local optima using Crossover and Mutation, and uses Memoization to prevent redundant training.
            """
            print("\n--- Starting Genetic Algorithm Feature Selection ---")
            
            # 1. Memoization Cache (The "Taboo/Memory" List)
            # Stores {tuple_of_features: score} to ensure we NEVER train the exact same subset twice.
            evaluated_cache = {}
            
            def evaluate_subset(subset):
                """Helper to evaluate a subset, utilizing the cache to save hours of training time."""
                # Sort to ensure ['tf', 'embedd'] and ['embedd', 'tf'] use the same cache key
                subset_key = tuple(sorted(subset))
                
                if not subset_key: # Failsafe for empty subsets
                    return 0.0
                    
                if subset_key in evaluated_cache:
                    print(f"   [Cache Hit] Skipping training for: {list(subset_key)} | Score: {evaluated_cache[subset_key]:.4f}")
                    return evaluated_cache[subset_key]
                    
                # If not in cache, train the model
                print(f"   [Training] Evaluating new subset: {list(subset_key)}")
                self.__trainModel(list(subset), settings, dense_settings, lda_settings, epochs, batch_size)
                
                # Build validation generator to handle varying lengths dynamically
                lda_gating_active = lda_settings.get("lda_gating", False) if lda_settings is not None else False
                
                validation_generator = DynamicBatchGenerator(
                    df = self.global_instances.test_df,
                    features = list(subset),
                    labels = self.global_instances.test_labels,
                    batch_size = batch_size,
                    settings = settings, # NEW: Pass the settings here
                    lda_gating = lda_gating_active,
                    full_doc_col = "original_tfidf"
                )
                
                # Get evaluation dictionary and extract the target metric (e.g., 'Macro F1')
                new_score_dict = self.global_instances.evaluate_model(validation_generator)
                score = new_score_dict[evaluation]
                
                self.global_instances.print_performance(list(subset), new_score_dict)
                
                # Save to cache
                evaluated_cache[subset_key] = score
                return score

            # 2. Smart Initialization
            # Start with individual features and a few random pairs, bypassing the massive cost of Forward Selection
            population = [[f] for f in features_lst]
            
            while len(population) < population_size:
                # Create random subsets of size 2 or 3 to inject immediate diversity
                random_k = random.randint(2, min(3, len(features_lst)))
                population.append(random.sample(features_lst, random_k))
                
            population = population[:population_size] # Ensure exact population size
            
            best_overall_features = []
            best_overall_score = -1.0

            # 3. The Evolutionary Loop
            for gen in range(generations):
                print(f"\n=== Generation {gen + 1} / {generations} ===")
                
                # Evaluate current population
                pop_scores = []
                for ind in population:
                    score = evaluate_subset(ind)
                    pop_scores.append((ind, score))
                    
                    # Track the global best
                    if score > best_overall_score:
                        best_overall_score = score
                        best_overall_features = ind
                
                # Sort population by score (Highest first)
                pop_scores.sort(key=lambda x: x[1], reverse=True)
                print(f"-> Best in Gen {gen+1}: {pop_scores[0][0]} | Score: {pop_scores[0][1]:.4f}")
                
                # Selection: Keep the top 50% as "Parents" for the next generation
                num_parents = max(2, population_size // 2)
                parents = [p[0] for p in pop_scores[:num_parents]]
                
                # Elitism: The best parents automatically survive to the next generation
                next_generation = parents.copy()
                
                # Crossover & Mutation to create "Children" (This handles your Swapping/Dropping logic)
                while len(next_generation) < population_size:
                    # Pick two random parents
                    p1, p2 = random.sample(parents, 2)
                    
                    # Crossover: Combine the traits of the parents
                    if random.random() > 0.5:
                        # Union: Take all unique features from both parents
                        child = list(set(p1) | set(p2))
                    else:
                        # Intersection & Random Mix: Take shared features + a random feature
                        shared = list(set(p1) & set(p2))
                        unique = list(set(p1) ^ set(p2))
                        child = shared + (random.sample(unique, 1) if unique else [])
                    
                    # Mutation: The Swapping/Dropping Mechanism
                    if random.random() < mutation_rate:
                        if random.random() > 0.5 and len(child) > 1:
                            # Drop a random feature (Explores smaller networks)
                            child.remove(random.choice(child))
                        else:
                            # Add a random new feature (Explores larger networks)
                            available_to_add = [f for f in features_lst if f not in child]
                            if available_to_add:
                                child.append(random.choice(available_to_add))
                                
                    # Failsafe: Ensure child is not empty
                    if not child:
                        child = [random.choice(features_lst)]
                        
                    next_generation.append(child)
                    
                # Move to next generation
                population = next_generation

            print(f"\n--- Genetic Algorithm Complete ---")
            print(f"Global Best Features Found: {best_overall_features}")
            print(f"Global Best Score ({evaluation}): {best_overall_score:.4f}")
            
            return best_overall_features, best_overall_score

        def __trainModel(self, features, settings, dense_settings, lda_settings, epochs, batch_size):
            
            # 1. Force TensorFlow to destroy the old hidden computational graphs
            tf.keras.backend.clear_session()
            
            # 2. Delete the old Python object
            if hasattr(self.global_instances, 'model') and self.global_instances.model is not None:
                del self.global_instances.model
                
            # 3. Force Python to immediately free up the physical RAM
            gc.collect()
            
            self.__customAdjustableModel(features=features,
                                         settings=settings,
                                         dense_settings=dense_settings,
                                         lda_settings=lda_settings)
            
            # Check if gating is active in the settings
            lda_gating = False
            if lda_settings is not None:
                lda_gating = lda_settings.get("lda_gating", False)

            # Initialize the dynamic batch generator with the new parameter
            training_generator = DynamicBatchGenerator(
                df = self.global_instances.train_df,
                features = features,
                labels = self.global_instances.train_labels,
                batch_size = batch_size,
                settings = settings,
                lda_gating = lda_gating,
                full_doc_col = "original_tfidf",
            )

            # Train the model using the generator
            self.global_instances.model.fit(
                x = training_generator, 
                epochs = epochs, 
                verbose = 0
            )

