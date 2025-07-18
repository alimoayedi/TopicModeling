from keras.models import Model
from keras.layers import (Input, concatenate, Dense, Flatten, 
                          Dropout, Conv1D, MaxPooling1D, AveragePooling1D, 
                          Embedding, BatchNormalization, Lambda, Multiply)

from tensorflow.keras.layers import Reshape, AdditiveAttention 
import tensorflow as tf
from keras.optimizers import Adam
from sklearn.metrics import f1_score, accuracy_score, classification_report
import numpy as np
import pandas as pd
import random


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
    
    def evaluate_model(self, validate_array):
        # Get model predictions
        predictions = self.model.predict(validate_array)
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

    def get_lda(self, dense, dropout):

        lda_input = Input(shape=(self.num_classes,), name='lda_input')
        x_lda = BatchNormalization(name="lda_batch_input")(lda_input)
        x_lda = Dense(dense, activation='relu', name="lda_proj")(x_lda)
        x_lda = Dropout(dropout, name="lda_dropout")(x_lda)
        x_lda = BatchNormalization(name="lda_batch_proj")(x_lda)                   
        x_lda = Flatten()(x_lda)

        return lda_input, x_lda



    class EqualSizedFeatureSelection:
        
        def __init__(self, feature_dim):
            self.feature_dim = feature_dim
            self.model = None

        def CustomModel(self, features):
            inputs = []
            layers_to_concatinate = []
            for feature in features:
                if feature == "embedd":
                    input = Input(shape=(self.feature_dim,))
                    embedded_layer = Embedding(
                                                input_dim=self.vocab_size, 
                                                output_dim=self.embedded_output_dim, 
                                                weights=self.embedding_weight, 
                                                trainable = self.trainable)(input)
                    conv_layer_1 = Conv1D(filters=256, kernel_size=5, activation='relu')(embedded_layer) 
                else:
                    input = Input(shape=(self.feature_dim,1))
                    conv_layer_1 = Conv1D(filters=256, kernel_size=5, activation='relu')(input)
                
                inputs.append(input)
                pooling_layer_1 = AveragePooling1D(pool_size=2)(conv_layer_1)
                conv_layer_2 = Conv1D(filters=128, kernel_size=4, activation='relu')(pooling_layer_1)
                pooling_layer_2 = AveragePooling1D(pool_size=2)(conv_layer_2)
                conv_layer_3 = Conv1D(filters=64, kernel_size=3, activation='relu')(pooling_layer_2)
                pooling_layer_3 = AveragePooling1D(pool_size=2)(conv_layer_3)
                conv_layer_4 = Conv1D(filters=32, kernel_size=3, activation='relu')(pooling_layer_3)
                pooling_layer_4 = AveragePooling1D(pool_size=2)(conv_layer_4)
                flatten_layer = Flatten()(pooling_layer_4)
                layers_to_concatinate.append(flatten_layer)
            
            if len(layers_to_concatinate) > 1:
                merged = concatenate(layers_to_concatinate)
            else:
                merged = layers_to_concatinate[0]

            # Additional dense layers for further processing
            dense_layer_1 = Dense(64, activation='relu')(merged)
            dropout_layer_1 = Dropout(0.1)(dense_layer_1)
            dense_layer_2 = Dense(32, activation='relu')(dropout_layer_1)
            dropout_layer_2 = Dropout(0.1)(dense_layer_2)
            dense_layer_3 = Dense(16, activation='relu')(dropout_layer_2)
            output_layer = Dense(self.num_classes, activation='softmax')(dense_layer_3)

            # Compile the model
            model = Model(inputs=inputs, outputs=output_layer)
            model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
            # model.summary()
            self.model = model
        
        # Function to evaluate model performance on validation data
        def evaluate_model(self, features, evaluation):
            validate_array = [np.stack(self.test_df[feature].to_numpy()).reshape(-1, self.feature_dim, 1) for feature in features]
            predictions = self.model.predict(validate_array)
            predicted_labels = np.argmax(predictions, axis=1)
            
            if self.test_labels.ndim > 1:  # if test_labels is one-hot
                predicted_labels = np.eye(self.num_classes)[predicted_labels]

            if evaluation == 'accuracy':    
                return accuracy_score(self.test_labels, predicted_labels)
            if evaluation == 'micro':
                return f1_score(self.test_labels, predicted_labels, average="micro")

        
        def forward_feature_selection(self, evaluation = 'accuracy', epochs=5, batch_size=32):
            remained_features = list(self.train_df.columns)
            best_score = 0.0
            selected_features = []

            while remained_features:
                best_feature = None
                for feature in remained_features:
                    current_features = selected_features + [feature]

                    # Build and train CNN model for the current feature combination
                    self.CustomModel(current_features)

                    train_array = [np.stack(self.train_df[feature].to_numpy()).reshape(-1, self.feature_dim, 1) for feature in current_features]

                    self.model.fit(train_array, self.train_labels, epochs=epochs, batch_size=batch_size, verbose=0)
                    
                    # Evaluate model on validation data
                    score = self.evaluate_model(current_features, evaluation)

                    if score > best_score:
                        best_score = score
                        best_feature = feature
                    
                    self.print_performance(current_features, score)
                
                # If no improvement, stop the process
                if best_feature:
                    selected_features.append(best_feature)
                    remained_features.remove(best_feature)
                else:
                    break

            return selected_features, best_score

        # Backward feature elimination using micro F-score
        def backward_feature_selection(self, evaluation = 'accuracy', epochs=5, batch_size=32):
            selected_features = set(self.train_df.columns)
            best_score = 0.0
            
            while len(selected_features) > 1:
                best_combination = selected_features
                for feature in selected_features:
                    current_features = list(selected_features - {feature})
                    
                    # Build and train CNN model for the current feature combination
                    self.CustomModel(current_features)

                    train_array = [np.stack(self.train_df[feature].to_numpy()).reshape(-1, self.feature_dim, 1) for feature in current_features]

                    self.model.fit(train_array, self.train_labels, epochs=epochs, batch_size=batch_size, verbose=0)
                    
                    # Evaluate model on validation data
                    score = self.evaluate_model(current_features,evaluation)
                    
                    if score > best_score:
                        best_score = score
                        best_combination = set(current_features)

                    self.print_performance(current_features, score)
                
                # Update selected features with the best combination if it improved F-score
                if best_combination != selected_features:
                    selected_features = best_combination
                else:
                    break

            return selected_features, best_score

        def randomized_feature_selection(self, num_iterations=20, evaluation = 'accuracy', epochs=5, batch_size=32):

            features_lst = list(self.train_df.columns)
            waiting_features = set()
            iteration_count = 0

            # Select the best feature from the features_lst
            current_features, current_score = self.forward_feature_selection(evaluation=evaluation, epochs=epochs, batch_size=batch_size)

            while iteration_count < num_iterations and len(current_features) < len(features_lst):
                # Select a random feature from the remaining features_lst
                remaining_features = list(set(features_lst) - set(current_features) - waiting_features)

                if not remaining_features:
                    break
                
                random_feature = random.choice(remaining_features)
                current_features.append(random_feature)
                
                waiting_features.clear()  # clear waiting features for future chance of being added to the feature combination

                self.CustomModel(current_features)

                train_array = [np.stack(self.train_df[feature].to_numpy()).reshape(-1, self.feature_dim, 1) for feature in current_features]

                self.model.fit(train_array, self.train_labels, epochs=epochs, batch_size=batch_size, verbose=0)
                
                # Evaluate model on validation data
                new_score = self.evaluate_model(current_features, evaluation)

                self.print_performance(current_features, new_score)

                if new_score > current_score:
                    # If performance improved, update current performance
                    current_score = new_score
                elif len(current_features) > 2:
                    # If performance did not improve, remove a random feature from current_features
                    feature_to_remove = random.choice(current_features)
                    current_features.remove(feature_to_remove)
                    waiting_features.add(feature_to_remove)

                iteration_count += 1

            return current_features, current_score

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

            # loops over features
            for feature in features:

                if feature == 'lda':
                    lda_input, x_lda = self.global_instances.get_lda(lda_settings.get("lda_dense"), lda_settings.get("lda_dropout"))
                    inputs.append(lda_input)
                    layers_to_concatinate.append(x_lda)  # shape (None, lda_proj_dim)
                    continue

                # gets the settings of each feature
                setting = settings.loc[feature]
                
                # check neural network parameters
                if not (len(setting['filter_sizes']) == len(setting['kernel_sizes']) == len(setting['pool_sizes'])):
                    raise ValueError("The lists 'filter_sizes', 'kernel_sizes', and 'pool_sizes' do not match in size.")
                
                # gets the feature dimension
                feature_dim = setting['feature_dim']
                # gets neural network hyper-parammeters and zip them
                model_hyper_parameters = list(zip(setting['filter_sizes'], 
                                                setting['kernel_sizes'],
                                                setting['pool_sizes']))

                if setting['embedding']:
                    input_layer = Input(shape=(feature_dim,))
                    embedded_layer = Embedding(input_dim=self.global_instances.vocab_size, 
                                                output_dim=self.global_instances.embedded_output_dim, 
                                                weights=self.global_instances.embedding_weight, 
                                                trainable = self.global_instances.trainable)(input_layer)            
                    passing_layer = embedded_layer
                else:
                    input_layer = Input(shape=(feature_dim,1))
                    passing_layer = input_layer
                
                inputs.append(input_layer)
                
                if setting['normalize']:
                        passing_layer = BatchNormalization()(passing_layer)

                for param_set in model_hyper_parameters:
                    passing_layer = Conv1D(filters=param_set[0], kernel_size=param_set[1], activation='relu', padding='same')(passing_layer)
                    passing_layer = AveragePooling1D(pool_size=param_set[2])(passing_layer)

                flatten_layer = Flatten()(passing_layer)

                layers_to_concatinate.append(flatten_layer)
            
            if lda_gating:
                # apply lda gating
                _, x_lda = self.global_instances.get_lda(lda_settings.get("lda_dense"), lda_settings.get("lda_dropout"))
                gate_hidden = Dense(max(16, self.global_instances.num_classes), activation='relu', name="gate_hidden")(x_lda)
                gate_vector = Dense(len(layers_to_concatinate), activation='sigmoid', name="branch_gate")(gate_hidden)

                gated_layers_to_concatinate = []

                for index, branch in enumerate(layers_to_concatinate):
                    # Extract gate scalar for branch i
                    # Using Lambda to slice: gate_vector[:, i:i+1]
                    gate_i = Lambda(lambda vector, idx=index: vector[:, idx:idx+1], name=f"gate_slice_{index}")(gate_vector)
                    # Multiply branch representation by gate scalar (broadcast)
                    gated_branch = Multiply(name=f"gated_branch_{index}")([branch, gate_i])
                    gated_layers_to_concatinate.append(gated_branch)
                
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
        
        def randomized_selection(self, features_settings, dense_settings, num_iterations=20, evaluation = 'accuracy', epochs=5, batch_size=32):
            if not isinstance(self.global_instances.train_df,  pd.DataFrame):
                raise ValueError("train type must be a pandas dataframe")
            
            if not isinstance(self.global_instances.test_df,  pd.DataFrame):
                raise ValueError("test type must be a pandas dataframe")
            
            if not isinstance(features_settings,  pd.DataFrame):
                raise ValueError("settings type must be a pandas dataframe, including kernel_size, pool_size, embedding (boolean), feature_dim (dimension -> int)")
            
            if not isinstance(dense_settings,  pd.DataFrame):
                raise ValueError("settings type must be a pandas dataframe, including kernel_size, pool_size, embedding (boolean), feature_dim (dimension -> int)")

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

        def __trainModel(self, features, settings, dense_settings, lda_settings, epochs, batch_size):
            
            del self.global_instances.model
            
            self.__customAdjustableModel(features=features,
                                         settings=settings,
                                         dense_settings=dense_settings,
                                         lda_settings=lda_settings)
                    
            train_array = [np.stack(self.global_instances.train_df[feature]) for feature in features]

            self.global_instances.model.fit(train_array, self.global_instances.train_labels, epochs=epochs, batch_size=batch_size, verbose=0)
                    

