from keras.models import Model
from keras.layers import Input, concatenate, Dense, Flatten, Dropout, Conv1D, MaxPooling1D, AveragePooling1D, Embedding
from keras.optimizers import Adam
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import pandas as pd
import random

class CustomFeatureSelection:
    
    def __init__(self, train_feature_df, train_labels, validation_feature_df, validate_labels, feature_dim, num_classes):
        self.train_feature_df = train_feature_df
        self.train_labels = train_labels
        self.validation_feature_df = validation_feature_df
        self.validate_labels = validate_labels
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.model = None
        self.vocab_size = None
        self.embedded_output_dim = None
        self.trainable = True
        self.embedding_weight = None


    def embedding_setup(self, vocab_size, embedded_output_dim, pretrained=None):
        
        self.vocab_size = vocab_size
        self.embedded_output_dim = embedded_output_dim
        if pretrained == 'GloVe':
            
            self.trainable = False
        elif pretrained == 'Word2Vec':

            self.trainable = False
        

    def CustomModel(self, features):
        inputs = []
        layers_to_concatinate = []
        for feature in features:
            if feature == "embedd":
                input = Input(shape=(self.feature_dim,))
                embedded_layer = Embedding(input_dim=self.vocab_size, output_dim=self.embedded_output_dim, weights=self.embedding_weight, trainable = self.trainable)(input)
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
        validate_array = [np.stack(self.validation_feature_df[feature].to_numpy()).reshape(-1, self.feature_dim, 1) for feature in features]
        predictions = self.model.predict(validate_array)
        predicted_labels = np.argmax(predictions, axis=1)
        
        if self.validate_labels.ndim > 1:  # if validate_labels is one-hot
            predicted_labels = np.eye(self.num_classes)[predicted_labels]

        if evaluation == 'accuracy':    
            return accuracy_score(self.validate_labels, predicted_labels)
        if evaluation == 'micro':
            return f1_score(self.validate_labels, predicted_labels, average="micro")

    
    def forward_feature_selection(self, evaluation = 'accuracy', epochs=5, batch_size=32):
        remained_features = list(self.train_feature_df.columns)
        best_score = 0.0
        selected_features = []

        while remained_features:
            best_feature = None
            for feature in remained_features:
                current_features = selected_features + [feature]

                # Build and train CNN model for the current feature combination
                self.CustomModel(current_features)

                train_array = [np.stack(self.train_feature_df[feature].to_numpy()).reshape(-1, self.feature_dim, 1) for feature in current_features]

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
        selected_features = set(self.train_feature_df.columns)
        best_score = 0.0
        
        while len(selected_features) > 1:
            best_combination = selected_features
            for feature in selected_features:
                current_features = list(selected_features - {feature})
                
                # Build and train CNN model for the current feature combination
                self.CustomModel(current_features)

                train_array = [np.stack(self.train_feature_df[feature].to_numpy()).reshape(-1, self.feature_dim, 1) for feature in current_features]

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

        features_lst = list(self.train_feature_df.columns)
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

            train_array = [np.stack(self.train_feature_df[feature].to_numpy()).reshape(-1, self.feature_dim, 1) for feature in current_features]

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

    def print_performance(self, features, score):
        print(f"{features}: {score}")