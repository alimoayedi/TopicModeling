
from tensorflow.keras.layers import Input, Conv1D, AveragePooling1D, GlobalAveragePooling1D, Embedding
from tensorflow.keras.models import Model

class CustomModelGenerator:
    
    def __init__(self):
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
        

    def CustomModel(self, input_shape, filter_sizes, kernel_sizes, pool_sizes, embedding=None):
        if not (len(filter_sizes) == len(kernel_sizes) == len(pool_sizes)):
            raise ValueError("The lists 'filter_sizes', 'kernel_sizes', and 'pool_sizes' do not match in size.")
        
        if not isinstance(input_shape, tuple):
            raise ValueError("input_shape must be a tuple")
        
        model_hyper_parameters = list(zip(filter_sizes, kernel_sizes, pool_sizes))

        input_layer = Input(shape=input_shape)
        
        if not embedding:
            embedded_layer = Embedding(
                                        input_dim=self.vocab_size, 
                                        output_dim=self.embedded_output_dim, 
                                        weights=self.embedding_weight, 
                                        trainable = self.trainable)(input_layer)

        passing_layer = embedded_layer

        for setting in model_hyper_parameters:
            passing_layer = Conv1D(filters=setting[0], kernel_size=setting[1], activation='relu', padding='same')(passing_layer)
            passing_layer = AveragePooling1D(pool_size=setting[2])(passing_layer)

        output_layer = GlobalAveragePooling1D()(passing_layer)
        
        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.model.summary()
        
        return self.model
    

