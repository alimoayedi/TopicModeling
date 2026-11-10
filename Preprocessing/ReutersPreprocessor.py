from Tokenizer import Tokenizer

class ReutersPreprocessor:

    def __init__(self, col_name):
        self.col_name = col_name

    def preprocess(self, documents, min_doc_length, remove_stopwords=True, lemmatize=True, lang='english'):
        
        # preprocess data by tokenization and lemmatization
        documents['preprocess'] = documents[self.col_name].apply(lambda text: Tokenizer(lang).tokenize(text, remove_stopwords, lemmatize))

        # drop preprocessed documents with length less than 6
        documents = documents[documents['preprocess'].str.len() > min_doc_length]

        return documents