import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')

class Tokenizer:
    def __init__(self, lang) -> None:
        self.stop_words = set(stopwords.words(lang))
        self.lemmatizer = WordNetLemmatizer()
        self.regex_tokenizer = RegexpTokenizer(r'\w+')

    def tokenize(self, text, remove_stopwords, lemmatize) -> list:
        # Tokenize
        symbol_removed_text = self.regex_tokenizer.tokenize(text)
        tokenized = word_tokenize(' '.join(symbol_removed_text))

        # Remove stopwords
        if remove_stopwords:
            stopwords_filtered = [token for token in tokenized if token.lower() not in self.stop_words]
        else:
            stopwords_filtered = tokenized
        

        # Lemmatization
        if lemmatize:
            lemmatized_words = [self.lemmatizer.lemmatize(word) for word in stopwords_filtered]
            return lemmatized_words
        else:
            return stopwords_filtered