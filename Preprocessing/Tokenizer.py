import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')

class Tokenizer:
    def __init__(self, lang) -> None:
        self.stop_words = set(stopwords.words(lang))
        self.lemmatizer = WordNetLemmatizer()



    def tokenize(self, text):
        # Tokenize
        words = word_tokenize(text)

        # Remove stopwords
        filtered_words = [word for word in words if word.lower() not in self.stop_words]

        # Lemmatization
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in filtered_words]

        return lemmatized_words
