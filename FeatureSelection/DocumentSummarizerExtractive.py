import networkx as nx
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

class DocumentSummarizerExtractive:
    def __init__(self, max_sentences=1):
        """
        Initializes the summarization module using the TextRank graph-based algorithm.
        """
        self.max_sentences = max_sentences
        # Ensure the sentence tokenizer is available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

    def set_max_sentences(self, max_sentences):
        """
        Sets the maximum number of sentences for the summary.
        """
        if isinstance(max_sentences, int) and max_sentences > 0:
            self.max_sentences = max_sentences
        else:
            raise ValueError("max_sentences must be a positive integer.")

    def generate_summary(self, text):
        """
        Generates an extractive summary using eigenvector centrality.
        """
        # Return empty if invalid input
        if not isinstance(text, str) or len(text.strip()) == 0:
            return ""

        sentences = sent_tokenize(text)
        
        # If document is already shorter than the target length, return it whole
        if len(sentences) <= self.max_sentences:
            return text

        # 1. Map sentences to dense feature arrays (TF-IDF representation)
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            sentence_vectors = vectorizer.fit_transform(sentences)
        except ValueError:
            # Fallback if text contains only stop words or unrecognizable tokens
            return " ".join(sentences[:self.max_sentences])

        # 2. Compute the similarity matrix between all sentence pairs
        similarity_matrix = cosine_similarity(sentence_vectors)

        # 3. Formulate the topological graph
        nx_graph = nx.from_numpy_array(similarity_matrix)

        # 4. Calculate PageRank scores for sentence centrality
        scores = nx.pagerank(nx_graph)

        # 5. Rank and extract the top N sentences
        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
        extracted_sentences = [ranked_sentences[i][1] for i in range(self.max_sentences)]
        
        # 6. Re-order the extracted sentences to match chronological document order
        original_order_summary = [s for s in sentences if s in extracted_sentences]

        return " ".join(original_order_summary)