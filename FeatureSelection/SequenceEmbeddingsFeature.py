from sentence_transformers import SentenceTransformer

class SequenceEmbeddingsFeature:

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = 'all-MiniLM-L6-v2'

    def generate_contextual_embeddings(self, df, text_column='text'):
        """
        Passes documents through a lightweight BERT transformer.
        Returns a 384-dimensional dense semantic vector representing the entire text.
        """
        print(f"Generating Contextual Embeddings (Transformer) for {len(df)} documents...")
        
        # 'all-MiniLM-L6-v2' is chosen because it is extremely fast, highly accurate, 
        # and has a small memory footprint (perfect for multi-modal architectures).
        model = SentenceTransformer(self.model_name)
        
        # Convert the column to a list of strings
        texts = df[text_column].astype(str).tolist()
        
        # Encode the texts into embeddings
        # show_progress_bar gives you a nice visual in the terminal so you know it hasn't frozen
        embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
        
        # Convert the resulting 2D matrix into a list of 1D arrays for DataFrame compatibility
        return list(embeddings)