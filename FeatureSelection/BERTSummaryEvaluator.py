import pandas as pd
from bert_score import BERTScorer
import torch

class BERTSummaryEvaluator:
    def __init__(self, generated_col='generated_summary', reference_col='reference_summary', lang='en', model_type='bert-base-uncased', use_baseline=True):
        """
        Initialize the BERTScore evaluator.
        
        :param generated_col: Column containing generated summaries.
        :param reference_col: Column containing ground truth summaries.
        :param lang: Language for evaluation (default is 'en'). Use 'multilingual' for other languages.
        :param model_type: The underlying BERT model to use (default: 'bert-base-uncased').
        :param use_baseline: Whether to rescale the output scores for better human readability.
        """
        self.generated_col = generated_col
        self.reference_col = reference_col
        self.lang = lang
        self.model_type = model_type
        self.use_baseline = use_baseline
        
        # Check if GPU is available to speed up computation
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Initializing BERTScorer on: {self.device}")
        
        # Initialize the scorer object to cache the model in memory
        self.scorer = BERTScorer(model_type=self.model_type, 
                                 lang=self.lang, 
                                 rescale_with_baseline=self.use_baseline, 
                                 device=self.device)

    def evaluate_dataframe(self, df):
        """
        Evaluates BERTScore for summaries in a pandas DataFrame.
        """
        # Convert pandas columns to lists of strings
        cands = df[self.generated_col].astype(str).tolist()
        refs = df[self.reference_col].astype(str).tolist()
        
        print("Calculating BERTScore... (This may take a while depending on data size and hardware)")
        # Calculate scores in one batch operation
        P, R, F1 = self.scorer.score(cands, refs)
        
        # Convert PyTorch tensors to Python lists
        p_list = P.tolist()
        r_list = R.tolist()
        f1_list = F1.tolist()
        
        # Append detailed scores back to the dataframe
        detailed_df = df.copy()
        detailed_df['bertscore_precision'] = p_list
        detailed_df['bertscore_recall'] = r_list
        detailed_df['bertscore_f1'] = f1_list
        
        # Calculate Aggregated (Mean) Scores
        agg_scores = {
            'bertscore_precision': sum(p_list) / len(p_list) if p_list else 0,
            'bertscore_recall': sum(r_list) / len(r_list) if r_list else 0,
            'bertscore_f1': sum(f1_list) / len(f1_list) if f1_list else 0
        }
        
        return agg_scores, detailed_df