import pandas as pd
import numpy as np
from rouge_score import rouge_scorer
import multiprocessing as mp
import warnings

# Suppress minor warnings for cleaner output
warnings.filterwarnings('ignore')

# We define the scorer globally for the worker processes 
# so it doesn't need to be instantiated repeatedly.
_scorer = None

def _init_worker():
    global _scorer
    _scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def _score_row(args):
    gen, ref = args
    scores = _scorer.score(str(ref), str(gen))
    return {
        'rouge1_f1': scores['rouge1'].fmeasure,
        'rouge1_p': scores['rouge1'].precision,
        'rouge1_r': scores['rouge1'].recall,
        'rouge2_f1': scores['rouge2'].fmeasure,
        'rouge2_p': scores['rouge2'].precision,
        'rouge2_r': scores['rouge2'].recall,
        'rougeL_f1': scores['rougeL'].fmeasure,
        'rougeL_p': scores['rougeL'].precision,
        'rougeL_r': scores['rougeL'].recall
    }

class RougeEvaluator:
    def __init__(self, generated_col='generated_summary', reference_col='reference_summary', num_workers=None):
        """
        Initialize the multiprocessing evaluator.
        
        :param generated_col: Column name containing the generated summaries.
        :param reference_col: Column name containing the ground truth summaries.
        :param num_workers: Number of CPU cores to use. Defaults to CPU count - 1.
        """
        self.generated_col = generated_col
        self.reference_col = reference_col
        # Leave one core free for OS tasks, but ensure at least 1 worker
        self.num_workers = num_workers or max(1, mp.cpu_count() - 1)

    def evaluate_dataframe(self, df):
        """
        Evaluates ROUGE scores for summaries in a pandas DataFrame using multiprocessing.
        """
        # 1. Using zip to create a generator of tuples is significantly faster than df.iterrows()
        tasks = zip(df[self.generated_col], df[self.reference_col])
        
        # 2. Setup the multiprocessing pool
        with mp.Pool(processes=self.num_workers, initializer=_init_worker) as pool:
            # Map the scoring function across all rows concurrently
            results_list = pool.map(_score_row, tasks)
            
        # 3. Convert the list of result dictionaries directly into a DataFrame
        results_df = pd.DataFrame(results_list)
        
        # Concatenate the original DataFrame with the new results
        detailed_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)
        
        # Calculate Aggregated (Mean) Scores
        agg_scores = results_df.mean().to_dict()
        
        return agg_scores, detailed_df