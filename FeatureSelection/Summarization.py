from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoConfig, BertForSequenceClassification, AutoModelForSeq2SeqLM, EncoderDecoderModel, RobertaForSequenceClassification


class Summarization:
    def __init__(self, model_name):
        self.model_name = model_name
        self.device = 'cuda'
        self._tokenizer = None
        self._model = None
    
    def device(self, device):
        self.device = device

    def __summarize(self, text, original_max_length, sum_max_length, sum_min_length, truncation):
        inputs = self._tokenizer.encode(text, return_tensors="pt", max_length=original_max_length, truncation=truncation)
        summary_ids = self._model.generate(inputs.input_ids,
                                           num_beams=10,
                                           min_length=sum_min_length,
                                           max_length=sum_max_length,
                                           do_sample=True,
                                           top_p=0.95,
                                           length_penalty=1.0,
                                           early_stopping=True,
                                           repetition_penalty=2.5,
                                           no_repeat_ngram_size=3
                                           )
        summary = self._tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
        
    def summarize(self, original_text, original_max_length, sum_max_length, sum_min_length, truncation=True, ):
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        self._model.to(self.device)

        summarized_texts = []
        for text in tqdm(original_text, desc="Processing texts"):
            summary = self.__summarize(text,
                                       max_original_length=original_max_length,
                                       sum_max_length=sum_max_length,
                                       sum_min_length=sum_min_length,
                                       truncation = truncation
                                       )
            summarized_texts.append(summary)
        return summarized_texts



