from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoConfig, BertForSequenceClassification, AutoModelForSeq2SeqLM, EncoderDecoderModel, RobertaForSequenceClassification


class Summarization:
    def __init__(self, model_name):
        self.model_name = model_name
        self.device = 'cuda'
        self._tokenizer = None
        self._model = None
    
    def set_device(self, device):
        self.device = device

    def __summarize(self, text, sum_max_length, sum_min_length, truncation_length, truncation):
        inputs = self._tokenizer.encode(text, return_tensors="pt", max_length=truncation_length, truncation=truncation)
        inputs = inputs.to(self.device)
        summary_ids = self._model.generate(inputs,
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
        
    def summarize(self, original_text, oml, smaxl, sminl, truncation_length, truncation=True) -> list:
        """
        Summarize a given text using a language model.

        Parameters:
        ----------
        original_text : list or array
            The input text to summarize. It can be any string, such as an article or paragraph.
        oml : int
            (Original Max Lenght) The maximum length of a document that will not be summarized.
        smaxl : int
            (Summarized Maximum Length) The maximum lenght of generated summary.
        sminl : int
            (Summarized Minimum Length) The minimum lenght of generated summary.
        truncation_length : int
            The maximum length of document that is tokenized. Longer texts are truncated.
        truncation : bool, optional
            Longer texts than truncation_length should be truncated (default is True).
        Returns:
        -------
        str
            The summarized version of the input text.
        """

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        self._model.to(self.device)

        summarized_texts = []
        for text in tqdm(original_text, desc="Processing texts"):
            if len(text.split(" ")) > oml:
                summary = self.__summarize(text,
                                        sum_max_length=smaxl,
                                        sum_min_length=sminl,
                                        truncation_length = truncation_length,
                                        truncation = truncation
                                        )
                summarized_texts.append(summary)
            else:
                summarized_texts.append(text)
        return summarized_texts



