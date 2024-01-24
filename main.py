import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
response_data_fp = DATA_DIR / "DLA Piper LLM Prompts and Survey Results.csv"
risk_data_fp = DATA_DIR / "risk_register29112023.py"
OUTPUT_DIR = Path("output")

class BertStsbClf():
    def _init_(self,model_name:str="sentence-transformers/all-mpnet-base-v2", response_data_fp:str="", risk_data_fp:str="", cut:int=None, number_of_questions:int = 5):
        self._model_name= model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._mapped_above_threshold = None
        self._sentence_sentiments_list = None
        self._number_of_questions = number_of_questions
        self.load_risk_data(risk_data_fp=risk_data_fp)
        self.load_response_data(response_data_fp=response_data_fp, cut=cut)
    
    def load_response_data(self, response_data_fp: str, cut:int=None):
        df = pd.read_csv(response_data_fp)
        df_answers = df[["Answer 1", "Answer 2", "Answer 3", "Answer 4", "Answer 5"]]
        response_lists = df_answers.values.tolist()
        self._response_size_list = [[len(answer.split('.')) if answer else 0 for answer in response] for response in response_lists]
        self._cumsum_response = np.cumsum([0] + [sum(response) for response in self._response_size_list]).tolist()
        self._cumsum_answer = np.cumsum([0] + [answer for response in self._response_size_list for answer in response]).tolist()
        response_sentences = [sentence.strip() for entry in response_lists for sentence_group in entry for sentence in sentence_group.split('.') if sentence.strip()]
        if cut:
            self._response_size_list = self._response_size_list[:cut]
            sentences_cut = sum([sum(inner_list) for inner_list in self._response_size_list[:cut]])
            response_sentences = response_sentences[:sentences_cut]
        self._response_sentences = response_sentences

    def load_risk_data(self, risk_data_fp:str):
        loaded_data = {}
        exec(open(risk_data_fp).read(), {}, loaded_data)
        risk_sentences_dict = loaded_data.get("risk_register")
        thresholds_dict = loaded_data.get("thresholds")
        assert len(risk_sentences_dict) == len(thresholds_dict)
        risk_sentences = [item for sublist in risk_sentences_dict.values() for item in sublist]
        self._thresholds_list = [value for key, value in thresholds_dict.items()]
        self._risk_factor_size_list = [len(value) for key, value in risk_sentences_dict.items()]
        self._risk_factors = [key for key, value in risk_sentences_dict.items()]
        self._risk_sentences = risk_sentences

    def mean_pooling(self, model_output, attention_mask: torch.Tensor):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        mean_pooled_embeddings = sum_embeddings / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
        return mean_pooled_embeddings

    def get_sentence_embeddings(self, sentences:list[str]):
        tokenized_sentences = self._tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            model_output = self._model(**tokenized_sentences)
        return self.mean_pooling(model_output, tokenized_sentences["attention_mask"])

    def get_risk_embeddings(self):
        return self.get_sentence_embeddings(self._risk_sentences)

    def get_response_embeddings(self):
        return self.get_sentence_embeddings(self._response_sentences)

    def get_cosine_similarities(self, max_similarity:bool=False):
        """ 
        max_similarities is a boolean value to use the max similarity value or mean (default)
        
        mapped_above_threshold is a disagregated binary structure lst[lst[lst[boolean]]] where: 

                list of factors
                list of responses
                list of answers
                list of sentences
            
        """
        if self._mapped_above_threshold is None:
            risk_embeddings = self.get_risk_embeddings()
            response_embeddings = self.get_response_embeddings()
            similarities = []
            for risk_embedding in risk_embeddings:
                similarities.append(cosine_similarity(response_embeddings, risk_embedding.reshape(1, -1)))
            similarities_array = np.squeeze(np.array(similarities))
            if max_similarity:
                self._similarities_per_factor_array = np.array([np.max(subarray, axis=0) for subarray in np.vsplit(similarities_array, np.cumsum(self._risk_factor_size_list)[:-1])])
            else:
                self._similarities_per_factor_array = np.array([np.mean(subarray, axis=0) for subarray in np.vsplit(similarities_array, np.cumsum(self._risk_factor_size_list)[:-1])])
            self._mapped_similarities = [
                [self._similarities_per_factor_array[:, sum(self._response_size_list[i][:j]): sum(self._response_size_list[i][:j + 1])] for j in range(len(self._response_size_list[i]))]
                for i in range(len(self._response_size_list))
            ]
            self._above_thresholds_per_factor_and_sentence_array = self._similarities_per_factor_array > np.array(self._thresholds_list)[:, np.newaxis]
            self._mapped_above_threshold = []
            for factor in range(self._above_thresholds_per_factor_and_sentence_array.shape[0]):
                reponse_list = []    
                answer_list = []
                for i in range(len(self._cumsum_answer)-1):
                    answer_list.append(self._above_thresholds_per_factor_and_sentence_array[factor,self._cumsum_answer[i]:self._cumsum_answer[i+1]].tolist())
                    if len(answer_list) == self._number_of_questions:
                        reponse_list.append(answer_list)
                        answer_list = []
                self._mapped_above_threshold.append(reponse_list)
        return self._mapped_above_threshold

    def get_classification(self):
        if self._mapped_above_threshold == None:
            self.get_cosine_similarities()
        classified_sentences = {
            self._risk_factors[factor]: [self._response_sentences[i] for i in range(len(self._response_sentences)) if self._above_thresholds_per_factor_and_sentence_array[factor, i]]
            for factor in range(len(self._risk_factors))
        }
        response_binary = np.zeros(shape=(len(self._risk_factors),len(self._cumsum_response)-1))
        answer_binary = np.zeros(shape=(len(self._risk_factors),len(self._cumsum_answer)-1))
        for factor in range(self._above_thresholds_per_factor_and_sentence_array.shape[0]):
            for i in range(len(self._cumsum_response)-1):
                if np.any(self._above_thresholds_per_factor_and_sentence_array[factor,self._cumsum_response[i]:self._cumsum_response[i+1]]):
                    response_binary[factor,i] = 1 
        for factor in range(self._above_thresholds_per_factor_and_sentence_array.shape[0]):
            for i in range(len(self._cumsum_answer)-1):
                if np.any(self._above_thresholds_per_factor_and_sentence_array[factor,self._cumsum_answer[i]:self._cumsum_answer[i+1]]):
                    answer_binary[factor,i] = 1 
        sentence_counts = np.sum(self._above_thresholds_per_factor_and_sentence_array, axis=1, dtype=int).tolist()
        answer_counts = np.sum(answer_binary, axis=1, dtype=int).tolist()
        response_counts = np.sum(response_binary, axis=1, dtype=int).tolist()
        count_per_factor_and_sentence_dict = dict(zip(self._risk_factors,sentence_counts))
        count_per_factor_and_answer_dict = dict(zip(self._risk_factors,answer_counts))
        count_per_factor_and_response_dict = dict(zip(self._risk_factors,response_counts))
        return classified_sentences, count_per_factor_and_sentence_dict, count_per_factor_and_answer_dict, count_per_factor_and_response_dict
    
    def get_sentiments(self, pipeline_name:str="sentiment-analysis", model_name: str="Seethal/sentiment_analysis_generic_dataset"):
        if self._sentence_sentiments_list is None:
            sentiment_analysis = pipeline(pipeline_name,model=model_name)
            sentiment_list = [sentiment_analysis(response_sentence)[0] for response_sentence in self._response_sentences]
            label_mapping = {"LABEL_0": -1, "LABEL_1": 0, "LABEL_2": 1}
            self._sentence_sentiments_list = [label_mapping[entry["label"]]  for entry in sentiment_list]
            self._answer_sentiments_list = [np.nan_to_num(np.mean(self._sentence_sentiments_list[self._cumsum_answer[i]:self._cumsum_answer[i+1]])).tolist() for i in range(len(self._cumsum_answer)-1)]
            self._response_sentiments_list = [np.nan_to_num(np.mean(self._sentence_sentiments_list[self._cumsum_response[i]:self._cumsum_response[i+1]])).tolist() for i in range(len(self._cumsum_response)-1)]
        sentence_sentiment_dict = dict(zip([f"Sentence_{i}" for i in range(len(self._sentence_sentiments_list))], self._sentence_sentiments_list))
        answer_sentiment_dict = dict(zip([f"Answer_{i}" for i in range(len(self._answer_sentiments_list))], self._answer_sentiments_list))
        response_sentiment_dict = dict(zip([f"Response_{i}" for i in range(len(self._response_sentiments_list))], self._response_sentiments_list))
        return sentence_sentiment_dict, answer_sentiment_dict, response_sentiment_dict
    
    def get_mapped_sentiments(self):
        if self._mapped_above_threshold is None:
            self.get_cosine_similarities()
        if self._sentence_sentiments_list is None:
            self.get_sentiments()
        mapped_sentiments_per_factor_and_sentence = np.where(self._above_thresholds_per_factor_and_sentence_array, np.array(self._sentence_sentiments_list)[np.newaxis, :], np.nan)
        self._risk_factor_sentiments_list = np.nan_to_num(np.nanmean(mapped_sentiments_per_factor_and_sentence,axis=1)).tolist()
        count_pos = np.sum(np.where(mapped_sentiments_per_factor_and_sentence == 1, 1, 0), axis=1).tolist()
        count_neutral = np.sum(np.where(mapped_sentiments_per_factor_and_sentence == 0, 1, 0), axis=1).tolist()
        count_neg = np.sum(np.where(mapped_sentiments_per_factor_and_sentence == -1, 1, 0), axis=1).tolist()
        risk_factor_sentiments_dict = dict(zip(self._risk_factors, list(zip(count_pos, count_neutral, count_neg))))
        risk_factor_sentiments_dict = {key: list(value) for key, value in risk_factor_sentiments_dict.items()}
        # count_pos = np.sum(np.where(mapped_sentiments_per_factor_and_sentence == 1, 1, 0), axis=1).tolist()
        # count_neutral = np.sum(np.where(mapped_sentiments_per_factor_and_sentence == 0, 1, 0), axis=1).tolist()
        # count_neg = np.sum(np.where(mapped_sentiments_per_factor_and_sentence == -1, 1, 0), axis=1).tolist()
        return risk_factor_sentiments_dict