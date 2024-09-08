# Import libraries
import gc
import time
import torch
from torch.nn import DataParallel
import datasets
from transformers import AutoTokenizer, AutoModel



'''
Generate embeddings from encoder only models using mean pooling
'''
class EmbeddingsGenerator:
    def __init__(self, model_name: str, batch_size: int, **configs):
        self.model_name = model_name
        self.configs = configs
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.configs['device'])
        if torch.cuda.device_count() > 1:
            self.model = DataParallel(self.model)
        self.batch_size = batch_size
        self.model = self.model.eval()
    
    def tokenize_data(self, dataset):
        dataset = dataset.map(
            lambda x: self.tokenizer(
                x['text'],
                max_length=1024,
                add_special_tokens=True,
                padding='max_length', 
                truncation=True,
                return_tensors='pt'
            ),
            batched=True,
            batch_size=self.batch_size
        )
        return dataset
    
    @staticmethod
    def mean_pooling(model_output, attention_mask):
        last_hidden_state = model_output.last_hidden_state.detach().cpu()
        attention_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float().detach().cpu()
        masked_hidden_state = last_hidden_state * attention_mask
        sum_embeddings = masked_hidden_state.sum(dim=1)
        valid_token_count = attention_mask.sum(dim=1)
        valid_token_count = torch.clamp(valid_token_count, min=1e-9)
        mean_embeddings = sum_embeddings / valid_token_count
        return mean_embeddings

    def inference_by_model(self, batch):
        input_ids = torch.tensor(batch['input_ids']).to(self.configs['device'])
        attention_mask = torch.tensor(batch['attention_mask']).to(self.configs['device'])
        with torch.no_grad():
            model_output = self.model(input_ids=input_ids,
                                     attention_mask=attention_mask)
        if self.configs['pooling_strategy'][self.model_name] == 'mean':
            batch['{}_embeddings'.format(self.model_name)] = EmbeddingsGenerator.mean_pooling(model_output, attention_mask)
        return batch
    
    def get_embeddings(self, dataset):                
        dataset = dataset.map(
            lambda x: self.inference_by_model(x),
            batched=True,
            batch_size=self.batch_size
        )
        return dataset
    
    def filter_features(self, dataset):
        keep_features = ["essay_id", "text", "word_features", "sentence_features"]
        keep_features += [feature for feature in dataset.column_names if feature.endswith("_embeddings")]
        dataset = dataset.remove_columns([feature for feature in dataset.column_names if feature not in keep_features])
        return dataset
    
    def cleanup(self):
        del self.tokenizer
        del self.model
    
    def __call__(self, dataset: datasets.DatasetDict):
        dataset = self.tokenize_data(dataset)
        dataset = self.get_embeddings(dataset)
        dataset = self.filter_features(dataset)
        gc.collect()
        time.sleep(5)
        torch.cuda.empty_cache()
        return dataset