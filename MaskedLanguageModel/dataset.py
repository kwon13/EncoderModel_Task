from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer

class CorpusDataset(Dataset):
    def __init__(self, tokenizer:AutoTokenizer, file_path:str, max_len:int, mask_ratio:float):
        self.file_path = file_path
        with open(self.file_path, 'r') as docs:
            self.text =[t for t in docs.read().split('\n') if t.strip() ]
        
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.file_path = file_path
        self.mask_ratio = mask_ratio
        
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, idx):
    
        encodings = self.tokenizer(
            self.text[idx],
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
        )
        encodings['labels'] = encodings.input_ids.detach().clone().squeeze(0)
        # masking
        rand = torch.rand(encodings.input_ids.shape)
        rand_mask = (rand < self.mask_ratio) * (encodings.input_ids != self.tokenizer.cls_token_id) * \
            (encodings.input_ids != self.tokenizer.sep_token_id) * (encodings.input_ids != self.tokenizer.pad_token_id)
        selection = [torch.flatten(rand_mask[0].nonzero()).tolist()]
        encodings.input_ids[0, selection[0]] = self.tokenizer.mask_token_id
    
        encodings['input_ids'] = encodings['input_ids'].squeeze(0)
        encodings['attention_mask'] = encodings['attention_mask'].squeeze(0)
        encodings['token_type_ids'] = encodings['token_type_ids'].squeeze(0)
        
        return encodings
