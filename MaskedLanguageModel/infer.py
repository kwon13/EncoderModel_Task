import random
import os
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from dataset import CorpusDataset
from model import MaskedModel

if __name__ == "__main__":
    config = OmegaConf.load("config/train_config.yaml")
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_path)
    model = MaskedModel(config.model.model_path).to(config.train.device)
    checkpoint = torch.load('/root/MaskedLanguageModel/checkpoints/saved_model:4_model.pt')
    model.load_state_dict(checkpoint)
    
    encoding = tokenizer(
            config.test,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=config.model.max_len)
    
    with torch.no_grad():
        token_logits = model(
            input_ids = encoding.input_ids.to(config.train.device),
            attention_mask = encoding.attention_mask.to(config.train.device),
        ).logits
    
    mask_token_index = torch.where(encoding.input_ids == tokenizer.mask_token_id)[1]
    mask_token_logits = token_logits[0, mask_token_index, :]
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
    
    print(f'원래 문장:{config.test}')
    for token in top_5_tokens:
        print(f"'>>> {config.test.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")
