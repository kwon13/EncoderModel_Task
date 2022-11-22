from torch import nn
from transformers import AutoModelForMaskedLM

class MaskedModel(nn.Module):
    def __init__(self, model_name):
        super().__init__() 
        self.model_name = model_name
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)        
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels)
        return outputs