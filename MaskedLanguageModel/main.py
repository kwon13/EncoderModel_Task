import random
import os
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from dataset import CorpusDataset
from model import MaskedModel
from train import train
import wandb

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
    
if __name__ == "__main__":
    config = OmegaConf.load("config/train_config.yaml")
    seed_everything(config.train.seed)
    
    wandb.init(project=config.logging.project,
        group=config.logging.group,
        name=config.logging.experiment_name,
        entity=config.logging.entity)
    
    createFolder(config.train.saved_path)
    
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_path)
    model = MaskedModel(config.model.model_path).to(config.train.device)
    train_dataset = CorpusDataset(tokenizer, file_path = config.path.train_data_path, max_len = config.model.max_len, mask_ratio = config.model.mask_ratio) # train dataset
    val_dataset = CorpusDataset(tokenizer, file_path = config.path.val_data_path, max_len = config.model.max_len, mask_ratio = config.model.mask_ratio) # val dataset
    
    
    train_dataloader = DataLoader(train_dataset,
                                  shuffle=config.train.train_suffle,
                                  pin_memory=True,
                                  batch_size=config.train.train_batch_size)
    val_dataloader = DataLoader(val_dataset,
                                  shuffle=config.train.val_shffle,
                                  pin_memory=True,
                                  batch_size=config.train.val_batch_size)
    
    optimizer = AdamW(model.parameters(),
                      lr=config.train.learning_rate)
    total_steps = len(train_dataloader) * config.train.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
    
    train(epochs=config.train.epochs,
          train_data_loader=train_dataloader, 
          val_data_loader=val_dataloader,
          model=model, 
          optimizer=optimizer,
          device=config.train.device,
          scheduler=scheduler,
          saved_path=config.train.saved_path,
          saved_name=config.train.saved_name)
    