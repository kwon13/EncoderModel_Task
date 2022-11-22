import torch
from tqdm import tqdm
import wandb
import gc

def train(epochs, train_data_loader, val_data_loader, model, optimizer, device, scheduler, saved_path, saved_name):
    final_val_loss=0
    for epoch in range(epochs):
        gc.collect()
        model.train()
        pbar = tqdm(train_data_loader)
        for i, batch in enumerate(pbar):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask,
                            labels=labels)
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss = loss.detach().cpu().item()
            pbar.set_postfix({'step_train_loss': train_loss, 
                            "lr": optimizer.param_groups[0]["lr"]})
            
            wandb.log({'train_loss':train_loss})
            wandb.log({'learning_rate':optimizer.param_groups[0]["lr"]})
            
        model.eval()
        pbar = tqdm(val_data_loader)
        for i, batch in enumerate(pbar):
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask,
                            labels=labels)
            
            loss = outputs.loss
            
            val_loss = loss.detach().cpu().item()
            final_val_loss+=val_loss

        final_val_loss/=i+1
        wandb.log({'epoch_val_loss':final_val_loss})
        print(f'epoch_val_loss:{final_val_loss}')
        torch.save(model.state_dict(), f'{saved_path}/{saved_name}:{epoch}_model.pt')
        
    del train_data_loader, val_data_loader
    gc.collect()