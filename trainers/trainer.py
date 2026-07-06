from tqdm import tqdm
import torch

def train_one_epoch(model, loader, optimizer, loss_fn, tokenizer, device):
    model.train()
    epoch_loss = 0

    loop = tqdm(loader, desc="Batch Progress", leave=False)
    
    for batch in loop:            
        inputs = batch["image"].to(device)
        labels = batch["label"].to(device)
        texts = batch["text"]

        text_inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=48,
            return_tensors="pt"
        ).to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs, text_inputs)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        
        loop.set_postfix(loss=loss.item())

    return epoch_loss / len(loader)